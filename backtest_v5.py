"""
V5 完整實作：Qulla 全功能版

整合三種進場策略（並行）：
1. 日線突破（原 V4）
2. EP 情境轉折
3. 拋物線做空（小量）

新增過濾器：
- RS Rating ≥ 80（IBD 式相對強度排名）
- VCP 型態加權
- 週線上升趨勢確認
- 動態 trail（依 ADR）
- 加碼（partial 後 10MA pullback）

支援台股 / 美股，並支援 Walk-forward 切分。
"""
import os, glob, math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from qulla_indicators import (
    add_all_indicators, compute_rs_rank, detect_vcp,
    is_ep_setup, is_parabolic_short, weekly_uptrend
)

# ---------- 通用參數 ----------
DEFAULT_PARAMS = {
    "MOM_MID_THR": 0.30,
    "MOM_LONG_THR": 0.20,
    "NEAR_HIGH_PCT": 0.85,
    "VOLUME_MULT": 1.5,
    "MIN_PRICE": 5.0,
    "AVG_DOLLAR_VOL_MIN": 10_000_000,
    "RS_THRESHOLD": 80,
    "VCP_BONUS_RISK_MULT": 1.3,  # VCP 確認時加大倉位
    "WEEKLY_FILTER": True,
    "RISK_PER_TRADE": 0.015,
    "LEVERAGE": 1.3,
    "MAX_POS_PCT": 0.30,
    "MAX_CONCURRENT": 6,
    "MAX_STOP_ATR": 1.5,
    "ENTRY_ATR_FRAC": 0.66,
    "PARTIAL_TP_R": 1.0,
    "PARTIAL_TP_FRAC": 1/3,
    "MAX_HOLD_DAYS": 120,
    "COOLDOWN_DAYS": 20,
    "ADD_ON_MAX": 1,
    "ADD_ON_SIZE_FRAC": 0.5,
    "HIGH_ADR_THR": 0.05,
    "ENABLE_EP": True,
    "ENABLE_SHORT": False,  # 預設關閉，太多假訊號
    "ENABLE_ADD_ON": True,
}

US_CFG = {
    "data_dir": "data_us",
    "init_capital": 100_000,
    "buy_cost": 0.0015,
    "sell_cost": 0.0015 + 0.00003,
    "benchmark_keys": ["QQQ", "SPY"],
    "lot": 1,
    "market": "US",
    "gap_filter": 0.20,  # 日線突破日漲幅上限
}

TW_CFG = {
    "data_dir": "data",
    "init_capital": 1_000_000,
    "buy_cost": 0.001425 + 0.002,
    "sell_cost": 0.001425 + 0.003 + 0.002,
    "benchmark_keys": ["0050.TW"],
    "lot": 1000,
    "market": "TW",
    "gap_filter": 0.099,
}


def load_data(data_dir, market):
    out = {}
    for f in glob.glob(f"{data_dir}/*.pkl"):
        t = os.path.basename(f).replace(".pkl","")
        if market == "TW":
            t = t.replace("_", ".")
        df = pd.read_pickle(f)
        out[t] = df
    return out


def is_setup(row, params):
    if any(pd.isna(row[c]) for c in ["Ret60","Ret120","MA50","MA100","High60","ATR20","DollarVol20"]):
        return False
    if row["Close"] < params["MIN_PRICE"]: return False
    if row["DollarVol20"] < params["AVG_DOLLAR_VOL_MIN"]: return False
    if row["Ret60"] < params["MOM_MID_THR"]: return False
    if row["Ret120"] < params["MOM_LONG_THR"]: return False
    if row["Close"] < row["High60"] * params["NEAR_HIGH_PCT"]: return False
    if not (row["Close"] > row["MA50"] > row["MA100"]): return False
    return True


def is_breakout(today, prev, params, gap_max):
    if pd.isna(prev["BreakoutLevel"]) or pd.isna(today["VolMA20"]):
        return False
    if today["Close"] <= prev["BreakoutLevel"]:
        return False
    if today["Volume"] < today["VolMA20"] * params["VOLUME_MULT"]:
        return False
    if today["ATR20"] > 0 and (today["Close"] - today["Open"]) / today["ATR20"] > params["ENTRY_ATR_FRAC"] * 2:
        return False
    if today["TodayRet"] >= gap_max:
        return False
    return True


def pick_trail_ma(adr20, partial_taken, params):
    high_adr = (not pd.isna(adr20)) and adr20 > params["HIGH_ADR_THR"]
    if partial_taken:
        return "MA20" if high_adr else "MA50"
    else:
        return "MA10" if high_adr else None


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    initial_stop: float
    r_value: float
    adr20: float
    strategy: str  # "breakout" / "ep" / "short"
    is_short: bool = False
    partial_taken: bool = False
    bars_held: int = 0
    add_on_count: int = 0
    last_below_10ma: bool = False
    original_shares: int = 0
    has_vcp: bool = False


def run_v5(market="US", params=None, start_date="2018-01-01", end_date="2025-12-30"):
    cfg = US_CFG if market == "US" else TW_CFG
    if params is None:
        params = dict(DEFAULT_PARAMS)

    raw = load_data(cfg["data_dir"], market)
    enriched = {t: add_all_indicators(df) for t, df in raw.items() if len(df) >= 200}

    # 找基準
    bench = None
    for k in cfg["benchmark_keys"]:
        if k in enriched:
            bench = enriched.pop(k)
            break
    if bench is None:
        raise RuntimeError("Benchmark not found")

    all_dates = bench.loc[start_date:end_date].index

    # 預備 closes_dict 給 RS 排名（每月重算一次即可）
    closes_dict = {t: df["Close"] for t, df in enriched.items()}

    cash = cfg["init_capital"]
    equity_curve = []
    positions: dict[str, Position] = {}
    trades = []
    last_stop_date = {}
    rs_cache = {}
    rs_cache_date = None

    LOT = cfg["lot"]

    for date in all_dates:
        # 大盤狀態
        if date in bench.index:
            b_row = bench.loc[date]
            bull_market = (not pd.isna(b_row["MA200"])) and b_row["Close"] > b_row["MA200"]
            strong_market = (not pd.isna(b_row["MA50"])) and b_row["Close"] > b_row["MA50"]
        else:
            bull_market, strong_market = True, True

        # RS 排名快取（每 20 個交易日更新一次）
        if rs_cache_date is None or (date - rs_cache_date).days >= 20:
            rs_cache = compute_rs_rank(closes_dict, date, lookback=126)
            rs_cache_date = date

        # --- 處理現有部位 ---
        to_close = []
        for tkr, pos in positions.items():
            df = enriched.get(tkr)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["Close"]):
                continue
            pos.bars_held += 1

            sign = -1 if pos.is_short else 1

            # 停損（做多看 Low，做空看 High）
            stop_hit = (row["Low"] <= pos.stop_price) if not pos.is_short else (row["High"] >= pos.stop_price)
            if stop_hit:
                if not pos.is_short:
                    exit_price = pos.stop_price if row["Open"] >= pos.stop_price else row["Open"]
                else:
                    exit_price = pos.stop_price if row["Open"] <= pos.stop_price else row["Open"]
                proceeds = pos.shares * exit_price * (1 - cfg["sell_cost"])
                if pos.is_short:
                    # 平空：原來 cash 已記負部位市值，回補時 cash 增加 entry-exit 的差
                    pnl = pos.shares * (pos.entry_price - exit_price) - pos.shares * pos.entry_price * cfg["buy_cost"] - pos.shares * exit_price * cfg["sell_cost"]
                    cash += pos.shares * pos.entry_price + pnl  # 釋放保證金 + 盈虧
                else:
                    cash += proceeds
                    pnl = proceeds - pos.shares * pos.entry_price * (1 + cfg["buy_cost"])
                r_mult = sign * (exit_price - pos.entry_price) / pos.r_value
                trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": pnl, "r_multiple": r_mult, "reason": "stop", "bars": pos.bars_held,
                    "strategy": pos.strategy, "is_short": pos.is_short})
                last_stop_date[tkr] = date
                to_close.append(tkr)
                continue

            # Partial 停利（做多用 Close >= entry + R；做空用 Close <= entry - R）
            target_hit = (row["Close"] >= pos.entry_price + params["PARTIAL_TP_R"] * pos.r_value) if not pos.is_short \
                         else (row["Close"] <= pos.entry_price - params["PARTIAL_TP_R"] * pos.r_value)
            if not pos.partial_taken and 3 <= pos.bars_held <= 20 and target_hit:
                sell_shares = int(pos.shares * params["PARTIAL_TP_FRAC"] / LOT) * LOT if LOT > 1 else int(pos.shares * params["PARTIAL_TP_FRAC"])
                sell_shares = max(LOT, sell_shares) if LOT > 1 else max(1, sell_shares)
                if sell_shares <= pos.shares:
                    exit_price = row["Close"]
                    if pos.is_short:
                        pnl_partial = sell_shares * (pos.entry_price - exit_price) - sell_shares * pos.entry_price * cfg["buy_cost"] - sell_shares * exit_price * cfg["sell_cost"]
                        cash += sell_shares * pos.entry_price + pnl_partial
                    else:
                        proceeds = sell_shares * exit_price * (1 - cfg["sell_cost"])
                        cash += proceeds
                        pnl_partial = proceeds - sell_shares * pos.entry_price * (1 + cfg["buy_cost"])
                    r_mult = sign * (exit_price - pos.entry_price) / pos.r_value
                    trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                        "entry": pos.entry_price, "exit": exit_price, "shares": sell_shares,
                        "pnl": pnl_partial, "r_multiple": r_mult, "reason": "partial",
                        "bars": pos.bars_held, "strategy": pos.strategy, "is_short": pos.is_short})
                    pos.shares -= sell_shares
                    pos.partial_taken = True
                    if not pos.is_short:
                        pos.stop_price = max(pos.stop_price, pos.entry_price)
                    else:
                        pos.stop_price = min(pos.stop_price, pos.entry_price)

            # 加碼（僅做多 + breakout/EP）
            if (params["ENABLE_ADD_ON"] and not pos.is_short and pos.partial_taken
                    and pos.add_on_count < params["ADD_ON_MAX"]):
                ma10 = row["MA10"]
                if not pd.isna(ma10):
                    if row["Low"] <= ma10:
                        pos.last_below_10ma = True
                    elif pos.last_below_10ma and row["Close"] > ma10 and row["TodayRet"] > 0:
                        target_add = int(pos.original_shares * params["ADD_ON_SIZE_FRAC"] / LOT) * LOT if LOT > 1 else int(pos.original_shares * params["ADD_ON_SIZE_FRAC"])
                        target_add = max(LOT, target_add) if LOT > 1 else max(1, target_add)
                        add_cost = target_add * row["Close"] * (1 + cfg["buy_cost"])
                        if cash >= add_cost:
                            cash -= add_cost
                            pos.shares += target_add
                            pos.add_on_count += 1
                            pos.last_below_10ma = False
                            pos.stop_price = max(pos.stop_price, row["Low"] * 0.99)

            # 動態 trail
            trail_ma = pick_trail_ma(pos.adr20, pos.partial_taken, params)
            if trail_ma is not None and not pos.is_short:
                ma_val = row.get(trail_ma)
                if not pd.isna(ma_val) and row["Close"] < ma_val:
                    exit_price = row["Close"]
                    proceeds = pos.shares * exit_price * (1 - cfg["sell_cost"])
                    cash += proceeds
                    pnl = proceeds - pos.shares * pos.entry_price * (1 + cfg["buy_cost"])
                    trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                        "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                        "pnl": pnl, "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                        "reason": f"trail_{trail_ma.lower()}", "bars": pos.bars_held,
                        "strategy": pos.strategy, "is_short": False})
                    to_close.append(tkr)
                    continue

            # 做空 trail：用 MA20 反向 (close > MA20 出場)
            if pos.is_short and pos.partial_taken:
                if not pd.isna(row["MA20"]) and row["Close"] > row["MA20"]:
                    exit_price = row["Close"]
                    pnl = pos.shares * (pos.entry_price - exit_price) - pos.shares * pos.entry_price * cfg["buy_cost"] - pos.shares * exit_price * cfg["sell_cost"]
                    cash += pos.shares * pos.entry_price + pnl
                    trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                        "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                        "pnl": pnl, "r_multiple": -(exit_price - pos.entry_price) / pos.r_value,
                        "reason": "short_cover", "bars": pos.bars_held,
                        "strategy": pos.strategy, "is_short": True})
                    to_close.append(tkr)
                    continue

            # 強制出場
            if pos.bars_held >= params["MAX_HOLD_DAYS"]:
                exit_price = row["Close"]
                if pos.is_short:
                    pnl = pos.shares * (pos.entry_price - exit_price) - pos.shares * pos.entry_price * cfg["buy_cost"] - pos.shares * exit_price * cfg["sell_cost"]
                    cash += pos.shares * pos.entry_price + pnl
                else:
                    proceeds = pos.shares * exit_price * (1 - cfg["sell_cost"])
                    cash += proceeds
                    pnl = proceeds - pos.shares * pos.entry_price * (1 + cfg["buy_cost"])
                r_mult = sign * (exit_price - pos.entry_price) / pos.r_value
                trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": pnl, "r_multiple": r_mult, "reason": "time", "bars": pos.bars_held,
                    "strategy": pos.strategy, "is_short": pos.is_short})
                to_close.append(tkr)

        for tkr in to_close:
            positions.pop(tkr, None)

        # Mark-to-market
        mv = 0
        for tkr, pos in positions.items():
            df = enriched.get(tkr)
            if df is None:
                mv += pos.shares * pos.entry_price; continue
            sub = df.loc[:date, "Close"].dropna()
            px = sub.iloc[-1] if len(sub) > 0 else pos.entry_price
            if pos.is_short:
                mv += pos.shares * (2 * pos.entry_price - px)  # 短倉浮動價值
            else:
                mv += pos.shares * px
        equity = cash + mv
        equity_curve.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions)})

        # --- 不在多頭就跳過開新倉 ---
        if not bull_market:
            continue
        if len(positions) >= params["MAX_CONCURRENT"]:
            continue

        # --- 掃描候選 ---
        breakout_candidates = []
        ep_candidates = []
        short_candidates = []

        for tkr, df in enriched.items():
            if tkr in positions or date not in df.index:
                continue
            if tkr in last_stop_date and (date - last_stop_date[tkr]).days < params["COOLDOWN_DAYS"]:
                continue
            idx = df.index.get_loc(date)
            if idx < 252:
                continue
            today = df.iloc[idx]
            prev = df.iloc[idx - 1]

            # RS 過濾
            rs = rs_cache.get(tkr, 0)
            if rs < params["RS_THRESHOLD"]:
                continue

            # 週線過濾
            if params["WEEKLY_FILTER"] and not weekly_uptrend(today):
                continue

            # 1) 日線突破
            if is_setup(prev, params) and is_breakout(today, prev, params, cfg["gap_filter"]):
                vcp_info = detect_vcp(df, idx, window=40, min_contractions=2)
                breakout_candidates.append((tkr, today, prev, idx, rs, vcp_info))

            # 2) EP
            if params["ENABLE_EP"] and is_ep_setup(today, prev, df, idx, market=cfg["market"]):
                ep_candidates.append((tkr, today, prev, idx, rs))

            # 3) 做空
            if params["ENABLE_SHORT"] and is_parabolic_short(today, prev, df, idx):
                short_candidates.append((tkr, today, prev, idx))

        # 排序：突破依 RS、EP 依 gap 大小
        breakout_candidates.sort(key=lambda x: (x[5]["is_vcp"], x[4]), reverse=True)
        ep_candidates.sort(key=lambda x: x[1]["GapUp"], reverse=True)

        def open_long(tkr, today, prev, vcp_info=None, strategy="breakout"):
            nonlocal cash, mv
            if len(positions) >= params["MAX_CONCURRENT"]:
                return False
            entry_price = today["Close"]
            atr = today["ATR20"]
            stop = today["Low"]
            rps = entry_price - stop
            if rps <= 0 or rps > params["MAX_STOP_ATR"] * atr:
                return False
            risk_mult = params["VCP_BONUS_RISK_MULT"] if (vcp_info and vcp_info.get("is_vcp")) else 1.0
            risk_dollar = equity * params["RISK_PER_TRADE"] * risk_mult * (1.0 if strong_market else 0.5)
            shares_by_risk = (int(risk_dollar / rps / LOT) * LOT) if LOT > 1 else int(risk_dollar / rps)
            shares_by_pos = (int(equity * params["MAX_POS_PCT"] / entry_price / LOT) * LOT) if LOT > 1 else int(equity * params["MAX_POS_PCT"] / entry_price)
            buying_power = equity * params["LEVERAGE"] - mv
            shares_by_bp = (int(buying_power / (entry_price * (1 + cfg["buy_cost"])) / LOT) * LOT) if LOT > 1 else int(buying_power / (entry_price * (1 + cfg["buy_cost"])))
            shares_by_cash = (int(cash / (entry_price * (1 + cfg["buy_cost"])) / LOT) * LOT) if LOT > 1 else int(cash / (entry_price * (1 + cfg["buy_cost"])))
            shares = min(shares_by_risk, shares_by_pos, shares_by_bp, shares_by_cash)
            if (LOT > 1 and shares < LOT) or (LOT == 1 and shares < 1):
                return False
            cost = shares * entry_price * (1 + cfg["buy_cost"])
            cash -= cost
            mv += shares * entry_price
            positions[tkr] = Position(
                ticker=tkr, entry_date=date, entry_price=entry_price,
                shares=shares, stop_price=stop, initial_stop=stop, r_value=rps,
                adr20=today["ADR20"], strategy=strategy, original_shares=shares,
                has_vcp=vcp_info.get("is_vcp") if vcp_info else False)
            return True

        def open_short(tkr, today, prev):
            nonlocal cash, mv
            if len(positions) >= params["MAX_CONCURRENT"]:
                return False
            entry_price = today["Close"]
            atr = today["ATR20"]
            stop = today["High"] * 1.01  # 略高於今日高點
            rps = stop - entry_price
            if rps <= 0 or rps > params["MAX_STOP_ATR"] * atr:
                return False
            risk_dollar = equity * params["RISK_PER_TRADE"] * 0.5  # 做空只用一半風險
            shares_by_risk = (int(risk_dollar / rps / LOT) * LOT) if LOT > 1 else int(risk_dollar / rps)
            shares_by_pos = (int(equity * 0.15 / entry_price / LOT) * LOT) if LOT > 1 else int(equity * 0.15 / entry_price)
            shares = min(shares_by_risk, shares_by_pos)
            if (LOT > 1 and shares < LOT) or (LOT == 1 and shares < 1):
                return False
            margin = shares * entry_price * 0.5  # 假設 50% 保證金
            if cash < margin:
                return False
            # 做空：cash 暫時凍結保證金，盈虧後再結算
            cash -= shares * entry_price  # 凍結相當於 entry 市值
            positions[tkr] = Position(
                ticker=tkr, entry_date=date, entry_price=entry_price,
                shares=shares, stop_price=stop, initial_stop=stop, r_value=rps,
                adr20=today["ADR20"], strategy="short", is_short=True, original_shares=shares)
            return True

        # 優先順序：EP > VCP-confirmed Breakout > 一般 Breakout > Short
        for tkr, today, prev, idx, rs in ep_candidates:
            if len(positions) >= params["MAX_CONCURRENT"]: break
            open_long(tkr, today, prev, vcp_info=None, strategy="ep")
        for tkr, today, prev, idx, rs, vcp_info in breakout_candidates:
            if len(positions) >= params["MAX_CONCURRENT"]: break
            open_long(tkr, today, prev, vcp_info=vcp_info, strategy="breakout")
        for tkr, today, prev, idx in short_candidates:
            if len(positions) >= params["MAX_CONCURRENT"]: break
            open_short(tkr, today, prev)

    eq = pd.DataFrame(equity_curve).set_index("date")
    tr = pd.DataFrame(trades)
    return eq, tr, bench


def report(eq, tr, bench, label, init_capital):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    end_eq = eq["equity"].iloc[-1]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (end_eq / init_capital) ** (1/years) - 1
    daily_ret = eq["equity"].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * math.sqrt(252) if daily_ret.std() > 0 else 0
    mdd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    b = bench.loc[eq.index[0]:eq.index[-1], "Close"]
    b_cagr = (b.iloc[-1] / b.iloc[0]) ** (1/years) - 1

    print(f"期間 {eq.index[0].date()} → {eq.index[-1].date()} ({years:.2f}y)")
    print(f"期末權益 {end_eq:,.0f}  CAGR {cagr*100:.2f}%  MDD {mdd*100:.2f}%  Sharpe {sharpe:.2f}")
    print(f"基準 CAGR {b_cagr*100:.2f}%  vs 基準 {(cagr-b_cagr)*100:+.2f}pp")

    if len(tr) == 0:
        print("無交易")
        return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe, "b_cagr": b_cagr, "trades": 0}

    tr["trade_id"] = tr["ticker"] + "_" + tr["entry_date"].astype(str)
    agg = tr.groupby("trade_id").agg(
        entry_date=("entry_date","first"), ticker=("ticker","first"),
        entry=("entry","first"), total_pnl=("pnl","sum"),
        total_shares=("shares","sum"), max_r=("r_multiple","max"),
        bars=("bars","max"), strategy=("strategy","first"), is_short=("is_short","first"),
    ).reset_index()
    agg["return_pct"] = agg.apply(lambda r: r["total_pnl"] / (r["entry"] * r["total_shares"]) if r["total_shares"]>0 else 0, axis=1)
    wins = agg[agg["total_pnl"] > 0]
    winrate = len(wins) / len(agg) if len(agg) else 0
    avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
    avg_loss = agg[agg["total_pnl"] <= 0]["return_pct"].mean() if len(agg) > len(wins) else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    exp = winrate * avg_win + (1-winrate) * avg_loss

    print(f"交易 {len(agg)}  勝率 {winrate*100:.1f}%  R/R {rr:.2f}  期望值 {exp*100:.2f}%")

    # 拆分策略類型
    print("\n依策略類型：")
    for strat, sub in agg.groupby("strategy"):
        wr = (sub["total_pnl"] > 0).mean()
        avg_ret = sub["return_pct"].mean()
        print(f"  {strat:>10}: {len(sub)} 筆  勝率 {wr*100:.1f}%  平均報酬 {avg_ret*100:+.2f}%")

    # VCP 標記效果
    if "has_vcp" not in agg.columns:
        # 從 tr 接回 VCP 標記
        pass

    print("\n年度績效：")
    eq_yr = eq["equity"].resample("YE").last()
    eq_yr_first = eq["equity"].resample("YE").first()
    b_yr = b.resample("YE").last()
    b_yr_first = b.resample("YE").first()
    for yr in eq_yr.index:
        s = eq_yr[yr] / eq_yr_first[yr] - 1
        bm = b_yr[yr] / b_yr_first[yr] - 1 if yr in b_yr.index else 0
        print(f"  {yr.year}: 策略 {s*100:>7.2f}%  基準 {bm*100:>7.2f}%")

    print("\nTop 10 贏家：")
    for _, r in agg.nlargest(10, "return_pct").iterrows():
        short_tag = "S" if r["is_short"] else "L"
        print(f"  [{r['strategy']:>8} {short_tag}] {r['ticker']:>8} {r['entry_date'].date()} {r['return_pct']*100:>7.2f}% R={r['max_r']:>5.2f}")

    return {"label": label, "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
            "trades": len(agg), "winrate": winrate, "rr": rr, "exp": exp,
            "end_eq": end_eq, "b_cagr": b_cagr}


if __name__ == "__main__":
    import sys
    market = sys.argv[1] if len(sys.argv) > 1 else "US"

    print(f">>> V5 完整版 - {market}")
    params = dict(DEFAULT_PARAMS)
    eq, tr, bench = run_v5(market=market, params=params)
    cfg = US_CFG if market == "US" else TW_CFG
    r = report(eq, tr, bench, f"{market} V5 (full)", cfg["init_capital"])
    eq.to_csv(f"{market.lower()}_equity_v5.csv")
    tr.to_csv(f"{market.lower()}_trades_v5.csv", index=False)
