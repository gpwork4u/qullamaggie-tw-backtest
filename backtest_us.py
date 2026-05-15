"""美股版 Qullamaggie 回測（含 V1/V2/V3 三版）"""
import os, glob, math
import pandas as pd
import numpy as np
from dataclasses import dataclass

DATA_DIR = "data_us"
START_DATE = "2018-01-01"
END_DATE = "2025-12-30"
INIT_CAPITAL = 100_000   # USD

# 美股成本: 零佣金 + SEC fee (賣方) + 滑價
BUY_COST  = 0.0015           # 0.15% 滑價
SELL_COST = 0.0015 + 0.00003 # 滑價 + SEC fee 極小

# 策略參數（與台股版本一致）
LOOKBACK_MOM_MID = 60
LOOKBACK_MOM_LONG = 120
MOM_MID_THR = 0.30
MOM_LONG_THR = 0.20
NEAR_HIGH_PCT = 0.85
BREAKOUT_LOOKBACK = 15
VOLUME_MULT = 1.5
AVG_DOLLAR_VOL_MIN = 10_000_000  # $10M
MIN_PRICE = 5.0
ATR_WIN = 20

RISK_PER_TRADE = 0.0075
MAX_POS_PCT = 0.25
MAX_CONCURRENT = 5
MAX_STOP_ATR = 1.5
ENTRY_ATR_FRAC = 0.66

PARTIAL_TP_R = 1.0
PARTIAL_TP_FRAC = 1/3
MAX_HOLD_DAYS_V1 = 60
MAX_HOLD_DAYS_V2 = 100
COOLDOWN_DAYS = 20


def load_all():
    data = {}
    for f in glob.glob(f"{DATA_DIR}/*.pkl"):
        t = os.path.basename(f).replace(".pkl", "")
        df = pd.read_pickle(f)
        data[t] = df
    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["DollarVol20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["High60"] = df["High"].rolling(60).max()
    df["BreakoutLevel"] = df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR20"] = tr.rolling(ATR_WIN).mean()
    df["Ret60"] = df["Close"] / df["Close"].shift(LOOKBACK_MOM_MID) - 1
    df["Ret120"] = df["Close"] / df["Close"].shift(LOOKBACK_MOM_LONG) - 1
    df["TodayRet"] = df["Close"] / df["Close"].shift(1) - 1
    return df


def is_setup(row) -> bool:
    if any(pd.isna(row[c]) for c in ["Ret60", "Ret120", "MA50", "MA100", "High60", "ATR20", "DollarVol20"]):
        return False
    if row["Close"] < MIN_PRICE: return False
    if row["DollarVol20"] < AVG_DOLLAR_VOL_MIN: return False
    if row["Ret60"] < MOM_MID_THR: return False
    if row["Ret120"] < MOM_LONG_THR: return False
    if row["Close"] < row["High60"] * NEAR_HIGH_PCT: return False
    if not (row["Close"] > row["MA50"] > row["MA100"]): return False
    return True


def is_breakout(today, prev) -> bool:
    if pd.isna(prev["BreakoutLevel"]) or pd.isna(today["VolMA20"]):
        return False
    if today["Close"] <= prev["BreakoutLevel"]:
        return False
    if today["Volume"] < today["VolMA20"] * VOLUME_MULT:
        return False
    if today["ATR20"] > 0 and (today["Close"] - today["Open"]) / today["ATR20"] > ENTRY_ATR_FRAC * 2:
        return False
    # 美股無漲跌停限制，但仍過濾單日暴漲（買在山頂）
    if today["TodayRet"] >= 0.20:
        return False
    return True


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    initial_stop: float
    r_value: float
    partial_taken: bool = False
    bars_held: int = 0


def run(version: str = "v3"):
    """version: v1 (strict MA20), v2 (loose MA50), v3 (v2 + market filter + cooldown)"""
    data = load_all()
    enriched = {}
    for t, df in data.items():
        if len(df) < 200:
            continue
        enriched[t] = add_indicators(df)

    spy = enriched.pop("SPY", None)
    qqq = enriched.pop("QQQ", None)
    bench = qqq if qqq is not None else spy
    all_dates = bench.loc[START_DATE:END_DATE].index

    cash = INIT_CAPITAL
    equity_curve = []
    positions: dict[str, Position] = {}
    trades = []
    last_stop_date: dict[str, pd.Timestamp] = {}

    use_market_filter = (version == "v3")
    use_cooldown = (version == "v3")
    trail_after_partial_ma = "MA50" if version in ("v2", "v3") else "MA20"
    early_trail_v1 = (version == "v1")
    max_hold = MAX_HOLD_DAYS_V1 if version == "v1" else MAX_HOLD_DAYS_V2

    for date in all_dates:
        # 大盤狀態
        if date in bench.index:
            b_row = bench.loc[date]
            bull_market = (not pd.isna(b_row["MA200"])) and b_row["Close"] > b_row["MA200"]
            strong_market = (not pd.isna(b_row["MA50"])) and b_row["Close"] > b_row["MA50"]
        else:
            bull_market = True
            strong_market = True

        # 處理現有部位
        to_close = []
        for tkr, pos in positions.items():
            df = enriched.get(tkr)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["Close"]):
                continue
            pos.bars_held += 1

            if row["Low"] <= pos.stop_price:
                exit_price = pos.stop_price if row["Open"] >= pos.stop_price else row["Open"]
                proceeds = pos.shares * exit_price * (1 - SELL_COST)
                cash += proceeds
                trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": "stop", "bars": pos.bars_held})
                last_stop_date[tkr] = date
                to_close.append(tkr)
                continue

            if not pos.partial_taken and 3 <= pos.bars_held <= 15:
                if row["Close"] >= pos.entry_price + PARTIAL_TP_R * pos.r_value:
                    sell_shares = int(pos.shares * PARTIAL_TP_FRAC)
                    if sell_shares >= 1:
                        exit_price = row["Close"]
                        proceeds = sell_shares * exit_price * (1 - SELL_COST)
                        cash += proceeds
                        trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                            "entry": pos.entry_price, "exit": exit_price, "shares": sell_shares,
                            "pnl": proceeds - sell_shares * pos.entry_price * (1 + BUY_COST),
                            "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                            "reason": "partial", "bars": pos.bars_held})
                        pos.shares -= sell_shares
                        pos.partial_taken = True
                        pos.stop_price = max(pos.stop_price, pos.entry_price)

            # Trailing: V1 早期就 MA20 trail; V2/V3 只在 partial 後啟動
            trail_now = False
            if early_trail_v1 and (pos.partial_taken or pos.bars_held >= 10):
                if not pd.isna(row["MA20"]) and row["Close"] < row["MA20"]:
                    trail_now = True
                    reason = "trail_ma20"
            elif (not early_trail_v1) and pos.partial_taken:
                if not pd.isna(row[trail_after_partial_ma]) and row["Close"] < row[trail_after_partial_ma]:
                    trail_now = True
                    reason = f"trail_{trail_after_partial_ma.lower()}"
            if trail_now:
                exit_price = row["Close"]
                proceeds = pos.shares * exit_price * (1 - SELL_COST)
                cash += proceeds
                trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": reason, "bars": pos.bars_held})
                to_close.append(tkr)
                continue

            if pos.bars_held >= max_hold:
                exit_price = row["Close"]
                proceeds = pos.shares * exit_price * (1 - SELL_COST)
                cash += proceeds
                trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": "time", "bars": pos.bars_held})
                to_close.append(tkr)

        for tkr in to_close:
            positions.pop(tkr, None)

        # mark-to-market
        mv = 0
        for tkr, pos in positions.items():
            df = enriched.get(tkr)
            if df is None:
                mv += pos.shares * pos.entry_price; continue
            sub = df.loc[:date, "Close"].dropna()
            px = sub.iloc[-1] if len(sub) > 0 else pos.entry_price
            mv += pos.shares * px
        equity = cash + mv
        equity_curve.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions)})

        if use_market_filter and not bull_market:
            continue
        if len(positions) >= MAX_CONCURRENT:
            continue

        candidates = []
        for tkr, df in enriched.items():
            if tkr in positions or date not in df.index:
                continue
            if use_cooldown and tkr in last_stop_date:
                if (date - last_stop_date[tkr]).days < COOLDOWN_DAYS:
                    continue
            idx = df.index.get_loc(date)
            if idx < 2:
                continue
            today = df.iloc[idx]
            prev = df.iloc[idx - 1]
            if not is_setup(prev):
                continue
            if not is_breakout(today, prev):
                continue
            candidates.append((tkr, today, prev))

        candidates.sort(key=lambda x: x[1]["Ret60"], reverse=True)
        risk = RISK_PER_TRADE * (1.0 if (not use_market_filter or strong_market) else 0.5)

        for tkr, today, prev in candidates:
            if len(positions) >= MAX_CONCURRENT:
                break
            entry_price = today["Close"]
            atr = today["ATR20"]
            stop = today["Low"]
            rps = entry_price - stop
            if rps <= 0 or rps > MAX_STOP_ATR * atr:
                continue
            risk_dollar = equity * risk
            shares_by_risk = int(risk_dollar / rps)
            shares_by_pos = int(equity * MAX_POS_PCT / entry_price)
            shares_by_cash = int(cash / (entry_price * (1 + BUY_COST)))
            shares = min(shares_by_risk, shares_by_pos, shares_by_cash)
            if shares < 1:
                continue
            cost = shares * entry_price * (1 + BUY_COST)
            cash -= cost
            positions[tkr] = Position(
                ticker=tkr, entry_date=date, entry_price=entry_price,
                shares=shares, stop_price=stop, initial_stop=stop, r_value=rps)

    eq = pd.DataFrame(equity_curve).set_index("date")
    tr = pd.DataFrame(trades)
    return eq, tr, bench


def report(eq, tr, bench, label, init_capital=INIT_CAPITAL):
    print("\n" + "="*70)
    print(f"美股回測 — {label}")
    print("="*70)

    end_eq = eq["equity"].iloc[-1]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total_ret = end_eq / init_capital - 1
    cagr = (end_eq / init_capital) ** (1/years) - 1
    daily_ret = eq["equity"].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * math.sqrt(252) if daily_ret.std() > 0 else 0
    peak = eq["equity"].cummax()
    mdd = (eq["equity"] / peak - 1).min()
    b = bench.loc[eq.index[0]:eq.index[-1], "Close"]
    b_cagr = (b.iloc[-1] / b.iloc[0]) ** (1/years) - 1

    print(f"期間：{eq.index[0].date()} ~ {eq.index[-1].date()} ({years:.2f} 年)")
    print(f"期末權益：${end_eq:>14,.0f}  總報酬 {total_ret*100:>7.2f}%  CAGR {cagr*100:>6.2f}%")
    print(f"Sharpe {sharpe:.2f}  MDD {mdd*100:.2f}%")
    print(f"基準 QQQ CAGR: {b_cagr*100:.2f}%")

    if len(tr) > 0:
        tr["trade_id"] = tr["ticker"] + "_" + tr["entry_date"].astype(str)
        agg = tr.groupby("trade_id").agg(
            entry_date=("entry_date", "first"),
            ticker=("ticker", "first"),
            entry=("entry", "first"),
            total_pnl=("pnl", "sum"),
            total_shares=("shares", "sum"),
            max_r=("r_multiple", "max"),
            bars=("bars", "max"),
        ).reset_index()
        agg["return_pct"] = agg["total_pnl"] / (agg["entry"] * agg["total_shares"])
        wins = agg[agg["total_pnl"] > 0]
        losses = agg[agg["total_pnl"] <= 0]
        winrate = len(wins) / len(agg) if len(agg) > 0 else 0
        avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
        avg_loss = losses["return_pct"].mean() if len(losses) > 0 else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        expectancy = winrate * avg_win + (1 - winrate) * avg_loss

        print(f"\n交易 {len(agg)} 筆 | 勝率 {winrate*100:.1f}% | 勝幅 {avg_win*100:.2f}% | 敗幅 {avg_loss*100:.2f}%")
        print(f"賠率 R/R {rr:.2f} | 期望值 {expectancy*100:.2f}% | 平均持有 {agg['bars'].mean():.1f} 日")
        print(f"最大單筆贏 {agg['return_pct'].max()*100:.2f}%  ({agg.loc[agg['return_pct'].idxmax(), 'ticker']})")
        print(f"最大單筆輸 {agg['return_pct'].min()*100:.2f}%  ({agg.loc[agg['return_pct'].idxmin(), 'ticker']})")

        print("\n年度績效：")
        eq_yr = eq["equity"].resample("YE").last()
        eq_yr_first = eq["equity"].resample("YE").first()
        b_yr = b.resample("YE").last()
        b_yr_first = b.resample("YE").first()
        for yr in eq_yr.index:
            s = eq_yr[yr] / eq_yr_first[yr] - 1
            bm = b_yr.get(yr, None)
            if bm is not None:
                bm_r = b_yr[yr] / b_yr_first[yr] - 1
                print(f"  {yr.year}：策略 {s*100:>7.2f}%  |  QQQ {bm_r*100:>7.2f}%")

        print("\nTop 5 贏家：")
        for _, r in agg.nlargest(5, "return_pct").iterrows():
            print(f"  {r['ticker']:>6}  {r['entry_date'].date()}  {r['return_pct']*100:>7.2f}%  R={r['max_r']:>5.2f}  {int(r['bars'])} 日")

    return {"label": label, "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
            "trades": len(agg) if len(tr) > 0 else 0,
            "winrate": winrate if len(tr) > 0 else 0,
            "rr": rr if len(tr) > 0 else 0,
            "expectancy": expectancy if len(tr) > 0 else 0,
            "end_eq": end_eq, "b_cagr": b_cagr}


if __name__ == "__main__":
    results = []
    for v in ["v1", "v2", "v3"]:
        print(f"\n>>> Running US {v.upper()}...")
        eq, tr, bench = run(v)
        r = report(eq, tr, bench, f"US-{v.upper()}")
        results.append(r)
        eq.to_csv(f"us_equity_{v}.csv")
        tr.to_csv(f"us_trades_{v}.csv", index=False)

    print("\n" + "="*70)
    print("彙總")
    print("="*70)
    print(f"{'版本':>8} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'交易':>6} {'勝率':>7} {'R/R':>6} {'期望值':>8}")
    for r in results:
        print(f"{r['label']:>8} {r['cagr']*100:>7.2f}% {r['mdd']*100:>7.2f}% {r['sharpe']:>7.2f} "
              f"{r['trades']:>6} {r['winrate']*100:>6.1f}% {r['rr']:>5.2f} {r['expectancy']*100:>7.2f}%")
    print(f"\nQQQ buy-and-hold CAGR: {results[0]['b_cagr']*100:.2f}%")
