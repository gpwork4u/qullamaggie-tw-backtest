"""台股 V4 完整版（加碼 + 動態 trail + 高風險 + 槓桿）"""
import os, glob, math
import pandas as pd
from dataclasses import dataclass

DATA_DIR = "data"
START_DATE = "2018-01-01"
END_DATE = "2025-12-30"
INIT_CAPITAL = 1_000_000  # NT$

# 台股成本（含證交稅）
BUY_COST  = 0.001425 + 0.002
SELL_COST = 0.001425 + 0.003 + 0.002

MOM_MID_THR = 0.30
MOM_LONG_THR = 0.20
NEAR_HIGH_PCT = 0.85
BREAKOUT_LOOKBACK = 15
VOLUME_MULT = 1.5
AVG_DOLLAR_VOL_MIN = 50_000_000  # 5 千萬台幣
MIN_PRICE = 10.0
ATR_WIN = 20

RISK_PER_TRADE = 0.015
LEVERAGE = 1.3
MAX_POS_PCT = 0.30
MAX_CONCURRENT = 6
MAX_STOP_ATR = 1.5
ENTRY_ATR_FRAC = 0.66

ADD_ON_MAX = 1
ADD_ON_SIZE_FRAC = 0.5
HIGH_ADR_THR = 0.05

PARTIAL_TP_R = 1.0
PARTIAL_TP_FRAC = 1/3
MAX_HOLD_DAYS = 120
COOLDOWN_DAYS = 20

LOT = 1000  # 台股一張 = 1000 股


def load_all():
    return {os.path.basename(f).replace(".pkl","").replace("_","."): pd.read_pickle(f)
            for f in glob.glob(f"{DATA_DIR}/*.pkl")}


def add_indicators(df):
    df = df.copy()
    for w in [10, 20, 50, 100, 200]:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["DollarVol20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["High60"] = df["High"].rolling(60).max()
    df["BreakoutLevel"] = df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR20"] = tr.rolling(ATR_WIN).mean()
    df["ADR20"] = ((df["High"] - df["Low"]) / df["Close"]).rolling(20).mean()
    df["Ret60"] = df["Close"] / df["Close"].shift(60) - 1
    df["Ret120"] = df["Close"] / df["Close"].shift(120) - 1
    df["TodayRet"] = df["Close"] / df["Close"].shift(1) - 1
    return df


def is_setup(row):
    if any(pd.isna(row[c]) for c in ["Ret60","Ret120","MA50","MA100","High60","ATR20","DollarVol20"]):
        return False
    if row["Close"] < MIN_PRICE: return False
    if row["DollarVol20"] < AVG_DOLLAR_VOL_MIN: return False
    if row["Ret60"] < MOM_MID_THR: return False
    if row["Ret120"] < MOM_LONG_THR: return False
    if row["Close"] < row["High60"] * NEAR_HIGH_PCT: return False
    if not (row["Close"] > row["MA50"] > row["MA100"]): return False
    return True


def is_breakout(today, prev):
    if pd.isna(prev["BreakoutLevel"]) or pd.isna(today["VolMA20"]):
        return False
    if today["Close"] <= prev["BreakoutLevel"]:
        return False
    if today["Volume"] < today["VolMA20"] * VOLUME_MULT:
        return False
    if today["ATR20"] > 0 and (today["Close"] - today["Open"]) / today["ATR20"] > ENTRY_ATR_FRAC * 2:
        return False
    # 台股漲停過濾 (9.9%+)
    if today["TodayRet"] >= 0.099:
        return False
    return True


def pick_trail_ma(adr20, partial_taken):
    high_adr = (not pd.isna(adr20)) and adr20 > HIGH_ADR_THR
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
    partial_taken: bool = False
    bars_held: int = 0
    add_on_count: int = 0
    last_below_10ma: bool = False
    original_shares: int = 0


def run_v4(use_addon=True, leverage=LEVERAGE, risk_per_trade=RISK_PER_TRADE):
    data = load_all()
    enriched = {t: add_indicators(df) for t, df in data.items() if len(df) >= 200}

    bench = enriched.pop("0050.TW")
    all_dates = bench.loc[START_DATE:END_DATE].index

    cash = INIT_CAPITAL
    equity_curve = []
    positions: dict[str, Position] = {}
    trades = []
    last_stop_date = {}

    for date in all_dates:
        if date in bench.index:
            b_row = bench.loc[date]
            bull_market = (not pd.isna(b_row["MA200"])) and b_row["Close"] > b_row["MA200"]
            strong_market = (not pd.isna(b_row["MA50"])) and b_row["Close"] > b_row["MA50"]
        else:
            bull_market, strong_market = True, True

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

            if not pos.partial_taken and 3 <= pos.bars_held <= 20:
                if row["Close"] >= pos.entry_price + PARTIAL_TP_R * pos.r_value:
                    sell_shares = int(pos.shares * PARTIAL_TP_FRAC / LOT) * LOT
                    if sell_shares >= LOT:
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

            # 加碼
            if use_addon and pos.partial_taken and pos.add_on_count < ADD_ON_MAX:
                ma10 = row["MA10"]
                if not pd.isna(ma10):
                    if row["Low"] <= ma10:
                        pos.last_below_10ma = True
                    elif pos.last_below_10ma and row["Close"] > ma10 and row["TodayRet"] > 0:
                        add_shares_target = int(pos.original_shares * ADD_ON_SIZE_FRAC / LOT) * LOT
                        if add_shares_target >= LOT:
                            add_cost = add_shares_target * row["Close"] * (1 + BUY_COST)
                            mv_check = sum(p.shares * (enriched[t].loc[:date,"Close"].dropna().iloc[-1] if t in enriched else p.entry_price)
                                           for t, p in positions.items())
                            equity_check = cash + mv_check
                            buying_power = equity_check * leverage - mv_check
                            if cash >= add_cost and add_cost <= buying_power:
                                cash -= add_cost
                                pos.shares += add_shares_target
                                pos.add_on_count += 1
                                pos.last_below_10ma = False
                                pos.stop_price = max(pos.stop_price, row["Low"] * 0.99)

            # 動態 trail
            trail_ma = pick_trail_ma(pos.adr20, pos.partial_taken)
            if trail_ma is not None:
                ma_val = row.get(trail_ma)
                if not pd.isna(ma_val) and row["Close"] < ma_val:
                    exit_price = row["Close"]
                    proceeds = pos.shares * exit_price * (1 - SELL_COST)
                    cash += proceeds
                    trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                        "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                        "pnl": proceeds - pos.shares * pos.entry_price * (1 + BUY_COST),
                        "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                        "reason": f"trail_{trail_ma.lower()}", "bars": pos.bars_held})
                    to_close.append(tkr)
                    continue

            if pos.bars_held >= MAX_HOLD_DAYS:
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

        mv = 0
        for tkr, pos in positions.items():
            df = enriched.get(tkr)
            if df is None:
                mv += pos.shares * pos.entry_price; continue
            sub = df.loc[:date, "Close"].dropna()
            px = sub.iloc[-1] if len(sub) > 0 else pos.entry_price
            mv += pos.shares * px
        equity = cash + mv
        equity_curve.append({"date": date, "equity": equity, "cash": cash,
                             "positions": len(positions), "leverage": mv/equity if equity > 0 else 0})

        if not bull_market:
            continue
        if len(positions) >= MAX_CONCURRENT:
            continue

        candidates = []
        for tkr, df in enriched.items():
            if tkr in positions or date not in df.index:
                continue
            if tkr in last_stop_date and (date - last_stop_date[tkr]).days < COOLDOWN_DAYS:
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
        risk = risk_per_trade * (1.0 if strong_market else 0.5)

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
            shares_by_risk = int(risk_dollar / rps / LOT) * LOT
            shares_by_pos = int(equity * MAX_POS_PCT / entry_price / LOT) * LOT
            buying_power = equity * leverage - mv
            shares_by_bp = int(buying_power / (entry_price * (1 + BUY_COST)) / LOT) * LOT if buying_power > 0 else 0
            shares_by_cash = int(cash / (entry_price * (1 + BUY_COST)) / LOT) * LOT
            shares = min(shares_by_risk, shares_by_pos, shares_by_bp, shares_by_cash)
            if shares < LOT:
                continue
            cost = shares * entry_price * (1 + BUY_COST)
            cash -= cost
            mv += shares * entry_price
            positions[tkr] = Position(
                ticker=tkr, entry_date=date, entry_price=entry_price,
                shares=shares, stop_price=stop, initial_stop=stop, r_value=rps,
                adr20=today["ADR20"], original_shares=shares)

    eq = pd.DataFrame(equity_curve).set_index("date")
    tr = pd.DataFrame(trades)
    return eq, tr, bench


def report(eq, tr, bench, label, init=INIT_CAPITAL):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    end_eq = eq["equity"].iloc[-1]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (end_eq / init) ** (1/years) - 1
    daily_ret = eq["equity"].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * math.sqrt(252) if daily_ret.std() > 0 else 0
    mdd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    b = bench.loc[eq.index[0]:eq.index[-1], "Close"]
    b_cagr = (b.iloc[-1] / b.iloc[0]) ** (1/years) - 1

    print(f"期末權益: NT${end_eq:,.0f}  CAGR {cagr*100:.2f}%  MDD {mdd*100:.2f}%  Sharpe {sharpe:.2f}")
    print(f"基準 0050 CAGR: {b_cagr*100:.2f}%")
    print(f"平均槓桿: {eq['leverage'].mean():.2f}x  最高槓桿: {eq['leverage'].max():.2f}x")

    if len(tr) == 0:
        return {}
    tr["trade_id"] = tr["ticker"] + "_" + tr["entry_date"].astype(str)
    agg = tr.groupby("trade_id").agg(
        entry_date=("entry_date","first"), ticker=("ticker","first"),
        entry=("entry","first"), total_pnl=("pnl","sum"),
        total_shares=("shares","sum"), max_r=("r_multiple","max"),
        bars=("bars","max"),
    ).reset_index()
    agg["return_pct"] = agg["total_pnl"] / (agg["entry"] * agg["total_shares"])
    wins = agg[agg["total_pnl"] > 0]
    winrate = len(wins) / len(agg)
    avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
    avg_loss = agg[agg["total_pnl"] <= 0]["return_pct"].mean() if len(agg) > len(wins) else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    exp = winrate * avg_win + (1-winrate) * avg_loss

    print(f"交易 {len(agg)}  勝率 {winrate*100:.1f}%  勝幅 {avg_win*100:.2f}%  敗幅 {avg_loss*100:.2f}%")
    print(f"R/R {rr:.2f}  期望值 {exp*100:.2f}%  平均持有 {agg['bars'].mean():.1f} 日")
    print(f"最大贏 {agg['return_pct'].max()*100:.2f}% ({agg.loc[agg['return_pct'].idxmax(),'ticker']})")
    print(f"最大輸 {agg['return_pct'].min()*100:.2f}% ({agg.loc[agg['return_pct'].idxmin(),'ticker']})")

    print("年度績效：")
    eq_yr = eq["equity"].resample("YE").last()
    eq_yr_first = eq["equity"].resample("YE").first()
    b_yr = b.resample("YE").last()
    b_yr_first = b.resample("YE").first()
    for yr in eq_yr.index:
        s = eq_yr[yr] / eq_yr_first[yr] - 1
        bm = b_yr[yr] / b_yr_first[yr] - 1 if yr in b_yr.index else 0
        print(f"  {yr.year}: 策略 {s*100:>7.2f}%  0050 {bm*100:>7.2f}%")

    print("Top 10 贏家：")
    for _, r in agg.nlargest(10, "return_pct").iterrows():
        print(f"  {r['ticker']:>10} {r['entry_date'].date()} {r['return_pct']*100:>7.2f}% R={r['max_r']:>5.2f} {int(r['bars'])}d")

    return {"label": label, "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
            "trades": len(agg), "winrate": winrate, "rr": rr, "exp": exp,
            "end_eq": end_eq, "b_cagr": b_cagr}


if __name__ == "__main__":
    print(">>> TW V4 完整版")
    eq, tr, bench = run_v4(use_addon=True, leverage=1.3, risk_per_trade=0.015)
    r4 = report(eq, tr, bench, "TW V4 (full)")
    eq.to_csv("tw_equity_v4.csv")
    tr.to_csv("tw_trades_v4.csv", index=False)

    print("\n>>> TW V4 (no add-on)")
    eq_no, tr_no, _ = run_v4(use_addon=False, leverage=1.3, risk_per_trade=0.015)
    r_no = report(eq_no, tr_no, bench, "TW V4 (no add-on)")

    print("\n>>> TW V4 (aggressive: 1.5x + 2%)")
    eq_x, tr_x, _ = run_v4(use_addon=True, leverage=1.5, risk_per_trade=0.02)
    r_x = report(eq_x, tr_x, bench, "TW V4 (aggressive)")

    print("\n" + "="*70 + "\n彙總")
    print(f"{'版本':>25} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'交易':>5} {'勝率':>6} {'R/R':>5}")
    for r in [r4, r_no, r_x]:
        if r:
            print(f"{r['label']:>25} {r['cagr']*100:>7.2f}% {r['mdd']*100:>7.2f}% "
                  f"{r['sharpe']:>7.2f} {r['trades']:>5} {r['winrate']*100:>5.1f}% {r['rr']:>4.2f}")
    print(f"\n0050 buy-and-hold CAGR: {r4['b_cagr']*100:.2f}%")
