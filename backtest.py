"""
Qullamaggie 台股版回測引擎

執行：python3 backtest.py
"""
import os
import glob
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

DATA_DIR = "data"
START_DATE = "2018-01-01"
END_DATE = "2025-12-30"
INIT_CAPITAL = 1_000_000

# 手續費 + 證交稅 + 滑價
BUY_COST  = 0.001425 + 0.002  # 手續費 + 滑價
SELL_COST = 0.001425 + 0.003 + 0.002  # 手續費 + 證交稅 + 滑價

# 策略參數
LOOKBACK_MOM_MID = 60   # 中期動能 60 日
LOOKBACK_MOM_LONG = 120 # 長期動能 120 日
MOM_MID_THR = 0.30
MOM_LONG_THR = 0.20
NEAR_HIGH_PCT = 0.85    # 距 60 日高 ≤ 15%
CONSOLIDATION_MIN_DAYS = 10
CONSOLIDATION_MAX_DRAWDOWN = 0.25
BREAKOUT_LOOKBACK = 15
VOLUME_MULT = 1.5
AVG_DOLLAR_VOL_MIN = 50_000_000  # 5 千萬台幣
MIN_PRICE = 10.0
ATR_WIN = 20

RISK_PER_TRADE = 0.0075  # 0.75%
MAX_POS_PCT = 0.25
MAX_CONCURRENT = 5
MAX_STOP_ATR = 1.5
ENTRY_ATR_FRAC = 0.66    # 進場日漲幅 ≤ 2/3 ATR

PARTIAL_TP_DAYS = 5
PARTIAL_TP_R = 1.0
PARTIAL_TP_FRAC = 1/3
TRAIL_MA = 20
MAX_HOLD_DAYS = 60


# ---------- 載入資料 ----------
def load_all():
    data = {}
    for f in glob.glob(f"{DATA_DIR}/*.pkl"):
        t = os.path.basename(f).replace(".pkl", "").replace("_", ".")
        df = pd.read_pickle(f)
        df = df[(df.index >= "2017-06-01")].copy()
        data[t] = df
    return data


# ---------- 指標計算 ----------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["DollarVol20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["High60"] = df["High"].rolling(60).max()
    df["BreakoutLevel"] = df["High"].rolling(BREAKOUT_LOOKBACK).max().shift(1)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR20"] = tr.rolling(ATR_WIN).mean()

    df["Ret60"] = df["Close"] / df["Close"].shift(LOOKBACK_MOM_MID) - 1
    df["Ret120"] = df["Close"] / df["Close"].shift(LOOKBACK_MOM_LONG) - 1
    df["TodayRet"] = df["Close"] / df["Close"].shift(1) - 1
    return df


# ---------- 訊號 ----------
def is_setup(row) -> bool:
    """是否符合強勢股 + 整理"""
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
    """突破訊號"""
    if pd.isna(prev["BreakoutLevel"]) or pd.isna(today["VolMA20"]):
        return False
    if today["Close"] <= prev["BreakoutLevel"]:
        return False
    if today["Volume"] < today["VolMA20"] * VOLUME_MULT:
        return False
    # 當日漲幅 ≤ 2/3 ATR
    if today["ATR20"] > 0 and (today["Close"] - today["Open"]) / today["ATR20"] > ENTRY_ATR_FRAC * 2:
        return False
    # 漲停過濾：當日漲幅 ≥ 9.5% 視為漲停，放棄
    if today["TodayRet"] >= 0.095:
        return False
    return True


# ---------- 部位 ----------
@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    initial_stop: float
    r_value: float   # 每股風險
    partial_taken: bool = False
    bars_held: int = 0


# ---------- 主回測 ----------
def run_backtest():
    data = load_all()
    print(f"Loaded {len(data)} tickers")

    # 加指標
    enriched = {}
    for t, df in data.items():
        if len(df) < 200:
            continue
        enriched[t] = add_indicators(df)

    # 取所有交易日（用 0050）
    bench = enriched.pop("0050.TW")
    all_dates = bench.loc[START_DATE:END_DATE].index

    cash = INIT_CAPITAL
    equity_curve = []
    positions: dict[str, Position] = {}
    trades = []

    for date in all_dates:
        # --- 1. 處理現有部位（停損 / 停利 / 出場）---
        to_close = []
        for tkr, pos in positions.items():
            df = enriched.get(tkr)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["Close"]):
                continue
            pos.bars_held += 1

            # 停損檢查（盤中觸發 stop）
            if row["Low"] <= pos.stop_price:
                exit_price = min(row["Open"], pos.stop_price) if row["Open"] < pos.stop_price else pos.stop_price
                proceeds = pos.shares * exit_price * (1 - SELL_COST)
                cash += proceeds
                trades.append({
                    "ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": "stop",
                    "bars": pos.bars_held,
                })
                to_close.append(tkr)
                continue

            # 部分停利
            if not pos.partial_taken and pos.bars_held >= 3:
                if row["Close"] >= pos.entry_price + PARTIAL_TP_R * pos.r_value:
                    sell_shares = int(pos.shares * PARTIAL_TP_FRAC)
                    if sell_shares > 0:
                        exit_price = row["Close"]
                        proceeds = sell_shares * exit_price * (1 - SELL_COST)
                        cash += proceeds
                        trades.append({
                            "ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                            "entry": pos.entry_price, "exit": exit_price, "shares": sell_shares,
                            "pnl": proceeds - sell_shares * pos.entry_price * (1 + BUY_COST),
                            "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                            "reason": "partial",
                            "bars": pos.bars_held,
                        })
                        pos.shares -= sell_shares
                        pos.partial_taken = True
                        pos.stop_price = max(pos.stop_price, pos.entry_price)  # 移至 BE

            # 移動停利：MA20 收盤跌破
            if pos.partial_taken or pos.bars_held >= 10:
                if not pd.isna(row["MA20"]) and row["Close"] < row["MA20"]:
                    exit_price = row["Close"]
                    proceeds = pos.shares * exit_price * (1 - SELL_COST)
                    cash += proceeds
                    trades.append({
                        "ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                        "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                        "pnl": proceeds - pos.shares * pos.entry_price * (1 + BUY_COST),
                        "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                        "reason": "trail_ma20",
                        "bars": pos.bars_held,
                    })
                    to_close.append(tkr)
                    continue

            # 強制出場：超過最大持有
            if pos.bars_held >= MAX_HOLD_DAYS:
                exit_price = row["Close"]
                proceeds = pos.shares * exit_price * (1 - SELL_COST)
                cash += proceeds
                trades.append({
                    "ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": "time",
                    "bars": pos.bars_held,
                })
                to_close.append(tkr)

        for tkr in to_close:
            positions.pop(tkr, None)

        # --- 2. 計算當日權益（缺失資料用最近一日填補）---
        mv = 0
        for tkr, pos in positions.items():
            df = enriched.get(tkr)
            if df is None:
                mv += pos.shares * pos.entry_price
                continue
            sub = df.loc[:date, "Close"].dropna()
            px = sub.iloc[-1] if len(sub) > 0 else pos.entry_price
            mv += pos.shares * px
        equity = cash + mv
        equity_curve.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions)})

        # --- 3. 掃描新進場 ---
        if len(positions) >= MAX_CONCURRENT:
            continue
        candidates = []
        for tkr, df in enriched.items():
            if tkr in positions:
                continue
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < 2:
                continue
            today = df.iloc[idx]
            prev = df.iloc[idx - 1]
            if not is_setup(prev):  # 用前一日判斷整理（避免未來函數）
                continue
            if not is_breakout(today, prev):
                continue
            candidates.append((tkr, today, prev))

        # 按相對強度（Ret60）排序，優先強勢
        candidates.sort(key=lambda x: x[1]["Ret60"], reverse=True)

        for tkr, today, prev in candidates:
            if len(positions) >= MAX_CONCURRENT:
                break
            # 進場價：收盤價（隔日開盤模擬偏差由滑價吸收）
            entry_price = today["Close"]
            atr = today["ATR20"]
            stop = today["Low"]
            risk_per_share = entry_price - stop
            if risk_per_share <= 0 or risk_per_share > MAX_STOP_ATR * atr:
                continue
            # 風險倉位
            risk_dollar = equity * RISK_PER_TRADE
            shares_by_risk = int(risk_dollar / risk_per_share / 1000) * 1000  # 整張
            shares_by_pos = int(equity * MAX_POS_PCT / entry_price / 1000) * 1000
            shares_by_cash = int(cash / (entry_price * (1 + BUY_COST)) / 1000) * 1000
            shares = min(shares_by_risk, shares_by_pos, shares_by_cash)
            if shares < 1000:
                continue
            cost = shares * entry_price * (1 + BUY_COST)
            cash -= cost
            positions[tkr] = Position(
                ticker=tkr, entry_date=date, entry_price=entry_price,
                shares=shares, stop_price=stop, initial_stop=stop,
                r_value=risk_per_share,
            )

    # --- 結算 ---
    eq = pd.DataFrame(equity_curve).set_index("date")
    tr = pd.DataFrame(trades)
    return eq, tr, bench


# ---------- 績效 ----------
def report(eq, tr, bench):
    print("\n" + "="*70)
    print("回測結果")
    print("="*70)

    start_eq = INIT_CAPITAL
    end_eq = eq["equity"].iloc[-1]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total_ret = end_eq / start_eq - 1
    cagr = (end_eq / start_eq) ** (1/years) - 1

    daily_ret = eq["equity"].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * math.sqrt(252) if daily_ret.std() > 0 else 0

    peak = eq["equity"].cummax()
    dd = (eq["equity"] / peak - 1)
    mdd = dd.min()

    # 基準
    b = bench.loc[eq.index[0]:eq.index[-1], "Close"]
    b_ret = b.iloc[-1] / b.iloc[0] - 1
    b_cagr = (b.iloc[-1] / b.iloc[0]) ** (1/years) - 1

    print(f"\n期間：{eq.index[0].date()} ~ {eq.index[-1].date()} ({years:.2f} 年)")
    print(f"\n【策略】")
    print(f"  期末權益：    {end_eq:>15,.0f}")
    print(f"  總報酬：      {total_ret*100:>14.2f}%")
    print(f"  CAGR：        {cagr*100:>14.2f}%")
    print(f"  Sharpe：      {sharpe:>14.2f}")
    print(f"  最大回撤：    {mdd*100:>14.2f}%")

    print(f"\n【0050 買入持有】")
    print(f"  總報酬：      {b_ret*100:>14.2f}%")
    print(f"  CAGR：        {b_cagr*100:>14.2f}%")

    if len(tr) > 0:
        # 以「完整交易」聚合 (同 ticker 同 entry_date 算一筆)
        tr["trade_id"] = tr["ticker"] + "_" + tr["entry_date"].astype(str)
        agg = tr.groupby("trade_id").agg(
            entry_date=("entry_date", "first"),
            ticker=("ticker", "first"),
            entry=("entry", "first"),
            total_pnl=("pnl", "sum"),
            total_shares=("shares", "sum"),
            avg_r=("r_multiple", "mean"),
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

        print(f"\n【交易統計】")
        print(f"  總交易筆數：  {len(agg)}")
        print(f"  勝率：        {winrate*100:>14.2f}%")
        print(f"  平均勝幅：    {avg_win*100:>14.2f}%")
        print(f"  平均敗幅：    {avg_loss*100:>14.2f}%")
        print(f"  賠率(R/R)：   {rr:>14.2f}")
        print(f"  單筆期望值：  {expectancy*100:>14.2f}%")
        print(f"  最大單筆贏：  {agg['return_pct'].max()*100:>14.2f}%  ({agg.loc[agg['return_pct'].idxmax(), 'ticker']})")
        print(f"  最大單筆輸：  {agg['return_pct'].min()*100:>14.2f}%  ({agg.loc[agg['return_pct'].idxmin(), 'ticker']})")
        print(f"  平均持有天數：{agg['bars'].mean():>14.1f}")

        # 出場原因分布
        print(f"\n【出場原因】")
        print(tr["reason"].value_counts().to_string())

        # 年度報酬
        print(f"\n【年度績效】")
        eq_yr = eq["equity"].resample("YE").last()
        eq_yr_first = eq["equity"].resample("YE").first()
        bench_yr = b.resample("YE").last()
        bench_yr_first = b.resample("YE").first()
        for yr in eq_yr.index:
            strat = eq_yr[yr] / eq_yr_first[yr] - 1
            if yr in bench_yr.index:
                bm = bench_yr[yr] / bench_yr_first[yr] - 1
                print(f"  {yr.year}：策略 {strat*100:>7.2f}%  |  0050 {bm*100:>7.2f}%")

        # Top 5 贏家
        print(f"\n【Top 5 贏家】")
        top = agg.nlargest(5, "return_pct")[["ticker", "entry_date", "return_pct", "max_r", "bars"]]
        for _, r in top.iterrows():
            print(f"  {r['ticker']:>8}  {r['entry_date'].date()}  {r['return_pct']*100:>7.2f}%  R={r['max_r']:>5.2f}  {int(r['bars'])} 日")

    # 存檔
    eq.to_csv("equity_curve.csv")
    tr.to_csv("trades.csv", index=False)
    print("\n已儲存 equity_curve.csv / trades.csv")


if __name__ == "__main__":
    eq, tr, bench = run_backtest()
    report(eq, tr, bench)
