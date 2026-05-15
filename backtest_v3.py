"""V3: V2 + 大盤過濾 + 冷卻期

新增規則：
1. 大盤過濾：0050 收盤 < 200MA → 不開新倉，現有倉位照樣按規則出場
2. 冷卻期：同檔股票 20 個交易日內已停損過，本次不再進場
3. 弱大盤減倉：0050 < 50MA 時，單筆風險砍半 (0.375%)
"""
import backtest as bt
import backtest_v2 as v2
import pandas as pd
from collections import defaultdict

COOLDOWN_DAYS = 20

def run_v3():
    data = bt.load_all()
    enriched = {}
    for t, df in data.items():
        if len(df) < 200:
            continue
        enriched[t] = bt.add_indicators(df)
    bench_raw = enriched.pop("0050.TW")
    bench = bt.add_indicators(bench_raw)
    bench["MA200"] = bench["Close"].rolling(200).mean()
    all_dates = bench.loc[bt.START_DATE:bt.END_DATE].index

    cash = bt.INIT_CAPITAL
    equity_curve = []
    positions = {}
    trades = []
    last_stop_date: dict[str, pd.Timestamp] = {}  # ticker -> 最近一次停損出場日

    for date in all_dates:
        # 大盤狀態
        if date in bench.index:
            b_row = bench.loc[date]
            bull_market = (not pd.isna(b_row["MA200"])) and b_row["Close"] > b_row["MA200"]
            strong_market = (not pd.isna(b_row["MA50"])) and b_row["Close"] > b_row["MA50"]
        else:
            bull_market = True
            strong_market = True

        # --- 1. 處理現有部位 ---
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
                proceeds = pos.shares * exit_price * (1 - bt.SELL_COST)
                cash += proceeds
                trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + bt.BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": "stop", "bars": pos.bars_held})
                last_stop_date[tkr] = date
                to_close.append(tkr)
                continue

            if not pos.partial_taken and 3 <= pos.bars_held <= 15:
                if row["Close"] >= pos.entry_price + bt.PARTIAL_TP_R * pos.r_value:
                    sell_shares = int(pos.shares * bt.PARTIAL_TP_FRAC / 1000) * 1000
                    if sell_shares >= 1000:
                        exit_price = row["Close"]
                        proceeds = sell_shares * exit_price * (1 - bt.SELL_COST)
                        cash += proceeds
                        trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                            "entry": pos.entry_price, "exit": exit_price, "shares": sell_shares,
                            "pnl": proceeds - sell_shares * pos.entry_price * (1 + bt.BUY_COST),
                            "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                            "reason": "partial", "bars": pos.bars_held})
                        pos.shares -= sell_shares
                        pos.partial_taken = True
                        pos.stop_price = max(pos.stop_price, pos.entry_price)

            if pos.partial_taken:
                if not pd.isna(row["MA50"]) and row["Close"] < row["MA50"]:
                    exit_price = row["Close"]
                    proceeds = pos.shares * exit_price * (1 - bt.SELL_COST)
                    cash += proceeds
                    trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                        "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                        "pnl": proceeds - pos.shares * pos.entry_price * (1 + bt.BUY_COST),
                        "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                        "reason": "trail_ma50", "bars": pos.bars_held})
                    to_close.append(tkr)
                    continue

            if pos.bars_held >= 100:
                exit_price = row["Close"]
                proceeds = pos.shares * exit_price * (1 - bt.SELL_COST)
                cash += proceeds
                trades.append({"ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + bt.BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": "time", "bars": pos.bars_held})
                to_close.append(tkr)

        for tkr in to_close:
            positions.pop(tkr, None)

        # 權益
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
        equity_curve.append({"date": date, "equity": equity, "cash": cash,
                             "positions": len(positions), "bull": bull_market})

        # --- 大盤過濾：熊市不開新倉 ---
        if not bull_market:
            continue
        if len(positions) >= bt.MAX_CONCURRENT:
            continue

        candidates = []
        for tkr, df in enriched.items():
            if tkr in positions or date not in df.index:
                continue
            # 冷卻期過濾
            if tkr in last_stop_date:
                days_since_stop = (date - last_stop_date[tkr]).days
                if days_since_stop < COOLDOWN_DAYS:
                    continue
            idx = df.index.get_loc(date)
            if idx < 2:
                continue
            today = df.iloc[idx]
            prev = df.iloc[idx - 1]
            if not bt.is_setup(prev):
                continue
            if not v2.is_breakout_v2(today, prev):
                continue
            candidates.append((tkr, today, prev))

        candidates.sort(key=lambda x: x[1]["Ret60"], reverse=True)

        risk_per_trade = bt.RISK_PER_TRADE * (1.0 if strong_market else 0.5)

        for tkr, today, prev in candidates:
            if len(positions) >= bt.MAX_CONCURRENT:
                break
            entry_price = today["Close"]
            atr = today["ATR20"]
            stop = today["Low"]
            risk_per_share = entry_price - stop
            if risk_per_share <= 0 or risk_per_share > bt.MAX_STOP_ATR * atr:
                continue
            risk_dollar = equity * risk_per_trade
            shares_by_risk = int(risk_dollar / risk_per_share / 1000) * 1000
            shares_by_pos = int(equity * bt.MAX_POS_PCT / entry_price / 1000) * 1000
            shares_by_cash = int(cash / (entry_price * (1 + bt.BUY_COST)) / 1000) * 1000
            shares = min(shares_by_risk, shares_by_pos, shares_by_cash)
            if shares < 1000:
                continue
            cost = shares * entry_price * (1 + bt.BUY_COST)
            cash -= cost
            positions[tkr] = bt.Position(
                ticker=tkr, entry_date=date, entry_price=entry_price,
                shares=shares, stop_price=stop, initial_stop=stop, r_value=risk_per_share,
            )

    eq = pd.DataFrame(equity_curve).set_index("date")
    tr = pd.DataFrame(trades)
    return eq, tr, bench_raw


if __name__ == "__main__":
    eq, tr, bench = run_v3()
    bt.report(eq, tr, bench)
    eq.to_csv("equity_curve_v3.csv")
    tr.to_csv("trades_v3.csv", index=False)
