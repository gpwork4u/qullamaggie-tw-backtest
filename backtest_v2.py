"""V2: 更貼近 Qulla 原意的調整版

主要改動：
1. 漲停過濾放寬至 9.9% (台股漲停 10%)
2. partial 前只用初始停損，不啟動 MA20 trail
3. partial 後用 MA50（而非 MA20）做 trail，給趨勢更多空間
4. partial 條件放寬：T+10 內達 1R 即可（原本 T+5）
"""
import backtest as bt
import pandas as pd

# Monkey-patch 關鍵函數
def is_breakout_v2(today, prev) -> bool:
    if pd.isna(prev["BreakoutLevel"]) or pd.isna(today["VolMA20"]):
        return False
    if today["Close"] <= prev["BreakoutLevel"]:
        return False
    if today["Volume"] < today["VolMA20"] * bt.VOLUME_MULT:
        return False
    if today["ATR20"] > 0 and (today["Close"] - today["Open"]) / today["ATR20"] > bt.ENTRY_ATR_FRAC * 2:
        return False
    # 放寬：只過濾真正漲停 (9.9%+)
    if today["TodayRet"] >= 0.099:
        return False
    return True


def run_v2():
    data = bt.load_all()
    enriched = {}
    for t, df in data.items():
        if len(df) < 200:
            continue
        enriched[t] = bt.add_indicators(df)
    bench = enriched.pop("0050.TW")
    all_dates = bench.loc[bt.START_DATE:bt.END_DATE].index

    cash = bt.INIT_CAPITAL
    equity_curve = []
    positions = {}
    trades = []

    for date in all_dates:
        to_close = []
        for tkr, pos in positions.items():
            df = enriched.get(tkr)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["Close"]):
                continue
            pos.bars_held += 1

            # 停損
            if row["Low"] <= pos.stop_price:
                exit_price = pos.stop_price if row["Open"] >= pos.stop_price else row["Open"]
                proceeds = pos.shares * exit_price * (1 - bt.SELL_COST)
                cash += proceeds
                trades.append({
                    "ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + bt.BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": "stop", "bars": pos.bars_held,
                })
                to_close.append(tkr)
                continue

            # Partial: T+3 ~ T+10 內達 1R
            if not pos.partial_taken and pos.bars_held >= 3 and pos.bars_held <= 15:
                if row["Close"] >= pos.entry_price + bt.PARTIAL_TP_R * pos.r_value:
                    sell_shares = int(pos.shares * bt.PARTIAL_TP_FRAC / 1000) * 1000
                    if sell_shares >= 1000:
                        exit_price = row["Close"]
                        proceeds = sell_shares * exit_price * (1 - bt.SELL_COST)
                        cash += proceeds
                        trades.append({
                            "ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                            "entry": pos.entry_price, "exit": exit_price, "shares": sell_shares,
                            "pnl": proceeds - sell_shares * pos.entry_price * (1 + bt.BUY_COST),
                            "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                            "reason": "partial", "bars": pos.bars_held,
                        })
                        pos.shares -= sell_shares
                        pos.partial_taken = True
                        pos.stop_price = max(pos.stop_price, pos.entry_price)

            # 只在 partial 後啟動 MA50 trail（給趨勢更多空間）
            if pos.partial_taken:
                if not pd.isna(row["MA50"]) and row["Close"] < row["MA50"]:
                    exit_price = row["Close"]
                    proceeds = pos.shares * exit_price * (1 - bt.SELL_COST)
                    cash += proceeds
                    trades.append({
                        "ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                        "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                        "pnl": proceeds - pos.shares * pos.entry_price * (1 + bt.BUY_COST),
                        "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                        "reason": "trail_ma50", "bars": pos.bars_held,
                    })
                    to_close.append(tkr)
                    continue

            # 強制出場：超過 100 天
            if pos.bars_held >= 100:
                exit_price = row["Close"]
                proceeds = pos.shares * exit_price * (1 - bt.SELL_COST)
                cash += proceeds
                trades.append({
                    "ticker": tkr, "entry_date": pos.entry_date, "exit_date": date,
                    "entry": pos.entry_price, "exit": exit_price, "shares": pos.shares,
                    "pnl": proceeds - pos.shares * pos.entry_price * (1 + bt.BUY_COST),
                    "r_multiple": (exit_price - pos.entry_price) / pos.r_value,
                    "reason": "time", "bars": pos.bars_held,
                })
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
        equity_curve.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions)})

        if len(positions) >= bt.MAX_CONCURRENT:
            continue
        candidates = []
        for tkr, df in enriched.items():
            if tkr in positions or date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < 2:
                continue
            today = df.iloc[idx]
            prev = df.iloc[idx - 1]
            if not bt.is_setup(prev):
                continue
            if not is_breakout_v2(today, prev):
                continue
            candidates.append((tkr, today, prev))

        candidates.sort(key=lambda x: x[1]["Ret60"], reverse=True)

        for tkr, today, prev in candidates:
            if len(positions) >= bt.MAX_CONCURRENT:
                break
            entry_price = today["Close"]
            atr = today["ATR20"]
            stop = today["Low"]
            risk_per_share = entry_price - stop
            if risk_per_share <= 0 or risk_per_share > bt.MAX_STOP_ATR * atr:
                continue
            risk_dollar = equity * bt.RISK_PER_TRADE
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
    return eq, tr, bench


if __name__ == "__main__":
    eq, tr, bench = run_v2()
    bt.report(eq, tr, bench)
    eq.to_csv("equity_curve_v2.csv")
    tr.to_csv("trades_v2.csv", index=False)
