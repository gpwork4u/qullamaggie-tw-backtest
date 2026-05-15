"""
每日 Scanner — 從回測到實戰的橋樑

使用方式：
    python3 daily_scanner.py US         # 掃描美股
    python3 daily_scanner.py TW         # 掃描台股
    python3 daily_scanner.py US 100000  # 指定資金規模 (USD)

輸出：
    1. scanner_output_US_YYYYMMDD.md  — 給人看的 Markdown 報告
    2. scanner_output_US_YYYYMMDD.csv — 給程式接的 CSV
"""
import os, sys, glob, time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from qulla_indicators import (
    add_all_indicators, compute_rs_rank, detect_vcp,
    is_ep_setup, is_parabolic_short, weekly_uptrend
)
from backtest_v5 import (
    US_CFG, TW_CFG, DEFAULT_PARAMS, is_setup, is_breakout, pick_trail_ma
)


def refresh_data(data_dir, market, days_back=30):
    """重新下載最近 days_back 天的資料，覆蓋舊資料的尾端"""
    files = glob.glob(f"{data_dir}/*.pkl")
    print(f"[{market}] 更新 {len(files)} 檔資料中（近 {days_back} 天）...")
    today = pd.Timestamp.today().normalize()
    refresh_start = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")

    updated = 0
    for i, f in enumerate(files):
        t = os.path.basename(f).replace(".pkl", "")
        if market == "TW":
            yf_ticker = t.replace("_", ".")
        else:
            yf_ticker = t
        try:
            new_df = yf.download(yf_ticker, start=refresh_start, auto_adjust=True,
                                 progress=False, threads=False)
            if new_df is None or len(new_df) == 0:
                continue
            if isinstance(new_df.columns, pd.MultiIndex):
                new_df.columns = new_df.columns.get_level_values(0)
            new_df = new_df[["Open","High","Low","Close","Volume"]].dropna()

            old_df = pd.read_pickle(f)
            merged = pd.concat([old_df, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.to_pickle(f)
            updated += 1
        except Exception as e:
            pass
        time.sleep(0.05)
    print(f"[{market}] 更新完成: {updated} 檔")


def scan(market="US", account_size=None, as_of=None):
    cfg = US_CFG if market == "US" else TW_CFG
    if account_size is None:
        account_size = cfg["init_capital"]

    print(f"\n{'='*70}")
    print(f"  Qulla V5 Daily Scanner — {market}")
    print(f"  帳戶規模: {account_size:,.0f} {'USD' if market=='US' else 'TWD'}")
    print(f"  掃描時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    # 載入所有資料
    enriched = {}
    for f in glob.glob(f"{cfg['data_dir']}/*.pkl"):
        t = os.path.basename(f).replace(".pkl", "")
        if market == "TW":
            t = t.replace("_", ".")
        df = pd.read_pickle(f)
        if len(df) >= 252:
            enriched[t] = add_all_indicators(df)

    # 找基準
    bench = None
    for k in cfg["benchmark_keys"]:
        if k in enriched:
            bench = enriched.pop(k)
            break
    if bench is None:
        raise RuntimeError("找不到基準")

    # 取掃描日期
    if as_of is not None:
        target = pd.Timestamp(as_of)
        candidates = [d for df in enriched.values() for d in df.index if d <= target]
        latest_date = max(candidates) if candidates else target
    else:
        latest_date = max(df.index[-1] for df in enriched.values())
    print(f"掃描日期: {latest_date.date()}\n")

    # 大盤狀態
    b_latest = bench.loc[bench.index <= latest_date].iloc[-1]
    bull = (not pd.isna(b_latest["MA200"])) and b_latest["Close"] > b_latest["MA200"]
    strong = (not pd.isna(b_latest["MA50"])) and b_latest["Close"] > b_latest["MA50"]
    market_state = "🟢 多頭" if bull and strong else ("🟡 多頭弱" if bull else "🔴 熊市")
    bench_name = cfg["benchmark_keys"][0]
    print(f"大盤 ({bench_name}): {market_state}")
    print(f"  收盤 {b_latest['Close']:.2f}  vs MA50 {b_latest['MA50']:.2f}  vs MA200 {b_latest['MA200']:.2f}")
    print()

    if not bull:
        print("⚠️  大盤跌破 200MA，策略建議停止新進場（保留現有持倉）")
        return

    # RS 排名
    closes_dict = {t: df["Close"] for t, df in enriched.items()}
    rs_ranks = compute_rs_rank(closes_dict, latest_date, lookback=126)

    params = dict(DEFAULT_PARAMS)
    gap_max = cfg["gap_filter"]

    breakout_signals = []
    ep_signals = []

    for tkr, df in enriched.items():
        if latest_date not in df.index:
            continue
        idx = df.index.get_loc(latest_date)
        if idx < 252:
            continue
        today = df.iloc[idx]
        prev = df.iloc[idx - 1]

        rs = rs_ranks.get(tkr, 0)
        if rs < params["RS_THRESHOLD"]:
            continue

        # 週線過濾
        if params["WEEKLY_FILTER"] and not weekly_uptrend(today):
            continue

        # 突破
        if is_setup(prev, params) and is_breakout(today, prev, params, gap_max):
            vcp_info = detect_vcp(df, idx, window=40, min_contractions=2)
            breakout_signals.append({
                "ticker": tkr,
                "close": today["Close"],
                "stop": today["Low"],
                "atr20": today["ATR20"],
                "adr20": today["ADR20"],
                "ret60": today["Ret60"],
                "ret120": today["Ret120"],
                "volume": today["Volume"],
                "vol_ratio": today["Volume"] / today["VolMA20"],
                "rs_rank": rs,
                "vcp": vcp_info["is_vcp"],
                "vcp_contractions": vcp_info.get("contractions", 0),
                "high60": today["High60"],
                "pct_from_high60": (today["Close"] - today["High60"]) / today["High60"],
            })

        # EP
        if params["ENABLE_EP"] and is_ep_setup(today, prev, df, idx, market=cfg["market"]):
            ep_signals.append({
                "ticker": tkr,
                "close": today["Close"],
                "stop": today["Low"],
                "atr20": today["ATR20"],
                "adr20": today["ADR20"],
                "gap": today["GapUp"],
                "vol_ratio": today["Volume"] / today["VolMA20"],
                "rs_rank": rs,
            })

    # 排序
    breakout_signals.sort(key=lambda x: (x["vcp"], x["rs_rank"]), reverse=True)
    ep_signals.sort(key=lambda x: x["gap"], reverse=True)

    # 計算倉位
    risk_per_trade = params["RISK_PER_TRADE"]
    risk_dollar = account_size * risk_per_trade * (1.0 if strong else 0.5)
    max_pos_dollar = account_size * params["MAX_POS_PCT"]
    LOT = cfg["lot"]
    BUY_COST = cfg["buy_cost"]

    def calc_size(entry, stop, vcp=False):
        rps = entry - stop
        if rps <= 0:
            return 0, 0
        mult = params["VCP_BONUS_RISK_MULT"] if vcp else 1.0
        shares_risk = int(risk_dollar * mult / rps / LOT) * LOT if LOT > 1 else int(risk_dollar * mult / rps)
        shares_pos = int(max_pos_dollar / entry / LOT) * LOT if LOT > 1 else int(max_pos_dollar / entry)
        shares = min(shares_risk, shares_pos)
        cost = shares * entry * (1 + BUY_COST)
        return shares, cost

    # 輸出
    out_md = []
    out_md.append(f"# Qulla V5 Daily Scanner — {market}")
    out_md.append(f"\n**掃描日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out_md.append(f"**最新交易日**: {latest_date.date()}")
    out_md.append(f"**帳戶規模**: {account_size:,.0f} {'USD' if market=='US' else 'TWD'}")
    out_md.append(f"**大盤狀態**: {market_state} (vs MA200 {b_latest['Close']/b_latest['MA200']-1:+.2%})")
    out_md.append(f"**單筆風險**: {risk_per_trade*100:.2f}% = {risk_dollar:,.0f} {'USD' if market=='US' else 'TWD'}")
    out_md.append("")

    print(f"\n📊 訊號統計：突破 {len(breakout_signals)} 檔 | EP {len(ep_signals)} 檔")

    out_md.append(f"## 🚀 突破訊號 ({len(breakout_signals)} 檔)\n")
    if breakout_signals:
        out_md.append("| # | Ticker | Close | Stop | R-dist | ATR% | ADR% | RS | VCP | Ret60 | Vol× | Shares | Cost | Risk |")
        out_md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for i, s in enumerate(breakout_signals, 1):
            shares, cost = calc_size(s["close"], s["stop"], vcp=s["vcp"])
            risk = shares * (s["close"] - s["stop"])
            vcp_tag = f"✓({s['vcp_contractions']})" if s["vcp"] else "—"
            atr_pct = s["atr20"] / s["close"]
            adr_pct = s["adr20"]
            out_md.append(f"| {i} | **{s['ticker']}** | {s['close']:.2f} | {s['stop']:.2f} | "
                         f"{s['close']-s['stop']:.2f} ({(s['close']-s['stop'])/s['close']*100:.1f}%) | "
                         f"{atr_pct*100:.1f}% | {adr_pct*100:.1f}% | {s['rs_rank']:.0f} | {vcp_tag} | "
                         f"{s['ret60']*100:+.0f}% | {s['vol_ratio']:.1f}× | "
                         f"{shares:,} | {cost:,.0f} | {risk:,.0f} |")
            rows.append({**s, "shares": shares, "cost": cost, "risk_dollar": risk, "type": "breakout"})
    else:
        out_md.append("_目前沒有符合突破條件的標的_")
        rows = []

    out_md.append(f"\n## ⚡ EP 訊號 ({len(ep_signals)} 檔)\n")
    if ep_signals:
        out_md.append("| # | Ticker | Close | Stop | Gap | Vol× | ADR% | RS | Shares | Cost | Risk |")
        out_md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, s in enumerate(ep_signals, 1):
            shares, cost = calc_size(s["close"], s["stop"])
            risk = shares * (s["close"] - s["stop"])
            out_md.append(f"| {i} | **{s['ticker']}** | {s['close']:.2f} | {s['stop']:.2f} | "
                         f"{s['gap']*100:+.1f}% | {s['vol_ratio']:.1f}× | {s['adr20']*100:.1f}% | "
                         f"{s['rs_rank']:.0f} | {shares:,} | {cost:,.0f} | {risk:,.0f} |")
            rows.append({**s, "shares": shares, "cost": cost, "risk_dollar": risk, "type": "ep"})
    else:
        out_md.append("_目前沒有 EP 訊號_")

    out_md.append("\n## 📋 進場執行清單\n")
    out_md.append("**今日盤後決策、明日開盤後執行：**\n")
    out_md.append("1. 確認候選股票今晨沒有重大新聞（盈警、訴訟、下市風險）")
    out_md.append("2. 開盤後等第一根 5 分鐘 K 線收，價格 > 昨日突破收盤即可進場")
    out_md.append("3. 停損掛單：進場後立刻掛單在 stop 價位")
    out_md.append("4. 若同時有多檔訊號超過倉位上限（最多 6 檔），優先順序：")
    out_md.append("   - EP（最稀有）→ VCP 確認的突破 → RS 最高的突破")
    out_md.append("5. 進場後 3-5 日內若有 1R 浮盈，賣 1/3 鎖利、剩餘停損移到 BE")
    out_md.append("")
    out_md.append("**風控提醒：**")
    out_md.append(f"- 今日為 **{market_state}**，{'執行標準倉位' if strong else '弱市半倉執行'}")
    out_md.append(f"- 同時持倉上限：6 檔")
    out_md.append(f"- 單檔風險上限：1.5% × {'×1.3 (VCP)' if any(s['vcp'] for s in breakout_signals) else ''}")

    # 寫檔
    date_str = latest_date.strftime("%Y%m%d")
    md_path = f"scanner_output_{market}_{date_str}.md"
    csv_path = f"scanner_output_{market}_{date_str}.csv"

    with open(md_path, "w") as f:
        f.write("\n".join(out_md))

    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(csv_path, index=False)

    print(f"\n✓ 報告已輸出: {md_path}")
    if rows:
        print(f"✓ 資料已輸出: {csv_path}")
    print("\n" + "="*70)
    print("摘要：")
    print("="*70)
    print("\n".join(out_md[-30:]))
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    market = args[0] if args else "US"
    account_size = float(args[1]) if len(args) > 1 else None

    as_of = None
    for a in sys.argv[1:]:
        if a.startswith("--date="):
            as_of = a.split("=")[1]

    cfg = US_CFG if market == "US" else TW_CFG
    refresh = "--no-refresh" not in sys.argv and as_of is None
    if refresh:
        refresh_data(cfg["data_dir"], market, days_back=30)

    scan(market=market, account_size=account_size, as_of=as_of)
