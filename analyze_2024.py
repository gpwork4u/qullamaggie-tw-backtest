"""分析 2024 漏掉的飆股 vs V2 實際進場名單"""
import os, glob
import pandas as pd
from collections import defaultdict

# 1. 找出 2024 年漲幅 > 50% 的股票
data = {}
for f in glob.glob("data/*.pkl"):
    t = os.path.basename(f).replace(".pkl", "").replace("_", ".")
    if t == "0050.TW":
        continue
    df = pd.read_pickle(f)
    if "2024-01-02" in df.index.astype(str).tolist() or len(df.loc["2024":"2024"]) > 200:
        d24 = df.loc["2024-01-01":"2024-12-31"]
        if len(d24) < 200:
            continue
        ret = d24["Close"].iloc[-1] / d24["Close"].iloc[0] - 1
        data[t] = ret

winners_2024 = sorted(data.items(), key=lambda x: -x[1])[:30]

# 2. V2 在 2024 實際進場的標的
tr = pd.read_csv("trades_v2.csv", parse_dates=["entry_date", "exit_date"])
tr_2024 = tr[(tr["entry_date"] >= "2024-01-01") & (tr["entry_date"] < "2025-01-01")]
traded_2024 = set(tr_2024["ticker"].unique())

print("="*70)
print("2024 年漲幅 Top 30 vs 策略實際進場")
print("="*70)
print(f"{'排名':>4} {'代號':>10} {'2024漲幅':>10}  {'策略有進場?':>12}")
for i, (t, r) in enumerate(winners_2024, 1):
    flag = "✓ 有進場" if t in traded_2024 else "✗ 漏掉"
    print(f"{i:>4} {t:>10} {r*100:>9.1f}%  {flag:>12}")

print(f"\n2024 策略總進場檔數: {len(traded_2024)}")
print(f"2024 策略進場標的: {sorted(traded_2024)}")

# 3. 分析「為什麼漏掉」- 看每個 top 標的最強 30 日的行為
print("\n" + "="*70)
print("漏掉的 Top 10 - 嘗試找出漏掉原因")
print("="*70)
import sys
sys.path.insert(0, ".")
from backtest import add_indicators, is_setup, is_breakout

for t, r in winners_2024[:15]:
    if t in traded_2024:
        continue
    df = pd.read_pickle(f"data/{t.replace('.', '_')}.pkl")
    df = add_indicators(df)
    d24 = df.loc["2024-01-01":"2024-12-31"]
    # 統計 2024 期間 setup / breakout 出現天數
    setup_days = 0
    breakout_days = 0
    for i in range(1, len(d24)):
        prev = d24.iloc[i-1]
        today = d24.iloc[i]
        if is_setup(prev):
            setup_days += 1
            if is_breakout(today, prev):
                breakout_days += 1
    print(f"{t}: 2024漲幅 {r*100:>6.1f}%  | setup 出現 {setup_days:>3}/{len(d24)} 日 | breakout {breakout_days} 次")
