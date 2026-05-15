"""Walk-forward 驗證：
切分 2018-2022 (in-sample) vs 2023-2025 (out-of-sample)
比較 V4 / V5 在 IS 與 OOS 的差距，揭露過擬合程度。
"""
import sys
sys.path.insert(0, ".")
from backtest_v5 import run_v5, report, US_CFG, TW_CFG, DEFAULT_PARAMS
import backtest_us_v4 as us_v4
import backtest_tw_v4 as tw_v4


def split_test(market="US"):
    cfg = US_CFG if market == "US" else TW_CFG
    label_v4 = "V4"
    label_v5 = "V5"

    periods = [
        ("In-sample 2018-2022", "2018-01-01", "2022-12-31"),
        ("Out-of-sample 2023-2025", "2023-01-01", "2025-12-30"),
        ("Full 2018-2025", "2018-01-01", "2025-12-30"),
    ]

    rows = []
    for name, start, end in periods:
        print("\n" + "#"*70)
        print(f"### {market} — {name} ({start} ~ {end})")
        print("#"*70)

        # V4 (注意 V4 引擎沒有切時間參數，要修改) — 用 V5 的 leverage=1.3, risk=0.015 跑 V4-like
        # 簡化：直接用 V5 但關掉新功能，等同 V4
        v4_params = dict(DEFAULT_PARAMS)
        v4_params["RS_THRESHOLD"] = 0
        v4_params["WEEKLY_FILTER"] = False
        v4_params["VCP_BONUS_RISK_MULT"] = 1.0
        v4_params["ENABLE_EP"] = False
        eq4, tr4, bench = run_v5(market=market, params=v4_params, start_date=start, end_date=end)
        r4 = report(eq4, tr4, bench, f"{market} V4-equivalent — {name}", cfg["init_capital"])

        # V5 完整版
        eq5, tr5, bench5 = run_v5(market=market, params=DEFAULT_PARAMS, start_date=start, end_date=end)
        r5 = report(eq5, tr5, bench5, f"{market} V5 (full) — {name}", cfg["init_capital"])

        rows.append((name, r4, r5))

    print("\n\n" + "="*70)
    print(f"{market} Walk-forward 彙總")
    print("="*70)
    print(f"{'期間':<28}{'版本':>8} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'vs Bench':>10}")
    for name, r4, r5 in rows:
        for label, r in [("V4-eq", r4), ("V5", r5)]:
            diff = r["cagr"] - r["b_cagr"]
            print(f"{name:<28}{label:>8} {r['cagr']*100:>7.2f}% {r['mdd']*100:>7.2f}% "
                  f"{r['sharpe']:>6.2f} {diff*100:>+9.2f}pp")
    return rows


if __name__ == "__main__":
    market = sys.argv[1] if len(sys.argv) > 1 else "US"
    split_test(market)
