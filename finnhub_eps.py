"""
Finnhub EPS Surprise 整合

註冊免費 API key: https://finnhub.io/register
免費方案：60 calls/min，足以日掃描使用

使用方式：
    export FINNHUB_API_KEY=your_key_here
    python3 finnhub_eps.py AAPL

回傳該股最近一次財報的 EPS surprise 百分比、營收 surprise 等
可用於強化 EP 策略的精度
"""
import os, sys
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = os.environ.get("FINNHUB_API_KEY", "")
BASE = "https://finnhub.io/api/v1"


def get_earnings_surprise(ticker: str, limit=4) -> list:
    """取得最近 N 季 EPS 與營收的 surprise %"""
    if not API_KEY:
        return []
    try:
        r = requests.get(f"{BASE}/stock/earnings",
                        params={"symbol": ticker, "limit": limit, "token": API_KEY},
                        timeout=5)
        if r.status_code != 200:
            return []
        return r.json()
    except Exception:
        return []


def get_recent_eps_surprise_pct(ticker: str) -> dict | None:
    """取得最近一季 EPS surprise %（actual vs estimate）"""
    data = get_earnings_surprise(ticker, limit=1)
    if not data:
        return None
    latest = data[0]
    est = latest.get("estimate")
    act = latest.get("actual")
    if est is None or act is None or est == 0:
        return None
    surprise_pct = (act - est) / abs(est) * 100
    return {
        "period": latest.get("period"),
        "estimate": est,
        "actual": act,
        "surprise_pct": surprise_pct,
        "quarter": latest.get("quarter"),
        "year": latest.get("year"),
    }


def is_ep_eps_qualified(ticker: str, min_surprise_pct=15.0, days_since_max=10) -> bool:
    """檢查股票最近一次 EPS 是否：
    1. 大幅優於預期（surprise ≥ min_surprise_pct）
    2. 距今 ≤ days_since_max 天（避免接到舊財報）
    """
    info = get_recent_eps_surprise_pct(ticker)
    if not info:
        return False
    if info["surprise_pct"] < min_surprise_pct:
        return False
    try:
        period_dt = pd.Timestamp(info["period"])
        days_ago = (pd.Timestamp.today() - period_dt).days
        if days_ago > days_since_max + 7:  # 加 7 天緩衝（財報報告日 vs period 結束日差）
            return False
    except Exception:
        pass
    return True


def enrich_scanner_with_eps(scanner_signals: list) -> list:
    """給 daily_scanner 結果加上 EPS 資訊"""
    if not API_KEY:
        print("⚠️  未設定 FINNHUB_API_KEY，跳過 EPS 強化")
        return scanner_signals
    for sig in scanner_signals:
        info = get_recent_eps_surprise_pct(sig["ticker"])
        if info:
            sig["eps_surprise_pct"] = info["surprise_pct"]
            sig["eps_period"] = info["period"]
    return scanner_signals


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: 未設定 FINNHUB_API_KEY 環境變數")
        print("註冊取得免費 key: https://finnhub.io/register")
        sys.exit(1)
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(f"查詢 {ticker} 的 EPS surprise...")
    info = get_recent_eps_surprise_pct(ticker)
    if info:
        print(f"\n最近一季財報 ({info['year']} Q{info['quarter']}):")
        print(f"  Period: {info['period']}")
        print(f"  Estimate: {info['estimate']}")
        print(f"  Actual: {info['actual']}")
        print(f"  Surprise: {info['surprise_pct']:+.2f}%")
        print(f"\nEP 資格 (surprise ≥ 15%): {'✓' if info['surprise_pct'] >= 15 else '✗'}")
    else:
        print("無資料")
