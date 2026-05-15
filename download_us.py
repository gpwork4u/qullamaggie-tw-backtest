"""下載美股歷史資料"""
import os, time
import pandas as pd
import yfinance as yf
from us_universe import get_us_universe, BENCHMARK_SPY, BENCHMARK_QQQ

DATA_DIR = "data_us"
START = "2017-06-01"
END = "2025-12-31"
os.makedirs(DATA_DIR, exist_ok=True)

def download_one(t):
    try:
        df = yf.download(t, start=START, end=END, auto_adjust=True,
                         progress=False, threads=False)
        if df is None or len(df) < 250:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  {t}: {e}")
        return None

def main():
    tickers = get_us_universe() + [BENCHMARK_SPY, BENCHMARK_QQQ]
    print(f"Downloading {len(tickers)} US tickers...")
    failed = []
    for i, t in enumerate(tickers, 1):
        path = f"{DATA_DIR}/{t}.pkl"
        if os.path.exists(path):
            continue
        df = download_one(t)
        if df is not None:
            df.to_pickle(path)
            print(f"[{i}/{len(tickers)}] {t}: {len(df)} rows")
        else:
            failed.append(t)
            print(f"[{i}/{len(tickers)}] {t}: SKIP")
        time.sleep(0.08)
    print(f"\nFailed: {len(failed)}: {failed[:20]}")

if __name__ == "__main__":
    main()
