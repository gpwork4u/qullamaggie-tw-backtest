"""下載台股歷史資料並存成 parquet"""
import os
import time
import pandas as pd
import yfinance as yf
from universe import get_universe, BENCHMARK

DATA_DIR = "data"
START = "2017-06-01"  # 多抓半年讓指標可以暖機
END = "2025-12-31"

os.makedirs(DATA_DIR, exist_ok=True)

def download_one(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=START, end=END, auto_adjust=True,
                         progress=False, threads=False)
        if df is None or len(df) < 250:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  {ticker} failed: {e}")
        return None

def main():
    tickers = get_universe() + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers...")
    results = {}
    failed = []
    for i, t in enumerate(tickers, 1):
        path = f"{DATA_DIR}/{t.replace('.', '_')}.pkl"
        if os.path.exists(path):
            results[t] = pd.read_pickle(path)
            continue
        df = download_one(t)
        if df is not None and len(df) > 0:
            df.to_pickle(path)
            results[t] = df
            print(f"[{i}/{len(tickers)}] {t}: {len(df)} rows  range={df.index[0].date()}~{df.index[-1].date()}")
        else:
            failed.append(t)
            print(f"[{i}/{len(tickers)}] {t}: SKIP")
        time.sleep(0.1)
    print(f"\nDone. Success: {len(results)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed tickers (first 20): {failed[:20]}")
    return results

if __name__ == "__main__":
    main()
