"""下載擴充版台股資料"""
import os, time
import pandas as pd
import yfinance as yf
from tw_universe_v2 import get_tw_universe_v2

DATA_DIR = "data"
START, END = "2017-06-01", "2025-12-31"

def download_one(t):
    try:
        df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False, threads=False)
        if df is None or len(df) < 250:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].copy().dropna()
        return df
    except Exception:
        return None

def main():
    tickers = get_tw_universe_v2()
    new_tickers = [t for t in tickers if not os.path.exists(f"{DATA_DIR}/{t.replace('.','_')}.pkl")]
    print(f"Total {len(tickers)}, need download {len(new_tickers)}")
    failed = []
    for i, t in enumerate(new_tickers, 1):
        df = download_one(t)
        if df is not None:
            df.to_pickle(f"{DATA_DIR}/{t.replace('.','_')}.pkl")
            print(f"[{i}/{len(new_tickers)}] {t}: {len(df)}")
        else:
            failed.append(t)
        time.sleep(0.08)
    print(f"Failed: {len(failed)}")

if __name__ == "__main__":
    main()
