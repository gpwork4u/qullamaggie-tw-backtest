"""Qulla 完整指標庫：VCP、RS、EP、Parabolic、多時間框架"""
import numpy as np
import pandas as pd


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """加上所有指標"""
    df = df.copy()
    for w in [10, 20, 50, 100, 200]:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["VolMA50"] = df["Volume"].rolling(50).mean()
    df["DollarVol20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["High60"] = df["High"].rolling(60).max()
    df["High252"] = df["High"].rolling(252).max()
    df["BreakoutLevel"] = df["High"].rolling(15).max().shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR20"] = tr.rolling(20).mean()
    df["ADR20"] = ((df["High"] - df["Low"]) / df["Close"]).rolling(20).mean()
    df["Ret20"] = df["Close"] / df["Close"].shift(20) - 1
    df["Ret60"] = df["Close"] / df["Close"].shift(60) - 1
    df["Ret120"] = df["Close"] / df["Close"].shift(120) - 1
    df["Ret252"] = df["Close"] / df["Close"].shift(252) - 1
    df["TodayRet"] = df["Close"] / df["Close"].shift(1) - 1
    df["GapUp"] = df["Open"] / df["Close"].shift(1) - 1  # 開盤 vs 昨日收盤

    # 週線（resample 到週、再 reindex 回日線）
    weekly = df["Close"].resample("W-FRI").last()
    w_ma10 = weekly.rolling(10).mean()
    w_ma30 = weekly.rolling(30).mean()
    df["W_MA10"] = w_ma10.reindex(df.index, method="ffill")
    df["W_MA30"] = w_ma30.reindex(df.index, method="ffill")
    df["W_Close"] = weekly.reindex(df.index, method="ffill")
    return df


def compute_rs_rank(closes_dict: dict, date, lookback=126) -> dict:
    """IBD 式 RS 排名：每檔股票過去 lookback 日報酬，跨所有股票做百分位排名。
    回傳 {ticker: rs_rank 0-99}"""
    rets = {}
    for tkr, s in closes_dict.items():
        if date not in s.index:
            continue
        idx = s.index.get_loc(date)
        if idx < lookback:
            continue
        ret = s.iloc[idx] / s.iloc[idx - lookback] - 1
        if not np.isnan(ret) and not np.isinf(ret):
            rets[tkr] = ret
    if not rets:
        return {}
    arr = pd.Series(rets)
    ranks = arr.rank(pct=True) * 99
    return ranks.to_dict()


def detect_vcp(df: pd.DataFrame, idx: int, window=40, min_contractions=2) -> dict:
    """VCP 偵測：找到價格收斂 + 量能萎縮的型態。
    回傳 {'is_vcp': bool, 'contractions': int, 'last_dd': float, 'vol_dryup': bool}"""
    if idx < window:
        return {"is_vcp": False}
    seg = df.iloc[idx - window:idx + 1]
    closes = seg["Close"].values
    if len(closes) < 10:
        return {"is_vcp": False}

    # 簡化版 VCP：找到 N 個 swing high 與 swing low，計算回檔幅度
    # 用 rolling window peak detection (3 日 high)
    peaks_idx = []
    troughs_idx = []
    for i in range(2, len(closes) - 2):
        if closes[i] > closes[i-1] and closes[i] > closes[i-2] and closes[i] > closes[i+1] and closes[i] > closes[i+2]:
            peaks_idx.append(i)
        elif closes[i] < closes[i-1] and closes[i] < closes[i-2] and closes[i] < closes[i+1] and closes[i] < closes[i+2]:
            troughs_idx.append(i)

    # 計算每次回檔的幅度（peak → 下一個 trough）
    drawdowns = []
    for p in peaks_idx:
        future_troughs = [t for t in troughs_idx if t > p]
        if future_troughs:
            t = future_troughs[0]
            dd = (closes[t] - closes[p]) / closes[p]
            drawdowns.append(abs(dd))

    if len(drawdowns) < min_contractions:
        return {"is_vcp": False, "contractions": len(drawdowns)}

    # 「收斂」：後續回檔幅度遞減
    contracting = all(drawdowns[i] >= drawdowns[i+1] * 0.7 for i in range(len(drawdowns) - 1))

    # 量能萎縮：後半段量能 < 前半段
    half = len(seg) // 2
    vol_dryup = seg["Volume"].iloc[half:].mean() < seg["Volume"].iloc[:half].mean() * 0.9

    is_vcp = contracting and vol_dryup and drawdowns[-1] < 0.15

    return {
        "is_vcp": is_vcp,
        "contractions": len(drawdowns),
        "last_dd": drawdowns[-1] if drawdowns else 0,
        "vol_dryup": vol_dryup,
    }


def is_ep_setup(today, prev, df, idx, market="US") -> bool:
    """EP（情境轉折）：跳空 + 巨量 + 過去未大漲 + 站上 50MA
    沒有財報資料，用價格行為近似
    """
    gap_thr = 0.08 if market == "US" else 0.05  # 美股 8%、台股 5%（受漲跌停限制）
    if pd.isna(today["GapUp"]) or today["GapUp"] < gap_thr:
        return False
    # 巨量：當日量 > 過去 20 日均量 * 2
    if pd.isna(today["VolMA20"]) or today["Volume"] < today["VolMA20"] * 2.0:
        return False
    # 站上 50MA
    if pd.isna(today["MA50"]) or today["Close"] < today["MA50"]:
        return False
    # 過去 60 日報酬 < 30%（避免在已暴漲後跳空高開）
    if pd.isna(today["Ret60"]) or today["Ret60"] > 0.30:
        return False
    # 過去 60 日內未曾有過 5% 以上 gap up（避免重複事件）
    recent = df.iloc[max(0, idx-60):idx]
    if (recent["GapUp"] > 0.05).any():
        return False
    # 流動性
    if today["DollarVol20"] < 5_000_000:
        return False
    return True


def is_parabolic_short(today, prev, df, idx) -> bool:
    """拋物線做空：短期暴漲 + 跌破短均線 + 巨量
    """
    if idx < 20:
        return False
    # 過去 15 日漲幅 > 50%
    r15 = today["Close"] / df["Close"].iloc[idx - 15] - 1
    if pd.isna(r15) or r15 < 0.50:
        return False
    # 距離 10MA 偏離過大
    if pd.isna(today["MA10"]) or today["Close"] < today["MA10"] * 1.10:
        return False
    # 今日跌破前一日低點
    if today["Close"] >= prev["Low"]:
        return False
    # 巨量
    if today["Volume"] < today["VolMA20"] * 1.5:
        return False
    return True


def weekly_uptrend(today) -> bool:
    """週線是否在上升趨勢"""
    if pd.isna(today["W_MA10"]) or pd.isna(today["W_MA30"]):
        return True  # 資料不足時不擋
    return today["W_Close"] > today["W_MA10"] >= today["W_MA30"]
