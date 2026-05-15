"""美股股票池 - S&P 500 + 高動能名單 + Qulla 偏好的中小型成長股"""

# 大型權值股 (Mega cap)
MEGA_CAP = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "AVGO",
    "JPM", "V", "WMT", "MA", "PG", "JNJ", "ORCL", "HD", "COST", "ABBV",
    "BAC", "KO", "MRK", "PEP", "ADBE", "CRM", "TMO", "NFLX", "AMD", "LIN",
    "CSCO", "ACN", "DIS", "MCD", "ABT", "WFC", "DHR", "QCOM", "TXN", "INTU",
    "VZ", "NKE", "BX", "PM", "INTC", "AMGN", "UNH", "CVX", "XOM", "PFE",
]

# 半導體 / 科技成長股
TECH_GROWTH = [
    "ASML", "TSM", "ARM", "MU", "LRCX", "AMAT", "KLAC", "MRVL", "NXPI", "MCHP",
    "ON", "SWKS", "MPWR", "TER", "ENPH", "SEDG", "SMCI", "ANET", "PANW", "CRWD",
    "ZS", "DDOG", "SNOW", "MDB", "OKTA", "NET", "FTNT", "TEAM", "WDAY", "NOW",
    "SHOP", "SQ", "PYPL", "ABNB", "UBER", "LYFT", "DASH", "RBLX", "U", "PATH",
    "PLTR", "AI", "SOUN", "BBAI", "CRSP", "EDIT", "NTLA", "RXRX", "TWLO", "ZM",
    "DOCU", "PINS", "SNAP", "ROKU", "SPOT", "NFLX", "TTD", "APP", "BMBL", "HOOD",
]

# Qulla 偏好的爆發型股票（電動車、加密、太陽能、生物科技、迷因等）
HIGH_MOMENTUM = [
    "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "QS", "CHPT", "BLNK", "PLUG",
    "FCEL", "BLDP", "FSLR", "RUN", "NOVA", "SHLS", "ARRY", "MAXN",
    "COIN", "MARA", "RIOT", "CLSK", "HUT", "BITF", "MSTR", "HIVE", "WULF", "IREN",
    "MRNA", "BNTX", "NVAX", "PFE", "OCGN", "INO", "VXRT", "CODX",
    "GME", "AMC", "BBBY", "BBIG", "ATER", "PROG", "MULN", "NKLA", "WKHS", "RIDE",
    "SOFI", "AFRM", "UPST", "OPEN", "Z", "RDFN", "COMP", "BLND",
    "DKNG", "PENN", "RKLB", "ASTS", "JOBY", "ACHR", "VRT", "POWL", "GEV",
    "AVAV", "KTOS", "MNTS", "STEM", "FLNC", "GROY", "NEM", "AEM", "AU",
]

# 中型成長股 / 細分領域強勢
MID_CAP_GROWTH = [
    "ROKU", "ETSY", "PINS", "HUBS", "VEEV", "BILL", "PCTY", "PAYC", "DOCN", "DBX",
    "FROG", "CFLT", "S", "TENB", "VRNS", "RPD", "QLYS", "CYBR", "GTLB", "DOMO",
    "PSTG", "NTNX", "BOX", "SMAR", "WIX", "FVRR", "UPWK", "ENVA", "LMND", "ROOT",
    "MELI", "GLBE", "SE", "GRAB", "BABA", "PDD", "JD", "BIDU", "NTES", "BILI",
    "EXEL", "TECH", "VRTX", "REGN", "ALNY", "BMRN", "INCY", "SRPT", "BLUE", "FOLD",
    "CELH", "MNST", "STZ", "TAP", "DECK", "CROX", "ULTA", "LULU", "RH", "WSM",
    "PINS", "CART", "INST", "GTLB", "S", "ESTC", "SPLK", "PD", "FROG",
]

# 高 ADR / 高波動小型股（最像 Qulla 的菜）
SMALL_CAP_VOLATILE = [
    "AMSC", "IRDM", "OPTT", "VLN", "GOEV", "SPRT", "GREE", "ANY", "FAZE",
    "DWAC", "PHUN", "CFVI", "BKKT", "CHGG", "WISH", "CLOV", "SDC", "HOOD",
    "FUBO", "CURI", "ATER", "RDBX", "BBIG", "EXPR", "KOSS", "NEGG", "VINC",
    "INVZ", "AEVA", "MVIS", "LAZR", "OUST", "CGNT", "EVGO", "WBX", "TPIC",
    "ENVX", "SLDP", "MP", "REE", "PSNY", "INDI", "ALLG",
]

def get_us_universe():
    all_tickers = set(MEGA_CAP + TECH_GROWTH + HIGH_MOMENTUM + MID_CAP_GROWTH + SMALL_CAP_VOLATILE)
    return sorted(all_tickers)

BENCHMARK_SPY = "SPY"
BENCHMARK_QQQ = "QQQ"

if __name__ == "__main__":
    u = get_us_universe()
    print(f"Total US universe size: {len(u)}")
    print(u[:30])
