"""擴充版美股股票池 - 加入更多 Russell 1000 + 細分產業"""
from us_universe import MEGA_CAP, TECH_GROWTH, HIGH_MOMENTUM, MID_CAP_GROWTH, SMALL_CAP_VOLATILE

# 健康照護 / 生技
HEALTHCARE = [
    "LLY", "TMO", "UNH", "ELV", "CI", "HUM", "MDT", "GILD", "ZTS", "ISRG",
    "BSX", "SYK", "BDX", "EW", "HOLX", "IDXX", "RMD", "PODD", "DXCM", "ALGN",
    "INSP", "AXNX", "PEN", "TNDM", "OMI", "MDGL", "VKTX", "VTRS", "IQV", "MTD",
    "WST", "WAT", "A", "BIO", "CRL", "ILMN", "RVMD", "ASND", "ARGX", "BMY",
    "GMAB", "LEGN", "NTRA", "VEEV",
]

# 工業 / 國防 / 航太
INDUSTRIAL = [
    "CAT", "DE", "BA", "LMT", "RTX", "GD", "NOC", "TDG", "HEI", "AXON",
    "PH", "ITW", "EMR", "ETN", "ROK", "AME", "DOV", "FTV", "XYL", "PNR",
    "URI", "BLDR", "VMC", "MLM", "EXP", "FAST", "PWR", "PRIM", "MTZ", "FIX",
    "ATKR", "ATR", "GTLS", "SPXC", "NPO", "WCC", "TT", "CARR", "OTIS",
]

# 金融 / 保險 / 房地產
FINANCIAL = [
    "GS", "MS", "SCHW", "BLK", "AXP", "SPGI", "MCO", "ICE", "CME", "CBOE",
    "MMC", "AON", "WTW", "AJG", "BRO", "PGR", "TRV", "ALL", "MET", "PRU",
    "AFL", "HIG", "CNS", "BX", "KKR", "APO", "ARES", "OWL", "TPG", "EQT",
    "AMP", "RJF", "LPLA", "TROW", "BEN", "IVZ", "NTRS", "STT", "BK", "USB",
]

# 消費 / 零售
CONSUMER = [
    "TJX", "ROST", "BURL", "LULU", "DECK", "ONON", "BIRK", "CROX", "RH", "WSM",
    "TGT", "DG", "DLTR", "FIVE", "OLLI", "BJ", "TPR", "RL", "PVH", "TPB",
    "KMX", "AZO", "ORLY", "AAP", "MNRO", "DPZ", "QSR", "CMG", "WING", "CAVA",
    "DRI", "SHAK", "PLAY", "TXRH", "BLMN", "CAKE", "EAT", "BROS", "CELH",
]

# 能源 / 礦業
ENERGY = [
    "EOG", "PXD", "DVN", "OXY", "FANG", "MRO", "APA", "HES", "OVV", "MTDR",
    "SLB", "HAL", "BKR", "FTI", "CHX", "NOV", "MPC", "PSX", "VLO", "HF",
    "TPL", "TRGP", "WMB", "OKE", "KMI", "EQT", "AR", "RRC", "GPOR",
    "FCX", "SCCO", "TECK", "NEM", "GOLD", "AEM", "WPM", "FNV", "RGLD", "PAAS",
    "LITM", "MP", "USAR", "UEC", "CCJ",
]

# AI / 機器人 / 雲端原生
AI_ROBOTICS = [
    "NVDA", "AMD", "AVGO", "MRVL", "ARM", "QCOM", "TSM",
    "PLTR", "AI", "SOUN", "BBAI", "GTLB", "FROG", "S", "TENB", "CFLT",
    "MDB", "ESTC", "SNOW", "DDOG", "NET", "PATH", "U", "RBLX",
    "RKLB", "ASTS", "JOBY", "ACHR", "EH",
    "IONQ", "RGTI", "QBTS", "QUBT",
]

# 新興 / IPO / 主題股
EMERGING = [
    "RDDT", "DJT", "WEAV", "AS", "OSCR", "AGS", "OBDC", "MAIN", "ARCC",
    "NVMI", "ASTS", "ACHR", "JOBY", "BLDE", "EH", "ALAB", "TEM",
    "CRDO", "RXRX", "TEM", "BBAI", "CIFR", "BTBT", "EBON", "CAN", "DGHI",
    "SOXL", "TQQQ", "FNGU",
]

def get_us_universe_v2():
    all_lists = (MEGA_CAP + TECH_GROWTH + HIGH_MOMENTUM + MID_CAP_GROWTH + SMALL_CAP_VOLATILE
                 + HEALTHCARE + INDUSTRIAL + FINANCIAL + CONSUMER + ENERGY + AI_ROBOTICS + EMERGING)
    return sorted(set(all_lists))

if __name__ == "__main__":
    u = get_us_universe_v2()
    print(f"V2 Universe size: {len(u)}")
