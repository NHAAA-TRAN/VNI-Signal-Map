import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="VN-INDEX MA10-MA20 Signal Map & Auto Email")

# =====================================================================
# 1. CẤU HÌNH DANH MỤC QUÉT & BIẾN MÔI TRƯỜNG (VERCEL ENV)
# =====================================================================
WATCHLIST = [
    # Nhóm ưu tiên theo dõi sát
    "TCH", "VIX", "FRT", "STB", "GEX", "MBS", "EIB", "ACB", "PNJ", "VCB", "VPB",
    # Chứng khoán & Ngân hàng mở rộng
    "SSI", "VND", "HCM", "VCI", "FTS", "BSI", "CTG", "BID", "MBB", "TCB", "HDB",
    # Bất động sản, Thép, Bán lẻ, Khu công nghiệp, Dầu khí
    "DIG", "PDR", "DXG", "KDH", "NLG", "NVL", "HPG", "HSG", "NKG",
    "MWG", "DGW", "FPT", "CTR", "GMD", "HAH", "PVD", "PVS", "KBC", "IDC", "SZC"
]

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
EMAIL_TO = os.getenv("EMAIL_TO", "email_cua_ban@gmail.com")
EMAIL_FROM = os.getenv("EMAIL_FROM", "VNIndex Signal <onboarding@resend.dev>")
CRON_SECRET = os.getenv("CRON_SECRET", "")

# Bộ nhớ đệm (In-Memory Cache)
CACHE = {"timestamp": 0, "data": None}

# Biến chống gửi email trùng lặp (Anti-Duplicate Cache)
LAST_EMAIL_SENT_TIMESTAMP = 0


# =====================================================================
# 2. HÀM XỬ LÝ THỜI GIAN & NỘI SUY THANH KHOẢN (PROJECTED RVOL)
# =====================================================================
def get_vn_time() -> datetime:
    return datetime.now(timezone(timedelta(hours=7)))

def get_intraday_vol_multiplier(vn_now: datetime) -> tuple[float, str]:
    if vn_now.weekday() >= 5:
        return 1.0, "Dữ liệu Chốt Tuần (EOD)"
    
    hm = vn_now.hour * 100 + vn_now.minute
    if hm < 915:
        return 1.0, "Trước giờ mở cửa (Dữ liệu phiên trước)"
    elif 915 <= hm <= 1130:
        elapsed = max(15, (vn_now.hour - 9) * 60 + vn_now.minute - 15)
        mult = min(3.0, 225.0 / elapsed)
        return round(mult, 2), f"Live Phiên Sáng (Hệ số Vol x{mult:.1f})"
    elif 1130 < hm < 1300:
        return 1.8, "Chốt Phiên Sáng (11:30 Snapshot)"
    elif 1300 <= hm <= 1430:
        elapsed = 135 + (vn_now.hour - 13) * 60 + vn_now.minute
        mult = max(1.0, 225.0 / elapsed)
        status = "🔥 GIỜ VÀNG TRƯỚC ATC" if hm >= 1410 else "Live Phiên Chiều"
        return round(mult, 2), f"{status} (Hệ số Vol x{mult:.2f})"
    elif 1430 < hm <= 1445:
        return 1.0, "Đang khớp lệnh ATC (14:30 - 14:45)"
    else:
        return 1.0, "Chốt Phiên Chính Thức (Sau 15:00)"


# =====================================================================
# 3. ENGINE LƯỢNG HÓA DẢI MA10 - MA20 & ĐA LUỒNG
# =====================================================================
def fetch_stock_ohlcv(symbol: str, days: int = 45) -> pd.DataFrame:
    end_ts = int(time.time())
    start_ts = end_ts - (days * 86400)
    url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from={start_ts}&to={end_ts}&symbol={symbol}&resolution=1D"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=5)
    data = res.json()
    return pd.DataFrame({
        "Open": data["o"],
        "High": data["h"],
        "Low": data["l"],
        "Close": data["c"],
        "Volume": data["v"]
    })

def analyze_ticker(symbol: str, vol_mult: float) -> dict | None:
    try:
        df = fetch_stock_ohlcv(symbol)
        if len(df) < 22:
            return None

        df['Adj_Volume'] = df['Volume'].astype(float)
        df.loc[df.index[-1], 'Adj_Volume'] = df.loc[df.index[-1], 'Volume'] * vol_mult

        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Vol_MA20'] = df['Volume'].shift(1).rolling(20).mean()
        df['RVOL'] = df['Adj_Volume'] / df['Vol_MA20']

        candle_range = (df['High'] - df['Low']).replace(0, np.nan)
        close_pos = (df['Close'] - df['Low']) / candle_range
        df['Flow_Pressure'] = ((2 * df['Close'] - df['High'] - df['Low']) / candle_range) * df['RVOL']

        df['Dist_MA10_Pct'] = (df['Close'] - df['MA10']) / df['MA10'] * 100
        df['Dist_MA20_Pct'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['Room_To_MA20_Pct'] = (df['MA20'] - df['Close']) / df['Close'] * 100
        ma20_flat_or_up = df['MA20'] >= df['MA20'].shift(2) * 0.997

        # Quy tắc 4 tín hiệu
        reject_at_ma20 = (
            (df['MA10'] < df['MA20']) &
            (df['High'] >= df['MA20'] * 0.985) &
            (df['Close'] < df['MA20']) &
            ((df['Close'] < df['Open']) | (close_pos < 0.4)) &
            (df['RVOL'] >= 1.0)
        )
        break_back_ma10 = (
            (df['Close'].shift(1) > df['MA10'].shift(1)) &
            (df['Close'] < df['MA10']) &
            (df['Close'] < df['Low'].shift(1)) &
            (df['Flow_Pressure'] < -0.3)
        )
        cond_failed_rebound = reject_at_ma20 | break_back_ma10

        prev_day_low_vol_test = (
            (df['Low'].shift(1) <= df[['MA10', 'MA20']].max(axis=1).shift(1) * 1.01) &
            (df['Close'].shift(1) >= df['MA20'].shift(1) * 0.995) &
            (df['RVOL'].shift(1) < 0.8)
        )
        intraday_pinbar_test =
