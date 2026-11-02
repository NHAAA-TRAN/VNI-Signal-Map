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

# Cấu hình Resend API (Cài đặt trong Vercel -> Settings -> Environment Variables)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
EMAIL_TO = os.getenv("EMAIL_TO", "email_cua_ban@gmail.com")
EMAIL_FROM = os.getenv("EMAIL_FROM", "VNIndex Signal <onboarding@resend.dev>")
CRON_SECRET = os.getenv("CRON_SECRET", "") # Tuỳ chọn: Bảo mật đường dẫn Cron

# Bộ nhớ đệm (In-Memory Cache) giữ dữ liệu 180 giây để load web tức thì
CACHE = {"timestamp": 0, "data": None}


# =====================================================================
# 2. HÀM XỬ LÝ THỜI GIAN & NỘI SUY THANH KHOẢN (PROJECTED RVOL)
# =====================================================================
def get_vn_time() -> datetime:
    return datetime.now(timezone(timedelta(hours=7)))

def get_intraday_vol_multiplier(vn_now: datetime) -> tuple[float, str]:
    """Nội suy Volume trong phiên để RVOL không bị bóp méo khi chưa hết ngày"""
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
    elif 1430 < hm <= 144
