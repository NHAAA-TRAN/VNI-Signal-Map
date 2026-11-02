from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import requests
import time

app = FastAPI(title="VN-INDEX MA10-MA20 Signal Map")

# Danh sách rổ cổ phiếu thanh khoản cao đại diện các nhóm ngành (bạn có thể mở rộng thêm)
WATCHLIST = [
    "TCH", "VIX", "FRT", "STB", "GEX", "MBS", "EIB", "ACB", "PNJ", "VCB", "VPB",
    "SSI", "VND", "HCM", "VCI", "DIG", "PDR", "DXG", "KDH", "NLG",
    "HPG", "HSG", "NKG", "MWG", "DGW", "FPT", "CTR", "PVD", "PVS", "GMD", "HAH"
]

# Bộ nhớ đệm tạm thời để tránh bị chặn IP khi F5 liên tục
CACHE = {"timestamp": 0, "data": None, "updated_at_str": ""}

def get_vn_time():
    return datetime.now(timezone(timedelta(hours=7)))

def get_intraday_vol_multiplier(vn_now: datetime) -> tuple[float, str]:
    """
    Tính hệ số phóng chiếu Volume tùy theo khung giờ trong phiên
    để RVOL không bị bóp méo khi chưa hết ngày.
    """
    if vn_now.weekday() >= 5: # Thứ 7, Chủ Nhật
        return 1.0, "Dữ liệu Chốt Tuần (EOD)"
    
    hm = vn_now.hour * 100 + vn_now.minute
    if hm < 915:
        return 1.0, "Trước giờ mở cửa (Dữ liệu phiên trước)"
    elif 915 <= hm <= 1130:
        # Tính số phút đã trôi qua trong phiên sáng (tối đa 135 phút)
        elapsed = max(15, (vn_now.hour - 9) * 60 + vn_now.minute - 15)
        total_trading_mins = 225.0 # Tổng 225 phút giao dịch cả ngày
        mult = min(3.0, total_trading_mins / elapsed)
        return round(mult, 2), f"Live Phiên Sáng (Hệ số Vol x{mult:.1f})"
    elif 1130 < hm < 1300:
        return 1.8, "Chốt Phiên Sáng (11:30 Snapshot)"
    elif 1300 <= hm <= 1430:
        elapsed = 135 + (vn_now.hour - 13) * 60 + vn_now.minute
        mult = max(1.0, 225.0 / elapsed)
        status = "GIỜ VÀNG TRƯỚC ATC" if hm >= 1410 else "Live Phiên Chiều"
        return round(mult, 2), f"{status} (Hệ số Vol x{mult:.2f})"
    elif 1430 < hm <= 1445:
        return 1.0, "Đang khớp lệnh ATC (14:30 - 14:45)"
    else:
        return 1.0, "Dữ liệu Đóng Cửa Chính Thức (Sau 15:00)"

def fetch_stock_ohlcv(symbol: str, days: int = 45) -> pd.DataFrame:
    """Lấy dữ liệu OHLCV trực tiếp từ API public của Entrade/DNSE (Rất nhanh & nhẹ trên Vercel)"""
    end_ts = int(time.time())
    start_ts = end_ts - (days * 86400)
    url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from={start_ts}&to={end_ts}&symbol={symbol}&resolution=1D"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=5)
    data = res.json()
    df = pd.DataFrame({
        "Open": data["o"],
        "High": data["h"],
        "Low": data["l"],
        "Close": data["c"],
        "Volume": data["v"]
    })
    return df

def analyze_ticker(symbol: str, vol_mult: float) -> dict | None:
    try:
        df = fetch_stock_ohlcv(symbol)
        if len(df) < 22:
            return None

        # Áp dụng hệ số nội suy Volume cho cây nến cuối cùng (hiện tại)
        df['Adj_Volume'] = df['Volume'].astype(float)
        df.loc[df.index[-1], 'Adj_Volume'] = df.loc[df.index[-1], 'Volume'] * vol_mult

        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        # MA20 của Volume tính trên 20 phiên trước đó để làm chuẩn khách quan
        df['Vol_MA20'] = df['Volume'].shift(1).rolling(20).mean()
        df['RVOL'] = df['Adj_Volume'] / df['Vol_MA20']

        candle_range = (df['High'] - df['Low']).replace(0, np.nan)
        close_pos = (df['Close'] - df['Low']) / candle_range
        df['Flow_Pressure'] = ((2 * df['Close'] - df['High'] - df['Low']) / candle_range) * df['RVOL']

        df['Dist_MA10_Pct'] = (df['Close'] - df['MA10']) / df['MA10'] * 100
        df['Dist_MA20_Pct'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['Room_To_MA20_Pct'] = (df['MA20'] - df['Close']) / df['Close'] * 100
        ma20_flat_or_up = df['MA20'] >= df['MA20'].shift(2) * 0.997

        # Quy tắc 4 Tín hiệu quanh MA10 - MA20
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
        intraday_pinbar_test = (
            (df['Low'] <= df['MA20'] * 1.01) &
            (close_pos >= 0.65) &
            (df['RVOL'] >= 0.85)
        )
        cond_test_buy = (
            ma20_flat_or_up &
            (df['Close'] > df['MA20']) &
            (df['Close'] > df['MA10']) &
            (df['Close'] > df['Open']) &
            ((prev_day_low_vol_test & (df['Close'] > df['High'].shift(1)) & (df['Flow_Pressure'] > 0.3)) | intraday_pinbar_test)
        )

        cond_t_plus_ma = (
            (df['MA10'] < df['MA20']) &
            (df['Close'].shift(1) <= df['MA10'].shift(1)) &
            (df['Close'] > df['MA10']) &
            (df['Close'] < df['MA20']) &
            (df['Room_To_MA20_Pct'] >= 3.5) &
            (df['Close'] > df['Open']) &
            (close_pos >= 0.65) &
            (df['RVOL'] >= 1.15)
        )

        cond_watch_rebound = (
            (df['Close'] < df['MA10']) &
            (df['Dist_MA10_Pct'] <= -4.0) &
            (close_pos >= 0.5) &
            ((df['RVOL'] < 0.75) | ((df['RVOL'] > 1.5) & (close_pos >= 0.7)))
        )

        conditions = [cond_failed_rebound, cond_test_buy, cond_t_plus_ma, cond_watch_rebound]
        choices = ['Hồi thất bại', 'Mua test thành công', 'Mua lướt T+ MA', 'Theo dõi hồi']
        df['Signal'] = np.select(conditions, choices, default='Trung tính')

        last = df.iloc[-1]
        prev = df.iloc[-2]
        pct_change = (last['Close'] - prev['Close']) / prev['Close'] * 100

        return {
            "symbol": symbol,
            "close": round(float(last['Close']), 2),
            "pct_change": round(float(pct_change), 2),
            "dist_ma10": round(float(last['Dist_MA10_Pct']), 2),
            "dist_ma20": round(float(last['Dist_MA20_Pct']), 2),
            "room_to_ma20": round(float(last['Room_To_MA20_Pct']), 2),
            "rvol": round(float(last['RVOL']), 2),
            "flow_pressure": round(float(last['Flow_Pressure']), 2),
            "signal": str(last['Signal'])
        }
    except Exception:
        return None

@app.get("/api/signals")
def get_signals(force: bool = Query(False)):
    now_ts = time.time()
    # Giữ Cache trong 180 giây (3 phút) trừ khi người dùng bấm nút Force Refresh
    if not force and CACHE["data"] and (now_ts - CACHE["timestamp"] < 180):
        return JSONResponse(CACHE["data"])

    vn_now = get_vn_time()
    vol_mult, session_status = get_intraday_vol_multiplier(vn_now)

    results = []
    for sym in WATCHLIST:
        item = analyze_ticker(sym, vol_mult)
        if item:
            results.append(item)

    # Tính toán độ rộng thị trường (Bán tháo vs Hồi phục)
    total = len(results) or 1
    bearish_count = sum(1 for r in results if r["dist_ma20"] < 0 and r["flow_pressure"] < 0)
    selloff_pct = round(bearish_count / total * 100, 1)

    payload = {
        "updated_at": vn_now.strftime("%H:%M:%S - %d/%m/%Y"),
        "session_status": session_status,
        "selloff_ratio": selloff_pct,
        "recovery_ratio": round(100 - selloff_pct, 1),
        "stocks": results
    }
    CACHE["timestamp"] = now_ts
    CACHE["data"] = payload
    return JSONResponse(payload)
