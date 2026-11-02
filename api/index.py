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

def scan_market_data(force: bool = False) -> dict:
    now_ts = time.time()
    if not force and CACHE["data"] and (now_ts - CACHE["timestamp"] < 180):
        return CACHE["data"]

    vn_now = get_vn_time()
    vol_mult, session_status = get_intraday_vol_multiplier(vn_now)

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(analyze_ticker, sym, vol_mult) for sym in WATCHLIST]
        results = [f.result() for f in futures if f.result() is not None]

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
    return payload


# =====================================================================
# 4. API DỮ LIỆU VÀ GỬI EMAIL CHỐNG DUPLICATE (LÚC 16:00)
# =====================================================================
@app.get("/api/signals")
def api_get_signals(force: bool = Query(False)):
    return JSONResponse(scan_market_data(force=force))

def build_email_html(data: dict, app_url: str) -> str:
    stocks = data["stocks"]
    
    def render_rows(sig_name: str, badge_color: str) -> str:
        group = [s for s in stocks if s["signal"] == sig_name]
        group.sort(key=lambda x: x["flow_pressure"], reverse=True)
        if not group:
            return f'<tr><td colspan="6" style="padding:10px; color:#64748b; text-align:center;">Không có mã nào trong nhóm {sig_name}</td></tr>'
        
        rows = ""
        for s in group:
            pct_color = "#10b981" if s["pct_change"] >= 0 else "#ef4444"
            room_txt = f'+{s["room_to_ma20"]}%' if s["room_to_ma20"] > 0 else "Trên MA20"
            rows += f"""
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding:8px 10px; font-weight:bold; color:#f8fafc;">{s['symbol']}</td>
                <td style="padding:8px 10px;"><span style="background:{badge_color}25; color:{badge_color}; border:1px solid {badge_color}; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:bold;">{sig_name}</span></td>
                <td style="padding:8px 10px; color:{pct_color}; font-weight:bold;">{s['close']} ({s['pct_change']:+}%)</td>
                <td style="padding:8px 10px; color:#cbd5e1;">{s['dist_ma10']:+}%</td>
                <td style="padding:8px 10px; color:#38bdf8; font-weight:bold;">{room_txt}</td>
                <td style="padding:8px 10px; color:#cbd5e1;">{s['rvol']}x (Flow: {s['flow_pressure']:+})</td>
            </tr>
            """
        return rows

    return f"""
    <div style="background-color:#0f172a; color:#f8fafc; font-family:Arial,sans-serif; padding:20px; max-width:700px; margin:0 auto; border-radius:12px;">
        <h2 style="color:#10b981; margin:0 0 5px 0;">BÁO CÁO TÍN HIỆU DẢI MA10 – MA20</h2>
        <p style="color:#94a3b8; font-size:12px; margin:0 0 15px 0;">Cập nhật: <b>{data['updated_at']}</b> | {data['session_status']}</p>
        
        <div style="background:#1e293b; padding:12px; border-radius:8px; margin-bottom:20px; font-size:13px;">
            <span>🔴 Áp lực Bán tháo / Dưới MA20: <b style="color:#f87171;">{data['selloff_ratio']}%</b></span>
            <span style="float:right;">🟢 Giữ nhịp / Hồi phục: <b style="color:#34d399;">{data['recovery_ratio']}%</b></span>
        </div>

        <h3 style="color:#34d399; font-size:14px; margin-bottom:6px;">1. NHÓM MUA TEST THÀNH CÔNG (Khỏe hơn thị trường - Ưu tiên số 1)</h3>
        <table style="width:100%; border-collapse:collapse; background:#1e293b; font-size:12px; margin-bottom:18px; border-radius:6px; overflow:hidden;">
            <tr style="background:#334155; color:#cbd5e1; text-align:left;">
                <th style="padding:8px 10px;">Mã</th><th style="padding:8px 10px;">Tín hiệu</th><th style="padding:8px 10px;">Giá</th><th style="padding:8px 10px;">Cách MA10</th><th style="padding:8px 10px;">Room MA20</th><th style="padding:8px 10px;">RVOL & Flow</th>
            </tr>
            {render_rows('Mua test thành công', '#10b981')}
        </table>

        <h3 style="color:#60a5fa; font-size:14px; margin-bottom:6px;">2. NHÓM MUA LƯỚT T+ MA (Vượt MA10, Dư địa lên MA20 >= 3.5%)</h3>
        <table style="width:100%; border-collapse:collapse; background:#1e293b; font-size:12px; margin-bottom:18px; border-radius:6px; overflow:hidden;">
            <tr style="background:#334155; color:#cbd5e1; text-align:left;">
                <th style="padding:8px 10px;">Mã</th><th style="padding:8px 10px;">Tín hiệu</th><th style="padding:8px 10px;">Giá</th><th style="padding:8px 10px;">Cách MA10</th><th style="padding:8px 10px;">Room MA20</th><th style="padding:8px 10px;">RVOL & Flow</th>
            </tr>
            {render_rows('Mua lướt T+ MA', '#3b82f6')}
        </table>

        <h3 style="color:#f87171; font-size:14px; margin-bottom:6px;">3. CẢNH BÁO HỒI THẤT BẠI (Chạm cản MA20 bị dội hoặc gãy lại MA10)</h3>
        <table style="width:100%; border-collapse:collapse; background:#1e293b; font-size:12px; margin-bottom:18px; border-radius:6px; overflow:hidden;">
            <tr style="background:#334155; color:#cbd5e1; text-align:left;">
                <th style="padding:8px 10px;">Mã</th><th style="padding:8px 10px;">Tín hiệu</th><th style="padding:8px 10px;">Giá</th><th style="padding:8px 10px;">Cách MA10</th><th style="padding:8px 10px;">Room MA20</th><th style="padding:8px 10px;">RVOL & Flow</th>
            </tr>
            {render_rows('Hồi thất bại', '#ef4444')}
        </table>

        <h3 style="color:#fbbf24; font-size:14px; margin-bottom:6px;">4. NHÓM THEO DÕI HỒI (Chiết khấu sâu dưới MA10, cạn cung/rút chân)</h3>
        <table style="width:100%; border-collapse:collapse; background:#1e293b; font-size:12px; margin-bottom:24px; border-radius:6px; overflow:hidden;">
            <tr style="background:#334155; color:#cbd5e1; text-align:left;">
                <th style="padding:8px 10px;">Mã</th><th style="padding:8px 10px;">Tín hiệu</th><th style="padding:8px 10px;">Giá</th><th style="padding:8px 10px;">Cách MA10</th><th style="padding:8px 10px;">Room MA20</th><th style="padding:8px 10px;">RVOL & Flow</th>
            </tr>
            {render_rows('Theo dõi hồi', '#f59e0b')}
        </table>

        <div style="text-align:center; margin-top:20px;">
            <a href="{app_url}" style="background:#4f46e5; color:#ffffff; padding:12px 24px; text-decoration:none; border-radius:8px; font-weight:bold; font-size:13px; display:inline-block;">
                📊 Mở Bản Đồ Tương Tác 2D Trên Web
            </a>
        </div>
    </div>
    """

@app.get("/api/cron-email")
def trigger_email_report(request: Request):
    global LAST_EMAIL_SENT_TIMESTAMP
    
    # Chống gửi email trùng lặp trong vòng 10 phút
    current_ts = time.time()
    if current_ts - LAST_EMAIL_SENT_TIMESTAMP < 600:
        return JSONResponse({
            "status": "ignored",
            "message": "Đã chặn yêu cầu gửi trùng lặp (Email cho phiên này đã được gửi cách đây chưa lâu)."
        }, status_code=200)

    if CRON_SECRET:
        auth_header = request.headers.get("Authorization")
        if auth_header != f"Bearer {CRON_SECRET}":
            return JSONResponse({"status": "error", "message": "Unauthorized Cron"}, status_code=401)

    if not RESEND_API_KEY:
        return JSONResponse({
            "status": "error",
            "message": "Chưa cấu hình biến môi trường RESEND_API_KEY trên Vercel."
        }, status_code=400)

    data = scan_market_data(force=True)
    app_url = str(request.base_url).rstrip("/")
    html_body = build_email_html(data, app_url)

    cnt_test = sum(1 for s in data["stocks"] if s["signal"] == "Mua test thành công")
    cnt_tplus = sum(1 for s in data["stocks"] if s["signal"] == "Mua lướt T+ MA")
    cnt_fail = sum(1 for s in data["stocks"] if s["signal"] == "Hồi thất bại")

    subject = f"[VN-INDEX {data['updated_at'][-10:]}] 🟢 {cnt_test} Mua Test | 🔵 {cnt_tplus} Lướt T+ | 🔴 {cnt_fail} Hồi Thất Bại"

    res = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "from": EMAIL_FROM,
            "to": [e.strip() for e in EMAIL_TO.split(",")],
            "subject": subject,
            "html": html_body
        },
        timeout=10
    )

    if res.status_code in (200, 201):
        LAST_EMAIL_SENT_TIMESTAMP = time.time()
        return JSONResponse({
            "status": "success", 
            "message": f"Đã gửi báo cáo thành công lúc 16:00 tới {EMAIL_TO}", 
            "resend_id": res.json().get("id")
        })
    
    return JSONResponse({"status": "error", "message": res.text}, status_code=500)


# =====================================================================
# 5. GIAO DIỆN WEB APP TƯƠNG TÁC (PLOTLY.JS)
# =====================================================================
@app.get("/", response_class=HTMLResponse)
def render_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VN-INDEX Signal Map (MA10 - MA20)</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 p-4 md:p-6 font-sans">
        <div class="max-w-7xl mx-auto space-y-5">
            
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-900 p-4 rounded-xl border border-slate-800 gap-4">
                <div>
                    <div class="flex items-center gap-3">
                        <span class="h-3 w-3 rounded-full bg-emerald-400 animate-ping"></span>
                        <h1 class="text-xl md:text-2xl font-bold tracking-tight text-white">
                            BẢN ĐỒ TÍN HIỆU DẢI MA10 – MA20
                        </h1>
                    </div>
                    <p id="status-bar" class="text-xs md:text-sm text-slate-400 mt-1">Đang kết nối API dữ liệu...</p>
                </div>
                <div class="flex flex-wrap items-center gap-2 w-full md:w-auto">
                    <input type="text" id="search-input" oninput="applyFilters()" placeholder="Tìm mã (VD: TCH, STB...)" 
                           class="bg-slate-800 border border-slate-700 text-sm rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500 w-full sm:w-44">
                    <button onclick="loadData(true)" id="btn-refresh"
                            class="bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-lg font-semibold text-sm transition whitespace-nowrap shadow-lg">
                        🔄 Quét Live
                    </button>
                    <button onclick="sendEmailNow()" id="btn-email"
                            class="bg-emerald-700 hover:bg-emerald-600 px-3 py-2 rounded-lg font-semibold text-sm transition whitespace-nowrap shadow-lg">
                        📩 Gửi Mail Ngay
                    </button>
                </div>
            </div>

            <div class="bg-slate-900 p-4 rounded-xl border border-slate-800">
                <div class="flex justify-between text-xs font-semibold mb-2">
                    <span id="selloff-label" class="text-red-400">Áp lực Bán tháo / Dưới MA20: --%</span>
                    <span id="recovery-label" class="text-emerald-400">Giữ nhịp / Hồi phục: --%</span>
                </div>
                <div class="w-full h-3 bg-slate-800 rounded-full overflow-hidden flex">
                    <div id="bar-selloff" class="bg-red-500 h-full transition-all duration-500" style="width: 65%"></div>
                    <div id="bar-recovery" class="bg-emerald-500 h-full transition-all duration-500" style="width: 35%"></div>
                </div>
            </div>

            <div class="flex flex-wrap gap-2">
                <button onclick="setSignalFilter('ALL')" class="bg-slate-800 hover:bg-slate-700 border border-slate-600 px-3 py-1.5 rounded-lg text-xs font-bold text-white">
                    Tất cả (<span id="count-ALL">0</span>)
                </button>
                <button onclick="setSignalFilter('Mua test thành công')" class="bg-emerald-950/60 hover:bg-emerald-900/70 border border-emerald-500/50 text-emerald-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                    🟢 Mua test thành công (<span id="count-test">0</span>)
                </button>
                <button onclick="setSignalFilter('Mua lướt T+ MA')" class="bg-blue-950/60 hover:bg-blue-900/70 border border-blue-500/50 text-blue-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                    🔵 Mua lướt T+ MA (<span id="count-tplus">0</span>)
                </button>
                <button onclick="setSignalFilter('Theo dõi hồi')" class="bg-amber-950/60 hover:bg-amber-900/70 border border-amber-500/50 text-amber-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                    🟡 Theo dõi hồi (<span id="count-watch">0</span>)
                </button>
                <button onclick="setSignalFilter('Hồi thất bại')" class="bg-red-950/60 hover:bg-red-900/70 border border-red-500/50 text-red-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                    🔴 Hồi thất bại (<span id="count-fail">0</span>)
                </button>
            </div>

            <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 shadow-inner">
                <div id="scatter-map" style="width:100%; height:540px;"></div>
            </div>

            <div class="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
                <div class="overflow-x-auto">
                    <table class="w-full text-sm text-left">
                        <thead class="text-xs uppercase bg-slate-800/80 text-slate-400 border-b border-slate-700">
                            <tr>
                                <th class="p-3">Mã CP</th>
                                <th class="p-3">Trạng thái Tín hiệu</th>
                                <th class="p-3">Giá Khớp (% Phiên)</th>
                                <th class="p-3">% Cách MA10</th>
                                <th class="p-3">Dư địa lên MA20 (Room T+)</th>
                                <th class="p-3">RVOL (Ước tính)</th>
                                <th class="p-3">Áp lực Dòng tiền (Flow)</th>
                            </tr>
                        </thead>
                        <tbody id="stock-table" class="divide-y divide-slate-800"></tbody>
                    </table>
                </div>
            </div>

        </div>

        <script>
            let RAW_STOCKS = [];
            let CURRENT_FILTER = 'ALL';

            const COLOR_MAP = {
                'Mua test thành công': '#10b981',
                'Mua lướt T+ MA': '#3b82f6',
                'Theo dõi hồi': '#f59e0b',
                'Hồi thất bại': '#ef4444',
                'Trung tính': '#475569'
            };

            async function loadData(force = false) {
                const btn = document.getElementById('btn-refresh');
                btn.disabled = true;
                btn.innerText = "⏳ Đang quét...";

                try {
                    const res = await fetch(`/api/signals?force=${force}`);
                    const data = await res.json();
                    RAW_STOCKS = data.stocks || [];

                    document.getElementById('status-bar').innerHTML = 
                        `Cập nhật: <b class="text-white">${data.updated_at}</b> | Phiên: <span class="text-amber-400 font-semibold">${data.session_status}</span>`;
                    
                    document.getElementById('selloff-label').innerText = `Áp lực Bán tháo: ${data.selloff_ratio}%`;
                    document.getElementById('recovery-label').innerText = `Giữ nhịp / Hồi phục: ${data.recovery_ratio}%`;
                    document.getElementById('bar-selloff').style.width = `${data.selloff_ratio}%`;
                    document.getElementById('bar-recovery').style.width = `${data.recovery_ratio}%`;

                    document.getElementById('count-ALL').innerText = RAW_STOCKS.length;
                    document.getElementById('count-test').innerText = RAW_STOCKS.filter(s => s.signal === 'Mua test thành công').length;
                    document.getElementById('count-tplus').innerText = RAW_STOCKS.filter(s => s.signal === 'Mua lướt T+ MA').length;
                    document.getElementById('count-watch').innerText = RAW_STOCKS.filter(s => s.signal === 'Theo dõi hồi').length;
                    document.getElementById('count-fail').innerText = RAW_STOCKS.filter(s => s.signal === 'Hồi thất bại').length;

                    applyFilters();
                } catch (err) {
                    document.getElementById('status-bar').innerText = "❌ Lỗi kết nối dữ liệu.";
                } finally {
                    btn.disabled = false;
                    btn.innerText = "🔄 Quét Live";
                }
            }

            async function sendEmailNow() {
                const btn = document.getElementById('btn-email');
                btn.disabled = true;
                btn.innerText = "⏳ Đang gửi...";
                try {
                    const res = await fetch('/api/cron-email');
                    const data = await res.json();
                    alert(data.message || JSON.stringify(data));
                } catch (e) {
                    alert("Lỗi khi gọi API gửi mail!");
                } finally {
                    btn.disabled = false;
                    btn.innerText = "📩 Gửi Mail Ngay";
                }
            }

            function setSignalFilter(sig) {
                CURRENT_FILTER = sig;
                applyFilters();
            }

            function applyFilters() {
                const keyword = document.getElementById('search-input').value.trim().toUpperCase();
                const filtered = RAW_STOCKS.filter(s => {
                    const matchSig = (CURRENT_FILTER === 'ALL') || (s.signal === CURRENT_FILTER);
                    const matchKey = !keyword || s.symbol.includes(keyword);
                    return matchSig && matchKey;
                });

                renderPlot(filtered);
                renderTable(filtered);
            }

            function renderPlot(stocks) {
                const traces = [];
                Object.keys(COLOR_MAP).forEach(sig => {
                    const group = stocks.filter(s => s.signal === sig);
                    if (group.length === 0) return;

                    traces.push({
                        x: group.map(s => s.dist_ma20),
                        y: group.map(s => s.flow_pressure),
                        text: group.map(s => s.symbol),
                        customdata: group.map(s => [s.close, s.pct_change, s.dist_ma10, s.room_to_ma20, s.rvol]),
                        mode: 'markers+text',
                        textposition: 'top center',
                        textfont: { color: '#f8fafc', size: 11, weight: 'bold' },
                        name: `${sig} (${group.length})`,
                        marker: {
                            size: group.map(s => Math.min(24, Math.max(11, s.rvol * 10))),
                            color: COLOR_MAP[sig],
                            opacity: sig === 'Trung tính' ? 0.45 : 0.95,
                            line: { width: 1.5, color: '#0f172a' }
                        },
                        hovertemplate: 
                            '<b>%{text}</b> — Giá: %{customdata[0]} (%{customdata[1]}%)<br>' +
                            '• Vị thế MA20: <b>%{x}%</b> | Vị thế MA10: <b>%{customdata[2]}%</b><br>' +
                            '• Dư địa T+ tới MA20: <b>%{customdata[3]}%</b><br>' +
                            '• RVOL: <b>%{customdata[4]}x</b> | Áp lực Flow: <b>%{y}</b><extra></extra>'
                    });
                });

                const layout = {
                    title: {
                        text: 'TỌA ĐỘ VỊ THẾ MA20 (TRỤC X) vs. ÁP LỰC DÒNG TIỀN THỰC (TRỤC Y)',
                        font: { color: '#e2e8f0', size: 14 }
                    },
                    paper_bgcolor: '#0f172a',
                    plot_bgcolor: '#020617',
                    margin: { l: 55, r: 30, t: 50, b: 60 },
                    xaxis: {
                        title: 'Khoảng cách giá so với MA20 (%)  [ < 0: Dưới cản MA20 | > 0: Đã vượt MA20 ]',
                        gridcolor: '#1e293b', zerolinecolor: '#f8fafc', zerolinewidth: 2, color: '#94a3b8'
                    },
                    yaxis: {
                        title: 'Áp lực Dòng tiền (Flow Pressure)  [ > 0: Cầu đẩy | < 0: Cung xả ]',
                        gridcolor: '#1e293b', zerolinecolor: '#f8fafc', zerolinewidth: 1.5, color: '#94a3b8'
                    },
                    shapes: [
                        {
                            type: 'line', x0: -3.5, x1: -3.5, y0: -3, y1: 3,
                            line: { color: '#3b82f6', width: 1.5, dash: 'dot' }
                        }
                    ],
                    annotations: [
                        { x: -3.5, y: 2.3, text: 'Ranh giới đủ biên T+ (-3.5% tới MA20)', showarrow: false, font: { color: '#60a5fa', size: 11 } },
                        { x: 4, y: 2.3, text: 'VÙNG MUA TEST THÀNH CÔNG (Trên MA20)', showarrow: false, font: { color: '#34d399', size: 11 } },
                        { x: -6, y: -2.2, text: 'VÙNG BÁN THÁO / HỒI THẤT BẠI', showarrow: false, font: { color: '#f87171', size: 11 } }
                    ],
                    legend: { font: { color: '#cbd5e1' }, orientation: 'h', y: -0.18 }
                };

                Plotly.newPlot('scatter-map', traces, layout, { responsive: true, displayModeBar: false });
            }

            function renderTable(stocks) {
                const priority = {'Hồi thất bại': 1, 'Mua test thành công': 2, 'Mua lướt T+ MA': 3, 'Theo dõi hồi': 4, 'Trung tính': 5};
                const sorted = [...stocks].sort((a, b) => (priority[a.signal] - priority[b.signal]) || (b.flow_pressure - a.flow_pressure));

                const tbody = document.getElementById('stock-table');
                tbody.innerHTML = sorted.map(s => `
                    <tr class="hover:bg-slate-800/60 transition">
                        <td class="p-3 font-bold text-white">${s.symbol}</td>
                        <td class="p-3">
                            <span style="background:${COLOR_MAP[s.signal]}20; color:${COLOR_MAP[s.signal]}; border:1px solid ${COLOR_MAP[s.signal]}80" 
                                  class="px-2.5 py-1 rounded-md text-xs font-bold">
                                ${s.signal}
                            </span>
                        </td>
                        <td class="p-3 font-medium ${s.pct_change >= 0 ? 'text-emerald-400' : 'text-red-400'}">
                            ${s.close} (${s.pct_change > 0 ? '+' : ''}${s.pct_change}%)
                        </td>
                        <td class="p-3 ${s.dist_ma10 >= 0 ? 'text-sky-300' : 'text-slate-400'}">
                            ${s.dist_ma10 > 0 ? '+' : ''}${s.dist_ma10}%
                        </td>
                        <td class="p-3 font-semibold ${s.room_to_ma20 >= 3.5 ? 'text-blue-400' : 'text-slate-500'}">
                            ${s.room_to_ma20 > 0 ? 'Còn +' + s.room_to_ma20 + '%' : 'Đã nằm trên MA20'}
                        </td>
                        <td class="p-3 font-mono">${s.rvol}x</td>
                        <td class="p-3 font-mono font-bold ${s.flow_pressure >= 0 ? 'text-emerald-400' : 'text-red-400'}">
                            ${s.flow_pressure > 0 ? '+' : ''}${s.flow_pressure}
                        </td>
                    </tr>
                `).join('');
            }

            loadData(false);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
