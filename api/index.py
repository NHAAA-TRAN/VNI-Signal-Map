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
# =====================================================================
# BƯỚC 4: ROUTE CHÍNH TRẢ VỀ GIAO DIỆN WEB APP TƯƠNG TÁC (PLOTLY.JS)
# Dán đoạn này vào cuối file api/index.py
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
        <!-- Nhúng TailwindCSS để làm giao diện Dark Mode tài chính -->
        <script src="https://cdn.tailwindcss.com"></script>
        <!-- Nhúng Plotly.js để vẽ Bản đồ Tín hiệu 2D tương tác -->
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 p-4 md:p-6 font-sans">
        <div class="max-w-7xl mx-auto space-y-5">
            
            <!-- 1. THANH HEADER & TRẠNG THÁI PHIÊN -->
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-900 p-4 rounded-xl border border-slate-800 gap-4">
                <div>
                    <div class="flex items-center gap-3">
                        <span class="h-3 w-3 rounded-full bg-emerald-400 animate-ping"></span>
                        <h1 class="text-xl md:text-2xl font-bold tracking-tight text-white">
                            BẢN ĐỒ TÍN HIỆU DẢI MA10 – MA20
                        </h1>
                    </div>
                    <p id="status-bar" class="text-xs md:text-sm text-slate-400 mt-1">
                        Đang kết nối API dữ liệu...
                    </p>
                </div>
                <div class="flex items-center gap-3 w-full md:w-auto">
                    <input type="text" id="search-input" oninput="applyFilters()" placeholder="Tìm mã (VD: TCH, STB...)" 
                           class="bg-slate-800 border border-slate-700 text-sm rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500 w-full md:w-48">
                    <button onclick="loadData(true)" id="btn-refresh"
                            class="bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 px-4 py-2 rounded-lg font-semibold text-sm transition whitespace-nowrap shadow-lg">
                        🔄 Quét Live
                    </button>
                </div>
            </div>

            <!-- 2. THANH ĐO ĐỘ RỘNG THỊ TRƯỜNG (65% BÁN THÁO VS 35% HỒI PHỤC) -->
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

            <!-- 3. CÁC NÚT BẤM LỌC NHANH THEO 4 TÍN HIỆU -->
            <div class="flex flex-wrap gap-2" id="filter-buttons">
                <button onclick="setSignalFilter('ALL')" class="sig-btn bg-slate-800 border border-slate-600 px-3 py-1.5 rounded-lg text-xs font-bold text-white">
                    Tất cả (<span id="count-ALL">0</span>)
                </button>
                <button onclick="setSignalFilter('Mua test thành công')" class="sig-btn bg-emerald-950/60 border border-emerald-500/50 text-emerald-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                    🟢 Mua test thành công (<span id="count-test">0</span>)
                </button>
                <button onclick="setSignalFilter('Mua lướt T+ MA')" class="sig-btn bg-blue-950/60 border border-blue-500/50 text-blue-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                    🔵 Mua lướt T+ MA (<span id="count-tplus">0</span>)
                </button>
                <button onclick="setSignalFilter('Theo dõi hồi')" class="sig-btn bg-amber-950/60 border border-amber-500/50 text-amber-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                    🟡 Theo dõi hồi (<span id="count-watch">0</span>)
                </button>
                <button onclick="setSignalFilter('Hồi thất bại')" class="sig-btn bg-red-950/60 border border-red-500/50 text-red-300 px-3 py-1.5 rounded-lg text-xs font-bold">
                    🔴 Hồi thất bại (<span id="count-fail">0</span>)
                </button>
            </div>

            <!-- 4. KHUNG VẼ BẢN ĐỒ TỌA ĐỘ 2D PLOTLY -->
            <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 shadow-inner">
                <div id="scatter-map" style="width:100%; height:540px;"></div>
            </div>

            <!-- 5. BẢNG CHI TIẾT CỔ PHIẾU -->
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

        <!-- JAVASCRIPT XỬ LÝ DỮ LIỆU & VẼ BIỂU ĐỒ -->
        <script>
            let RAW_STOCKS = [];
            let CURRENT_FILTER = 'ALL';

            const COLOR_MAP = {
                'Mua test thành công': '#10b981', // Xanh lá
                'Mua lướt T+ MA': '#3b82f6',      // Xanh dương
                'Theo dõi hồi': '#f59e0b',        // Vàng cam
                'Hồi thất bại': '#ef4444',        // Đỏ
                'Trung tính': '#475569'           // Xám
            };

            async function loadData(force = false) {
                const btn = document.getElementById('btn-refresh');
                btn.disabled = true;
                btn.innerText = "⏳ Đang quét...";
                document.getElementById('status-bar').innerText = "Đang tải dữ liệu và tính toán dải MA10 - MA20...";

                try {
                    const res = await fetch(`/api/signals?force=${force}`);
                    const data = await res.json();
                    RAW_STOCKS = data.stocks || [];

                    // Cập nhật thanh trạng thái & thanh tỷ lệ Bán tháo / Hồi phục
                    document.getElementById('status-bar').innerHTML = 
                        `Cập nhật: <b class="text-white">${data.updated_at}</b> | Phiên: <span class="text-amber-400 font-semibold">${data.session_status}</span>`;
                    
                    document.getElementById('selloff-label').innerText = `Áp lực Bán tháo: ${data.selloff_ratio}%`;
                    document.getElementById('recovery-label').innerText = `Giữ nhịp / Hồi phục: ${data.recovery_ratio}%`;
                    document.getElementById('bar-selloff').style.width = `${data.selloff_ratio}%`;
                    document.getElementById('bar-recovery').style.width = `${data.recovery_ratio}%`;

                    // Đếm số lượng từng loại tín hiệu
                    document.getElementById('count-ALL').innerText = RAW_STOCKS.length;
                    document.getElementById('count-test').innerText = RAW_STOCKS.filter(s => s.signal === 'Mua test thành công').length;
                    document.getElementById('count-tplus').innerText = RAW_STOCKS.filter(s => s.signal === 'Mua lướt T+ MA').length;
                    document.getElementById('count-watch').innerText = RAW_STOCKS.filter(s => s.signal === 'Theo dõi hồi').length;
                    document.getElementById('count-fail').innerText = RAW_STOCKS.filter(s => s.signal === 'Hồi thất bại').length;

                    applyFilters();
                } catch (err) {
                    document.getElementById('status-bar').innerText = "❌ Lỗi kết nối dữ liệu. Vui lòng thử lại.";
                } finally {
                    btn.disabled = false;
                    btn.innerText = "🔄 Quét Live";
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
                            size: group.map(s => Math.min(24, Math.max(11, s.rvol * 10))), // Bong bóng to theo RVOL
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
                    // Vẽ đường ranh giới biên độ T+ (-3.5% cách MA20)
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

            // Tự động tải dữ liệu ngay khi mở trang
            loadData(false);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
