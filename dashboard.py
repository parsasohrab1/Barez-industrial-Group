"""
TVDI Dashboard — داشبورد حرفه‌ای سامانه بازرسی تایر
گروه صنعتی بارز

اجرا:
    pip install streamlit plotly pandas numpy requests pillow
    streamlit run dashboard.py --server.port 8501
"""

import base64
import io
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageDraw

# =============================================================================
# CONFIG
# =============================================================================

API_URL_DEFAULT = "http://127.0.0.1:8001"
CLASSES = ["good", "bubble", "tread_uneven", "layer_separation"]
CLASS_LABELS_FA = {
    "good": "سالم",
    "bubble": "حباب ظاهری آزمایشگاهی",
    "tread_uneven": "ناهمواری آج",
    "layer_separation": "جدایی لایه",
}
SEVERITY_FA = {"low": "کم", "medium": "متوسط", "high": "زیاد"}
RISK_FA = {"low": "کم", "medium": "متوسط", "high": "زیاد"}
GRADE_FA = {"A": "الف", "B": "ب", "C": "ج"}

BRAND_RED = "#C8102E"
BRAND_DARK = "#1A1A1A"
PALETTE = {
    "good": "#10B981",
    "bubble": "#F59E0B",
    "tread_uneven": "#3B82F6",
    "layer_separation": "#EF4444",
}

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TVDI | گروه صنعتی بارز",
    page_icon="TV",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# RTL + PROFESSIONAL CSS
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

    html, body, [class*="css"], .stApp {
        font-family: 'Vazirmatn', 'Tahoma', sans-serif !important;
        direction: rtl;
        text-align: right;
    }
    .stApp { background: #F4F5F7; }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
        max-width: 1240px;
    }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    #MainMenu, footer { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background: #14171C;
        border-left: 1px solid #2A2F37;
        min-width: 16.5rem;
        max-width: 16.5rem;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #D1D5DB !important;
    }
    section[data-testid="stSidebar"] label { font-size: .78rem !important; }
    section[data-testid="stSidebar"] hr { border-color: #2A2F37; margin: .8rem 0; }

    .side-brand { padding: 4px 0 16px; border-bottom: 1px solid #2A2F37; margin-bottom: 14px; }
    .side-brand .mark {
        color: #FFFFFF; font-weight: 700; font-size: 1.05rem; letter-spacing: .12em;
    }
    .side-brand .side-sub { color: #9CA3AF; font-size: .75rem; margin-top: 2px; }
    .side-label {
        color: #9CA3AF; font-size: .72rem; font-weight: 600;
        letter-spacing: .04em; margin: 12px 0 8px;
    }

    .appbar {
        display: flex; justify-content: space-between; align-items: flex-end;
        gap: 16px; padding-bottom: 12px; margin-bottom: 14px;
        border-bottom: 1px solid #E5E7EB;
    }
    .eyebrow {
        color: #6B7280; font-size: .72rem; font-weight: 600; letter-spacing: .06em;
    }
    .appbar h1 { margin: 2px 0 0; font-size: 1.35rem; font-weight: 700; color: #111827; }
    .appbar-meta { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .chip {
        background: #FFFFFF; border: 1px solid #E5E7EB; color: #374151;
        border-radius: 999px; padding: 4px 10px; font-size: .75rem; font-weight: 600;
    }

    .kpi-grid {
        display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 10px; margin-bottom: 14px;
    }
    .kpi-card {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px;
        padding: 12px 12px 10px; border-right: 3px solid #C8102E;
    }
    .kpi-label { font-size: .75rem; color: #6B7280; margin-bottom: 6px; }
    .kpi-value {
        font-size: 1.45rem; font-weight: 700; color: #111827; line-height: 1.15;
        font-variant-numeric: tabular-nums;
    }
    .kpi-delta { font-size: .72rem; margin-top: 6px; color: #6B7280; }
    .kpi-delta.up { color: #047857; }
    .kpi-delta.down { color: #B91C1C; }

    .panel {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px;
        padding: 14px 14px 6px; margin-bottom: 8px;
    }
    .panel h3 { margin: 0 0 10px; font-size: .92rem; font-weight: 700; color: #111827; }
    .statrow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .stat .k { font-size: .72rem; color: #6B7280; }
    .stat .v { font-size: 1rem; font-weight: 700; color: #111827; font-variant-numeric: tabular-nums; margin-top: 2px; }
    .note { color: #6B7280; font-size: .75rem; line-height: 1.7; margin: 10px 0 4px; }
    .strip {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px;
        padding: 10px 14px; margin: 4px 0 16px; color: #374151; font-size: .8rem;
        display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    }
    .flow { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 8px; margin: 8px 0 14px; }
    .step { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 10px 10px 8px; }
    .step .name { font-size: .78rem; font-weight: 700; color: #111827; }
    .step .state { font-size: .7rem; color: #6B7280; margin-top: 4px; }
    .lamp { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-left: 6px; }
    .lamp-ok { background: #047857; }
    .lamp-part { background: #B45309; }
    .lamp-none { background: #9CA3AF; }
    @media (max-width: 1100px) {
        .flow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .72rem; font-weight: 600; }
    .badge-high { background: #FEE2E2; color: #991B1B; }
    .badge-medium { background: #FEF3C7; color: #92400E; }
    .badge-low { background: #D1FAE5; color: #065F46; }

    .section-title {
        font-size: .92rem; font-weight: 700; color: #111827;
        margin: 4px 0 8px; padding: 0; border: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: transparent; padding: 0;
        border-bottom: 1px solid #E5E7EB; border-radius: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 0; padding: 10px 16px;
        font-weight: 600; font-size: .86rem; color: #6B7280;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important; color: #C8102E !important;
        border-bottom: 2px solid #C8102E;
    }
    .stButton > button {
        background: #C8102E; color: white; border: none; border-radius: 6px;
        font-weight: 600; font-family: 'Vazirmatn', sans-serif;
    }
    .stButton > button:hover { background: #A00D25; color: white; }
    [data-testid="stPlotlyChart"] {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 4px 4px 0;
    }
    [data-testid="stDataFrame"] { border: 1px solid #E5E7EB; border-radius: 8px; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600;
    }
    .status-online { background: #D1FAE5; color: #065F46; }
    .status-offline { background: #FEE2E2; color: #991B1B; }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; }
    .status-dot-online { background: #059669; }
    .status-dot-offline { background: #DC2626; }
    @media (max-width: 1100px) {
        .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# API HELPERS
# =============================================================================

def check_api_health(url: str) -> dict | None:
    try:
        response = requests.get(f"{url}/health", timeout=2)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    return response.json()


def _dev_credentials() -> dict | None:
    path = Path(__file__).resolve().parent / "data" / "dev-credentials.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def login(url: str, username: str, password: str) -> str:
    response = requests.post(
        f"{url}/auth/login",
        json={"username": username, "password": password},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def fetch_inspections(url: str, token: str, hours: int) -> list:
    response = requests.get(
        f"{url}/inspections",
        params={"hours": hours},
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def call_predict(url: str, image: Image.Image, metadata: dict, token: str) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "metadata": metadata,
    }
    response = requests.post(
        f"{url}/predict-json",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def call_predict_xray(url: str, image: Image.Image, token: str) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    response = requests.post(
        f"{url}/predict-xray",
        files={"file": ("tire.jpg", buf.getvalue(), "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _draw_regions(image: Image.Image, regions: list) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for region in regions:
        cx, cy, box_w, box_h = region["cx"], region["cy"], region["w"], region["h"]
        x0 = (cx - box_w / 2) * width
        y0 = (cy - box_h / 2) * height
        x1 = (cx + box_w / 2) * width
        y1 = (cy + box_h / 2) * height
        draw.rectangle([x0, y0, x1, y1], outline="#C8102E", width=2)
    return canvas


INSPECTION_COLUMNS = [
    "tire_id", "batch_id", "machine_id", "mold_id", "material_lot",
    "build_temp", "build_pressure", "cure_temp", "cure_time", "timestamp",
    "defect_type", "defect_probability", "severity", "risk_score",
    "risk_level", "grade", "action", "model_source",
]


def empty_inspections() -> pd.DataFrame:
    return pd.DataFrame(columns=INSPECTION_COLUMNS)


def _delta(current: float, previous: float, lower_is_better: bool = False) -> tuple[str, str]:
    if previous is None or pd.isna(previous):
        return "دوره قبل ندارد", ""
    if previous == 0:
        return "دوره قبل صفر است", ""
    change = (current - previous) / abs(previous) * 100
    if change == 0:
        return "بدون تغییر نسبت به دوره قبل", ""
    improved = change < 0 if lower_is_better else change > 0
    arrow = "▲" if change > 0 else "▼"
    return f"{arrow} {abs(change):.1f}٪ نسبت به دوره قبل", "up" if improved else "down"


# =============================================================================
# DEMO DATA
# =============================================================================

def _make_demo_image(defect_type: str, size: int = 128) -> Image.Image:
    rng = np.random.default_rng(random.randint(0, 10**6))
    img = rng.normal(128, 8, (size, size))
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    if defect_type == "bubble":
        cx, cy = rng.integers(20, size - 20, 2)
        r = int(rng.integers(6, 18))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=220)
    elif defect_type == "tread_uneven":
        arr = np.asarray(pil, dtype=float)
        y, x = np.mgrid[0:size, 0:size]
        arr += 20 * np.sin(2 * np.pi * x / rng.integers(8, 20))
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif defect_type == "layer_separation":
        x1, y1 = rng.integers(0, size, 2)
        x2, y2 = rng.integers(0, size, 2)
        draw.line([int(x1), int(y1), int(x2), int(y2)], fill=25, width=4)

    return pil


@st.cache_data(ttl=30)
def generate_demo_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weights = [0.60, 0.18, 0.14, 0.08]
    now = datetime.now()
    rows = []

    for i in range(n):
        defect_type = rng.choice(CLASSES, p=weights)
        ts = now - timedelta(minutes=int(rng.integers(0, 60 * 48)))
        meta = {
            "tire_id": f"TIRE-{i:05d}",
            "batch_id": f"B{int(rng.integers(1, 9)):03d}",
            "machine_id": f"M{int(rng.integers(1, 6)):02d}",
            "mold_id": f"D{int(rng.integers(1, 9)):02d}",
            "material_lot": f"L{int(rng.integers(1, 7)):02d}",
            "build_temp": float(rng.uniform(140, 170)),
            "build_pressure": float(rng.uniform(8, 14)),
            "cure_temp": float(rng.uniform(150, 180)),
            "cure_time": float(rng.uniform(10, 20)),
        }
        base_prob = 0.05 if defect_type == "good" else float(rng.uniform(0.55, 0.98))
        severity = (
            "high" if defect_type == "layer_separation"
            else "medium" if defect_type in ("bubble", "tread_uneven")
            else "low"
        )
        risk = base_prob
        if meta["build_temp"] > 168: risk = min(0.99, risk + 0.08)
        if meta["build_pressure"] < 9.5: risk = min(0.99, risk + 0.08)
        if meta["cure_temp"] < 155 or meta["cure_temp"] > 175: risk = min(0.99, risk + 0.10)
        if meta["cure_time"] < 12: risk = min(0.99, risk + 0.07)
        risk_level = "high" if risk >= 0.75 else "medium" if risk >= 0.45 else "low"

        if defect_type == "good" and base_prob < 0.35: grade, action = "A", "پذیرش"
        elif defect_type == "layer_separation" or base_prob >= 0.90: grade, action = "C", "ضایعات/بازکاری"
        elif base_prob >= 0.65 or severity == "high": grade, action = "B", "بازبینی"
        else: grade, action = "A", "پذیرش با پایش"

        rows.append({
            **meta, "timestamp": ts,
            "defect_type": defect_type, "defect_probability": base_prob,
            "severity": severity, "risk_score": risk,
            "risk_level": risk_level, "grade": grade, "action": action,
            "model_source": "sample",
        })

    return pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)


# =============================================================================
# SIDEBAR
# =============================================================================

def _frame(fig):
    fig.update_layout(
        margin=dict(l=8, r=8, t=36, b=8),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Vazirmatn, Tahoma, sans-serif", size=12, color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(font_family="Vazirmatn, Tahoma"),
    )
    if not any(getattr(trace, "type", "") == "pie" for trace in fig.data):
        fig.update_xaxes(gridcolor="#F3F4F6", linecolor="#E5E7EB", zeroline=False)
        fig.update_yaxes(gridcolor="#F3F4F6", linecolor="#E5E7EB", zeroline=False)
    return fig


with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="mark">TVDI</div>
            <div class="side-sub">بازرسی بینایی تایر</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">اتصال</div>', unsafe_allow_html=True)
    api_url = st.text_input("آدرس سرویس API", value=API_URL_DEFAULT)
    health = check_api_health(api_url)
    token = None
    load_error = None
    if health:
        creds = _dev_credentials()
        if creds:
            try:
                token = login(api_url, creds["username"], creds["password"])
            except Exception as exc:
                load_error = f"ورود به API ناموفق بود: {exc}"
        else:
            load_error = "API بالاست، ولی فایل ورود توسعه هنوز ساخته نشده است."

    if token:
        inference = (health or {}).get("checks", {}).get("inference", "نامشخص")
        opcua = (health or {}).get("checks", {}).get("opcua", "fail")
        mqtt = (health or {}).get("checks", {}).get("mqtt", "fail")
        if opcua == "lab" and mqtt == "lab":
            line_text = "فیکسچر محلی"
            line_detail = "OPC و MQTT روی همین رایانه هستند، نه PLC خط."
        elif opcua == "ok" and mqtt == "ok":
            line_text = "خط متصل"
            line_detail = "OPC و MQTT پاسخ می‌دهند."
        else:
            line_text = "اتصال خط قطع"
            line_detail = "OPC یا MQTT در دسترس نیست."
        st.markdown(
            f'<span class="status-pill status-online"><span class="status-dot status-dot-online"></span>مدل {inference}</span>',
            unsafe_allow_html=True,
        )
        st.caption(line_detail)
        data_source = "api"
    else:
        st.markdown(
            '<span class="status-pill status-offline"><span class="status-dot status-dot-offline"></span>آفلاین — داده نمونه</span>',
            unsafe_allow_html=True,
        )
        data_source = "demo"
        line_text = "سرویس قطع"
        if load_error:
            st.caption(load_error)

    st.markdown('<div class="side-label">فیلتر عملیات</div>', unsafe_allow_html=True)

    if data_source == "api":
        try:
            records = fetch_inspections(api_url, token, hours=48)
            df_all = pd.DataFrame(records) if records else empty_inspections()
            if not df_all.empty:
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
                if "action" not in df_all.columns:
                    df_all["action"] = df_all["recommendation"]
        except Exception as exc:
            load_error = str(exc)
            data_source = "demo"
            df_all = generate_demo_dataset()
            st.caption(f"خواندن بازرسی‌ها ناموفق بود: {exc}")
    else:
        df_all = generate_demo_dataset()

    machine_values = sorted(df_all["machine_id"].dropna().unique()) if not df_all.empty else []
    batch_values = sorted(df_all["batch_id"].dropna().unique()) if not df_all.empty else []
    machine_filter = st.selectbox("دستگاه ساخت", ["همه"] + machine_values)
    batch_filter = st.selectbox("Batch", ["همه"] + batch_values)
    grade_filter = st.selectbox("درجه کیفیت", ["همه", "A", "B", "C"])
    hours_back = st.slider("بازه (ساعت)", 1, 24, 24)

    st.markdown('<div class="side-label">آستانه</div>', unsafe_allow_html=True)
    risk_threshold = st.slider("ریسک هشدار", 0.0, 1.0, 0.75, 0.05)

    if st.button("بازخوانی", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("پتروپالاتوس · v0.1.0")


# =============================================================================
# APPLY FILTERS
# =============================================================================

def _apply_filters(frame: pd.DataFrame, start, end=None) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected = frame[frame["timestamp"] >= start]
    if end is not None:
        selected = selected[selected["timestamp"] < end]
    if machine_filter != "همه":
        selected = selected[selected["machine_id"] == machine_filter]
    if batch_filter != "همه":
        selected = selected[selected["batch_id"] == batch_filter]
    if grade_filter != "همه":
        selected = selected[selected["grade"] == grade_filter]
    return selected


window_start = datetime.now() - timedelta(hours=hours_back)
previous_start = window_start - timedelta(hours=hours_back)
df = _apply_filters(df_all, window_start)
previous_df = _apply_filters(df_all, previous_start, window_start)


# =============================================================================
# HEADER
# =============================================================================

source_note = "بازرسی ذخیره‌شده" if data_source == "api" else "داده نمونه"
st.markdown(
    f"""
    <div class="appbar">
        <div>
            <div class="eyebrow">گروه صنعتی بارز</div>
            <h1>بازرسی بینایی تایر</h1>
        </div>
        <div class="appbar-meta">
            <span class="chip">{source_note}</span>
            <span class="chip">{line_text}</span>
            <span class="chip">{datetime.now().strftime("%H:%M")}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _load_trl_report():
    try:
        response = requests.get(f"{api_url}/lab/report", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    path = Path(__file__).resolve().parent / "data" / "lab" / "trl4-report.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _read_report(url_path: str, file_path: Path):
    try:
        response = requests.get(f"{api_url}{url_path}", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    if file_path.exists():
        return json.loads(file_path.read_text(encoding="utf-8"))
    return None


root = Path(__file__).resolve().parent
trl_report = _load_trl_report()
cycle_report = _read_report("/lab/cycle", root / "data" / "lab" / "cycle-report.json")
real_report = _read_report("/real/report", root / "data" / "real" / "trl5-report.json")
loc_report = _read_report("/real/localization", root / "data" / "real" / "localization-report.json")
class_table = pd.DataFrame()
real_table = pd.DataFrame()
if trl_report:
    class_table = pd.DataFrame([
        {
            "کلاس": CLASS_LABELS_FA.get(name, name),
            "دقت": round(values["precision"], 3),
            "بازیابی": round(values["recall"], 3),
            "F1": round(values["f1"], 3),
            "تعداد": values["support"],
        }
        for name, values in trl_report["metrics"]["per_class"].items()
    ])
if real_report:
    real_table = pd.DataFrame([
        {
            "کلاس": values["label_fa"],
            "دقت": round(values["precision"], 3),
            "بازیابی": round(values["recall"], 3),
            "F1": round(values["f1"], 3),
            "تعداد": values["support"],
        }
        for values in real_report["metrics"]["per_class"].values()
    ])


# =============================================================================
# KPI CARDS
# =============================================================================

def _rate(frame: pd.DataFrame, predicate) -> float:
    if frame.empty:
        return 0.0
    return float(predicate(frame).mean() * 100)


total = len(df)
previous_total = len(previous_df)
defects = int((df["defect_type"] != "good").sum()) if total else 0
defect_rate = (defects / total * 100) if total else 0
pass_rate = 100 - defect_rate if total else 0
avg_risk = float(df["risk_score"].mean()) if total else 0
high_risk = int((df["risk_level"] == "high").sum()) if total else 0
grade_c = int((df["grade"] == "C").sum()) if total else 0

previous_defect_rate = _rate(previous_df, lambda frame: frame["defect_type"] != "good")
previous_pass_rate = 100 - previous_defect_rate if previous_total else 0
previous_risk = float(previous_df["risk_score"].mean()) if previous_total else None
previous_high = int((previous_df["risk_level"] == "high").sum()) if previous_total else None
previous_grade_c = int((previous_df["grade"] == "C").sum()) if previous_total else None

count_text, count_class = _delta(total, previous_total)
pass_text, pass_class = _delta(pass_rate, previous_pass_rate if previous_total else None)
defect_text, defect_class = _delta(defect_rate, previous_defect_rate if previous_total else None, lower_is_better=True)
risk_text, risk_class = _delta(avg_risk, previous_risk, lower_is_better=True)
high_text, high_class = _delta(high_risk, previous_high, lower_is_better=True)
grade_text, grade_class = _delta(grade_c, previous_grade_c, lower_is_better=True)

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">تعداد بازرسی</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-delta {count_class}">{count_text}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">نرخ عبور</div>
            <div class="kpi-value">{pass_rate:.1f}%</div>
            <div class="kpi-delta {pass_class}">{pass_text}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">نرخ عیب</div>
            <div class="kpi-value">{defect_rate:.1f}%</div>
            <div class="kpi-delta {defect_class}">{defect_text}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">میانگین ریسک</div>
            <div class="kpi-value">{avg_risk*100:.0f}%</div>
            <div class="kpi-delta {risk_class}">{risk_text}</div>
        </div>
        <div class="kpi-card" style="border-right-color:#EF4444;">
            <div class="kpi-label">هشدار بحرانی</div>
            <div class="kpi-value" style="color:#EF4444;">{high_risk}</div>
            <div class="kpi-delta {high_class}">{high_text}</div>
        </div>
        <div class="kpi-card" style="border-right-color:#F59E0B;">
            <div class="kpi-label">درجه C (ضایعات)</div>
            <div class="kpi-value" style="color:#F59E0B;">{grade_c}</div>
            <div class="kpi-delta {grade_class}">{grade_text}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

lab_col, xray_col = st.columns(2, gap="small")
with lab_col:
    if trl_report:
        causes = trl_report.get("traceability", {}).get("causes") or []
        cause_text = "، ".join(f"{item['field']}={item['value']}" for item in causes[:2]) or "بدون ردیف"
        st.markdown(
            f"""
            <div class="panel">
                <h3>آزمایشگاه ظاهر</h3>
                <div class="statrow">
                    <div class="stat"><div class="k">بلوغ</div><div class="v">TRL {trl_report['trl']}</div></div>
                    <div class="stat"><div class="k">F1 ترسیم</div><div class="v">{trl_report['metrics']['macro_f1']:.3f}</div></div>
                    <div class="stat"><div class="k">یک وصله</div><div class="v">{trl_report['metrics']['latency_ms']['p95']:.1f} ms</div></div>
                </div>
                <div class="note">این F1 روی ترسیم آزمایشگاهی است و کنار نتیجه مقاله ایکس‌ری قرار نمی‌گیرد. آموزش {trl_report['train_samples']} و آزمون {trl_report['test_samples']}. ردیف ساختگی: <span dir="ltr">{cause_text}</span>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(class_table, hide_index=True, use_container_width=True, height=180)
with xray_col:
    if real_report:
        if not loc_report:
            loc_state = "ثبت نشده"
        elif loc_report["gate_passed"]:
            loc_state = "قبول"
        else:
            loc_state = "رد شد"
        loc_note = "مکان‌یابی هنوز گزارش نشده است."
        if loc_report:
            recall = loc_report["test"]["recall_iou_0_30"]
            precision = loc_report["test"]["precision_iou_0_30"]
            loc_note = (
                f'بازیابی مکان <span dir="ltr">{recall:.3f}</span> و '
                f'دقت <span dir="ltr">{precision:.3f}</span> '
                f'در <span dir="ltr">IoU 0.30</span>.'
            )
        st.markdown(
            f"""
            <div class="panel">
                <h3>ایکس‌ری عمومی</h3>
                <div class="statrow">
                    <div class="stat"><div class="k">F1 آزمون</div><div class="v">{real_report['metrics']['macro_f1']:.3f}</div></div>
                    <div class="stat"><div class="k">دقت</div><div class="v">{real_report['metrics']['accuracy']:.3f}</div></div>
                    <div class="stat"><div class="k">مکان‌یابی</div><div class="v">{loc_state}</div></div>
                </div>
                <div class="note"><span dir="ltr">Zenodo 11381120</span>، چهار کلاس روی کل تصویر. تایر بارز نیست. {loc_note} شیاروگرافی در این مجموعه نیست.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(real_table, hide_index=True, use_container_width=True, height=180)

if cycle_report:
    motion = "سنجیده نشده" if not cycle_report["motion_measured"] else "سنجیده شده"
    claim = cycle_report["published_reference"]["xray_machine_cycle_claim_s"]
    st.markdown(
        f"""
        <div class="strip">
            <span>چرخه نرم‌افزار ۱۶ قطاع: {cycle_report['software_cycle_ms']:.0f} ms</span>
            <span>حرکت مکانیکی: {motion}</span>
            <span>مرجع دستگاه ایکس‌ری: زیر {claim:.0f} ثانیه</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TABS
# =============================================================================

tab_plant, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "کارخانه",
    "عملیات",
    "بازرسی",
    "ردیابی",
    "هشدار",
    "درجه‌بندی",
])


def _paint_status(value: str) -> str:
    colors = {
        "سنجش‌شده": "#047857",
        "آزمایشگاه": "#1D4ED8",
        "فیکسچر": "#B45309",
        "ناقص": "#B45309",
        "رد شد": "#B91C1C",
        "بدون سنجش": "#6B7280",
    }
    return f"color: {colors.get(value, '#374151')}; font-weight: 700"


with tab_plant:
    xray_f1 = f"{real_report['metrics']['macro_f1']:.3f}" if real_report else "ندارد"
    xray_acc = f"{real_report['metrics']['accuracy']:.3f}" if real_report else "ندارد"
    lab_f1 = f"{trl_report['metrics']['macro_f1']:.3f}" if trl_report else "ندارد"
    cycle_ms = f"{cycle_report['software_cycle_ms']:.0f} ms" if cycle_report else "ثبت نشده"
    if loc_report and loc_report.get("gate_passed"):
        loc_status, loc_fact = "سنجش‌شده", "آستانه مکان قبول شد"
    elif loc_report:
        loc_status = "رد شد"
        loc_fact = (
            f"بازیابی {loc_report['test']['recall_iou_0_30']:.3f} و "
            f"دقت {loc_report['test']['precision_iou_0_30']:.3f}"
        )
    else:
        loc_status, loc_fact = "بدون سنجش", "گزارش مکان نیست"
    steps = [
        ("اختلاط", "بدون سنجش", "none", "آمیزه و ویسکوزیته"),
        ("آماده‌سازی", "بدون سنجش", "none", "اکسترود و بید"),
        ("ساخت", "فیکسچر", "part", "دما و فشار محلی"),
        ("پخت", "فیکسچر", "part", "دما و زمان محلی"),
        ("بازرسی", "ناقص", "part", "ظاهر آزمایشگاه و ایکس‌ری عمومی"),
        ("درجه‌بندی", "ناقص", "part", "هندسه خط وصل نیست"),
        ("تحویل", "بدون سنجش", "none", "موجودی و شکایت مشتری"),
    ]
    cards = "".join(
        f'<div class="step"><div class="name"><span class="lamp lamp-{lamp}"></span>{name}</div>'
        f'<div class="state">{state}<br>{detail}</div></div>'
        for name, state, lamp, detail in steps
    )
    st.markdown(f'<div class="flow">{cards}</div>', unsafe_allow_html=True)
    st.caption("سبز یعنی آستانه سنجش قبول شده، کهربایی یعنی فقط آزمایشگاه یا فیکسچر محلی، خاکستری یعنی این دغدغه هنوز داده ندارد. هیچ خانه‌ای عملکرد خط بارز نیست.")

    concerns = pd.DataFrame([
        ["کیفیت", "ظاهر تایر در حال چرخش", "برگشت ظاهری و پذیرش OEM", "آزمایشگاه", f"F1 ترسیم {lab_f1}؛ کلاس برای کل تصویر است"],
        ["کیفیت", "ایکس‌ری ساختاری", "عیب کورد و کمربند پیش از مشتری", "سنجش‌شده", f"F1 {xray_f1} و دقت {xray_acc} روی Zenodo 11381120"],
        ["کیفیت", "مکان عیب", "تعمیر نقطه‌ای به‌جای اسقاط کل تایر", loc_status, loc_fact],
        ["کیفیت", "حباب داخلی", "شیاروگرافی، نه دایره روشن تصویر", "بدون سنجش", "کانال شیاروگرافی تصویر ندارد"],
        ["کیفیت", "یکنواختی نیرو", "RFV و LFV در مشخصات خودروساز", "فیکسچر", "سقف محلی است و مشخصات کارخانه نیست"],
        ["کیفیت", "بالانس", "لرزش و شکایت رانندگی", "فیکسچر", "عدم تعادل استاتیک و کوپل فقط روی فیکسچر"],
        ["کیفیت", "هندسه لیزری", "برآمدگی، فرورفتگی و لنگی", "فیکسچر", "سه مقدار لیزر از سرور محلی خوانده می‌شود"],
        ["کیفیت", "درجه نهایی فروش", "سهم ضایعات، بازکاری و ظرفیت قابل فروش", "ناقص", "بدون هندسه خط، درجه کارخانه ناقص می‌ماند"],
        ["تولید", "اختلاط آمیزه", "ثبات بچ و هزینه مواد", "بدون سنجش", "ویسکوزیته و بچ میکسر کانال ندارد"],
        ["تولید", "ساخت تایر سبز", "اتصال لایه و تکرارپذیری دستگاه", "فیکسچر", "دمای ساخت و فشار از OPC محلی"],
        ["تولید", "پخت", "سوختگی، پخت ناقص و ظرفیت پرس", "فیکسچر", "دما و زمان پخت از OPC محلی"],
        ["تولید", "هشدار حین فرآیند", "توقف پیش از یک دسته ضایعات", "آزمایشگاه", "ریسک روی بازرسی ذخیره‌شده همین سرویس"],
        ["تولید", "ردیابی منشأ", "قرنطینه مواد، قالب و دستگاه", "آزمایشگاه", "ضریب افزایش از همبستگی ساختگی است"],
        ["تولید", "چرخه و ظرفیت بازرسی", "مقایسه با دستگاهی که چرخه‌اش زیر ۲۲ ثانیه اعلام شده", "ناقص", f"نرم‌افزار {cycle_ms}؛ حرکت مکانیکی سنجیده نشده"],
        ["تولید", "اتصال خط و رد خودکار", "بستن حلقه کیفیت با PLC", "فیکسچر", f"وضعیت خط: {line_text}. رد خودکار پیش‌فرض خاموش است"],
        ["رقابت", "آماده‌به‌کاری و OEE", "تحویل به خودروساز", "بدون سنجش", "توقف پرس و دستگاه ساخت ثبت نشده"],
        ["رقابت", "نگهداری قالب", "نقش تایر و توقف خط پخت", "بدون سنجش", "عمر قالب و تعویض ثبت نشده"],
        ["رقابت", "انرژی پخت", "بهای تمام‌شده هر تایر", "بدون سنجش", "بخار و برق پرس ثبت نشده"],
        ["رقابت", "موجودی سبز و محصول", "خواب سرمایه و تعهد تحویل", "بدون سنجش", "انبار تایر سبز و محصول ثبت نشده"],
        ["رقابت", "شکایت و برگشت مشتری", "جایگاه نزد OEM و صادرات", "بدون سنجش", "PPM شکایت ثبت نشده"],
        ["رقابت", "ممیزی تصمیم", "پاسخگویی کیفی در ممیزی مشتری", "آزمایشگاه", "لاگ همین سرویس است، نه ممیزی کارخانه"],
    ], columns=["حوزه", "دغدغه", "اثر رقابتی", "وضعیت", "شاهد امروز"])
    styled = concerns.style.map(_paint_status, subset=["وضعیت"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=640)
    left, right = st.columns(2, gap="small")
    with left:
        counts = concerns["وضعیت"].value_counts()
        fig_status = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=["#047857" if name == "سنجش‌شده" else "#B91C1C" if name == "رد شد" else "#B45309" if name in {"فیکسچر", "ناقص"} else "#1D4ED8" if name == "آزمایشگاه" else "#9CA3AF" for name in counts.index],
            text=counts.values, textposition="outside",
        ))
        fig_status.update_layout(height=280, margin=dict(l=8, r=8, t=28, b=8), title="تعداد دغدغه‌ها به تفکیک وضعیت")
        st.plotly_chart(_frame(fig_status), use_container_width=True)
    with right:
        st.markdown(
            """
            <div class="panel">
                <h3>مرجع رقابت، بدون کنار هم گذاشتن دقت‌ها</h3>
                <div class="note">چرخه اعلام‌شده دستگاه ایکس‌ری صنعتی زیر ۲۲ ثانیه است. اسکن لیزری ظاهر در نقطه کار ۶۰ دور بر دقیقه، یک ثانیه برای هر دور است. مقاله ایکس‌ری با حدود ۱۲۳ هزار تصویر و ۱۵ نوع عیب، آزمون دیگری است و F1 این سامانه کنار آن قرار نمی‌گیرد.</div>
                <div class="note">اختلاط، انرژی، موجودی، نگهداری قالب و شکایت مشتری دغدغه واقعی کارخانه‌اند و تا وقتی برچسب فرآیندشان به سرویس نرسد در جدول بدون سنجش می‌مانند.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    if df.empty:
        st.info("در این بازه بازرسی ثبت نشده است.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="section-title">روند عیوب در بازه زمانی</div>', unsafe_allow_html=True)
            trend = df.copy()
            trend["hour"] = trend["timestamp"].dt.floor("h")
            pivot = trend.groupby(["hour", "defect_type"]).size().unstack(fill_value=0).reset_index()

            fig_trend = go.Figure()
            for cls in CLASSES:
                if cls in pivot.columns:
                    fig_trend.add_trace(go.Scatter(
                        x=pivot["hour"], y=pivot[cls],
                        name=CLASS_LABELS_FA[cls], mode="lines+markers",
                        line=dict(width=2.5, color=PALETTE[cls]), marker=dict(size=6),
                    ))
            fig_trend.update_layout(
                height=340, margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="", gridcolor="#F1F5F9"),
                yaxis=dict(title="تعداد", gridcolor="#F1F5F9"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(_frame(fig_trend), use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">توزیع عیوب</div>', unsafe_allow_html=True)
            counts = df["defect_type"].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=[CLASS_LABELS_FA[k] for k in counts.index],
                values=counts.values, hole=0.62,
                marker=dict(colors=[PALETTE[k] for k in counts.index]),
                textinfo="percent+label", textposition="outside",
            )])
            fig_pie.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                                  showlegend=False, paper_bgcolor="white")
            st.plotly_chart(_frame(fig_pie), use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-title">عملکرد دستگاه‌های ساخت</div>', unsafe_allow_html=True)
            m_stats = df.groupby("machine_id").agg(
                total=("tire_id", "count"),
                defects=("defect_type", lambda x: (x != "good").sum()),
            ).reset_index()
            m_stats["defect_rate"] = m_stats["defects"] / m_stats["total"] * 100
            m_stats = m_stats.sort_values("defect_rate")

            fig_m = px.bar(
                m_stats, x="defect_rate", y="machine_id", orientation="h",
                color="defect_rate",
                color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                labels={"defect_rate": "نرخ عیب (%)", "machine_id": "دستگاه"},
            )
            fig_m.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                plot_bgcolor="white", paper_bgcolor="white",
                                coloraxis_showscale=False)
            st.plotly_chart(_frame(fig_m), use_container_width=True)

        with col4:
            st.markdown('<div class="section-title">توزیع Batch</div>', unsafe_allow_html=True)
            b_counts = df.groupby(["batch_id", "defect_type"]).size().unstack(fill_value=0)
            fig_b = go.Figure()
            for cls in CLASSES:
                if cls in b_counts.columns:
                    fig_b.add_trace(go.Bar(name=CLASS_LABELS_FA[cls],
                                           x=b_counts.index, y=b_counts[cls],
                                           marker_color=PALETTE[cls]))
            fig_b.update_layout(
                barmode="stack", height=300, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Batch"), yaxis=dict(title="تعداد"),
            )
            st.plotly_chart(_frame(fig_b), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: LIVE INSPECTION
# -----------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">تست سریع تک تصویر</div>', unsafe_allow_html=True)
    up = st.file_uploader("تصویر تایر را آپلود کنید (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if up is not None:
        img = Image.open(up).convert("L").resize((128, 128))
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(img, caption="تصویر ورودی", use_container_width=True)
        with c2:
            if data_source == "api" and token:
                try:
                    meta = {
                        "tire_id": f"LIVE-{datetime.now().strftime('%H%M%S')}",
                        "batch_id": batch_filter if batch_filter != "همه" else "B001",
                        "machine_id": machine_filter if machine_filter != "همه" else "M01",
                        "mold_id": "D01", "material_lot": "L01",
                        "build_temp": 155, "build_pressure": 11,
                        "cure_temp": 165, "cure_time": 15,
                    }
                    with st.spinner("در حال پردازش در سرویس..."):
                        result = call_predict(api_url, img, meta, token)
                    appearance = (result.get("channels") or {}).get("appearance") or {}
                    label = appearance.get("label_fa") or CLASS_LABELS_FA.get(result["defect_type"], result["defect_type"])
                    st.success(f"**نوع عیب ظاهری:** {label}")
                    st.caption(appearance.get("note") or "کلاس برای کل تصویر است.")
                    shear = (result.get("channels") or {}).get("shearography") or {}
                    st.caption(f"شیاروگرافی: {shear.get('reason', 'در دسترس نیست')}")
                    c2a, c2b, c2c = st.columns(3)
                    c2a.metric("احتمال عیب", f"{result['defect_probability']*100:.1f}%")
                    c2b.metric("درجه ظاهر", GRADE_FA.get(result["grade"], result["grade"]))
                    station = result.get("station") or {}
                    station_text = "ناقص" if station.get("status") != "complete" else GRADE_FA.get(station.get("grade"), station.get("grade"))
                    c2c.metric("درجه کارخانه", station_text)
                    timing = result.get("timing") or {}
                    if timing:
                        st.caption(
                            f"چرخه نرم‌افزار {timing.get('software_cycle_ms', 0):.0f} ms برای {timing.get('sectors', 16)} قطاع. "
                            "حرکت مکانیکی تایر در این سنجش نیست."
                        )
                    st.info(f"**توصیه:** {result['recommendation']}")
                    if station.get("reason"):
                        st.caption(station["reason"])
                    if result.get("suspected_causes"):
                        st.write("**همبستگی آزمایشگاهی، نه علت خط:**")
                        for c in result["suspected_causes"]:
                            st.write(f"- {c['field']}: {c['value']} (Lift: {c['lift']:.2f})")
                except Exception as e:
                    st.error(f"خطا در اتصال به API: {e}")
            else:
                st.info("سرویس API در دسترس نیست. برای تست زنده، `python src.py serve` را اجرا کنید.")

    st.markdown("---")
    st.markdown('<div class="section-title">مکان عیب روی ایکس‌ری</div>', unsafe_allow_html=True)
    xray_up = st.file_uploader("تصویر ایکس‌ری (JPG/PNG)", type=["png", "jpg", "jpeg"], key="xray")
    if xray_up is not None and data_source == "api" and token:
        xray_image = Image.open(xray_up).convert("RGB")
        try:
            xray_result = call_predict_xray(api_url, xray_image, token)
            regions = xray_result.get("regions") or []
            shown = _draw_regions(xray_image, regions) if regions else xray_image
            xc1, xc2 = st.columns([1, 1])
            xc1.image(shown, caption="جعبه روی مختصات نرمال‌شده تصویر", use_container_width=True)
            xc2.write(f"کلاس کل تصویر: {xray_result.get('label_fa')} ({xray_result.get('confidence', 0):.2f})")
            if loc_report and not loc_report.get("gate_passed"):
                xc2.caption("جعبه مختصات تولید می‌شود، ولی آستانه مکان‌یابی روی آزمون رسمی رد شده است.")
            if regions:
                xc2.dataframe(pd.DataFrame(regions), hide_index=True, use_container_width=True)
            else:
                xc2.caption("جعبه بالای آستانه اعتبارسنجی پیدا نشد. کلاس کل تصویر همچنان یک برچسب بدون مختصات است.")
        except Exception as exc:
            st.error(f"سنجش ایکس‌ری ناموفق بود: {exc}")
    elif xray_up is not None:
        st.info("برای مکان‌یابی، سرویس API باید در دسترس باشد.")

    st.markdown("---")
    st.markdown('<div class="section-title">آخرین ۱۲ تایر بازرسی‌شده</div>', unsafe_allow_html=True)

    recent = df.head(12).reset_index(drop=True)
    for start in range(0, 12, 4):
        cols = st.columns(4)
        for j in range(4):
            idx = start + j
            if idx >= len(recent):
                break
            row = recent.iloc[idx]
            with cols[j]:
                image_path = row.get("image_path") if hasattr(row, "get") else None
                if data_source == "api" and isinstance(image_path, str) and Path(image_path).exists():
                    st.image(image_path, use_container_width=True)
                elif data_source == "api":
                    st.caption("تصویر ذخیره‌شده در دسترس نیست")
                else:
                    st.image(_make_demo_image(row["defect_type"]), caption="تصویر نمونه", use_container_width=True)
                badge = "badge-low" if row["defect_type"] == "good" else "badge-high"
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:-10px;">
                        <div style="font-weight:700; font-size:.85rem;">{row['tire_id']}</div>
                        <span class="badge {badge}">{CLASS_LABELS_FA[row['defect_type']]}</span>
                        <div style="font-size:.75rem; color:#6B7280; margin-top:6px;">
                            {row['machine_id']} · {row['batch_id']} · درجه {GRADE_FA[row['grade']]}
                        </div>
                        <div style="font-size:.75rem; font-weight:700; color:#C8102E;">
                            احتمال {row['defect_probability']*100:.0f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# TAB 3: TRACEABILITY
# -----------------------------------------------------------------------------
with tab3:
    if df.empty:
        st.info("در این بازه بازرسی ثبت نشده است.")
    else:
        st.markdown('<div class="section-title">تحلیل ردیابی منشأ عیب</div>', unsafe_allow_html=True)

        selected_defect = st.selectbox(
            "نوع عیب برای تحلیل",
            [c for c in CLASSES if c != "good"],
            format_func=lambda x: CLASS_LABELS_FA[x],
        )

        baseline = max((df["defect_type"] == selected_defect).mean(), 1e-6)
        trace_rows = []
        for field in ["machine_id", "mold_id", "material_lot", "batch_id"]:
            for val, group in df.groupby(field):
                rate = (group["defect_type"] == selected_defect).mean()
                trace_rows.append({
                    "field": field, "value": str(val),
                    "defect_rate": rate * 100, "lift": rate / baseline,
                    "count": len(group),
                })
        trace_df = pd.DataFrame(trace_rows).sort_values("lift", ascending=False)

        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown(f"**منشأهای محتمل برای «{CLASS_LABELS_FA[selected_defect]}»**")
            top = trace_df.head(8).copy()
            top["label"] = top["field"].map({
                "machine_id": "دستگاه", "mold_id": "قالب",
                "material_lot": "دسته مواد", "batch_id": "Batch",
            }) + ": " + top["value"]

            fig_lift = px.bar(
                top, x="lift", y="label", orientation="h", color="lift",
                color_continuous_scale=["#FEF3C7", "#F59E0B", "#EF4444"],
                labels={"lift": "ضریب افزایش ریسک (Lift)", "label": ""},
            )
            fig_lift.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                                   plot_bgcolor="white", paper_bgcolor="white",
                                   coloraxis_showscale=False)
            st.plotly_chart(_frame(fig_lift), use_container_width=True)

        with colB:
            st.markdown("**جدول جزئیات**")
            show = trace_df.head(10).copy()
            show["field"] = show["field"].map({
                "machine_id": "دستگاه", "mold_id": "قالب",
                "material_lot": "دسته مواد", "batch_id": "Batch",
            })
            show["defect_rate"] = show["defect_rate"].round(1)
            show["lift"] = show["lift"].round(2)
            show.columns = ["فیلد", "مقدار", "نرخ عیب (%)", "Lift", "تعداد"]
            st.dataframe(show, use_container_width=True, height=380, hide_index=True)

        st.info(
            "این Lift از بازرسی‌های ذخیره‌شده همین سامانه است. داده فعلی آزمایشگاهی و هم‌بسته ساختگی است "
            "و علت ریشه‌ای خط تولید بارز را ثابت نمی‌کند."
        )


# -----------------------------------------------------------------------------
# TAB 4: PREDICTION & ALERTS
# -----------------------------------------------------------------------------
with tab4:
    if df.empty:
        st.info("در این بازه بازرسی ثبت نشده است.")
    else:
        st.markdown('<div class="section-title">هشدارهای پیش‌نگر</div>', unsafe_allow_html=True)

        alerts = df[df["risk_score"] >= risk_threshold].sort_values("risk_score", ascending=False).head(20)

        if alerts.empty:
            st.success(f"✅ هیچ هشدار بحرانی در آستانه {risk_threshold*100:.0f}% وجود ندارد.")
        else:
            st.warning(f"{len(alerts)} هشدار بالای آستانه {risk_threshold*100:.0f}% یافت شد.")

            for _, row in alerts.head(8).iterrows():
                bg = "#FEE2E2" if row["risk_level"] == "high" else "#FEF3C7"
                border = "#EF4444" if row["risk_level"] == "high" else "#F59E0B"
                st.markdown(
                    f"""
                    <div style="background:{bg}; border-right:4px solid {border};
                                padding:12px 16px; border-radius:10px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                            <div><b>{row['tire_id']}</b> · {CLASS_LABELS_FA[row['defect_type']]} ·
                            دستگاه {row['machine_id']} · {row['batch_id']}</div>
                            <div style="font-weight:800; color:{border};">
                                ریسک {row['risk_score']*100:.0f}%
                            </div>
                        </div>
                        <div style="font-size:.8rem; color:#4B5563; margin-top:6px;">
                            دما ساخت {row['build_temp']:.1f}°C ·
                            فشار {row['build_pressure']:.1f} bar ·
                            پخت {row['cure_temp']:.1f}°C / {row['cure_time']:.1f} دقیقه
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown('<div class="section-title">تحلیل پارامترهای فرآیند</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_t = px.scatter(
                df, x="build_temp", y="cure_temp", color="defect_type",
                color_discrete_map=PALETTE,
                labels={"build_temp": "دمای ساخت (°C)", "cure_temp": "دمای پخت (°C)", "defect_type": "نوع"},
                opacity=0.7,
            )
            fig_t.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=10, r=10, t=20, b=10),
                                title="پراکندگی دما ساخت × دما پخت")
            st.plotly_chart(_frame(fig_t), use_container_width=True)

        with col2:
            fig_p = px.box(
                df, x="defect_type", y="build_pressure", color="defect_type",
                color_discrete_map=PALETTE,
                labels={"build_pressure": "فشار ساخت (bar)", "defect_type": "نوع"},
            )
            fig_p.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=10, r=10, t=20, b=10),
                                title="فشار ساخت به تفکیک نوع عیب", showlegend=False)
            st.plotly_chart(_frame(fig_p), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 5: GRADING & DECISION
# -----------------------------------------------------------------------------
with tab5:
    if df.empty:
        st.info("در این بازه بازرسی ثبت نشده است.")
    else:
        st.markdown('<div class="section-title">توزیع درجه‌بندی کیفیت</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            grade_counts = df["grade"].value_counts().sort_index()
            grade_colors = {"A": "#10B981", "B": "#F59E0B", "C": "#EF4444"}
            fig_g = go.Figure(data=[go.Bar(
                x=[f"درجه {GRADE_FA[g]}" for g in grade_counts.index],
                y=grade_counts.values,
                marker_color=[grade_colors[g] for g in grade_counts.index],
                text=grade_counts.values, textposition="outside",
            )])
            fig_g.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(title="تعداد"))
            st.plotly_chart(_frame(fig_g), use_container_width=True)

        with col2:
            action_counts = df["action"].value_counts()
            fig_a = go.Figure(data=[go.Pie(
                labels=action_counts.index, values=action_counts.values, hole=0.55,
                marker=dict(colors=["#10B981", "#F59E0B", "#3B82F6", "#EF4444"][:len(action_counts)]),
                textinfo="percent+label",
            )])
            fig_a.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
                                title="توزیع توصیه اقدام")
            st.plotly_chart(_frame(fig_a), use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">جدول تصمیم‌یار</div>', unsafe_allow_html=True)

        show_df = df[[
            "tire_id", "timestamp", "defect_type", "defect_probability",
            "severity", "risk_score", "risk_level", "grade", "action",
        ]].copy()
        show_df["defect_type"] = show_df["defect_type"].map(CLASS_LABELS_FA)
        show_df["severity"] = show_df["severity"].map(SEVERITY_FA)
        show_df["risk_level"] = show_df["risk_level"].map(RISK_FA)
        show_df["grade"] = show_df["grade"].map(GRADE_FA)
        show_df["defect_probability"] = (show_df["defect_probability"] * 100).round(1).astype(str) + "%"
        show_df["risk_score"] = (show_df["risk_score"] * 100).round(0).astype(int).astype(str) + "%"
        show_df["timestamp"] = pd.to_datetime(show_df["timestamp"]).dt.strftime("%H:%M")
        show_df.columns = [
            "شناسه تایر", "زمان", "نوع عیب", "احتمال عیب",
            "شدت", "امتیاز ریسک", "سطح ریسک", "درجه", "توصیه اقدام",
        ]
        st.dataframe(show_df.head(50), use_container_width=True, height=420, hide_index=True)

        csv = show_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "دانلود گزارش",
            data=csv,
            file_name=f"tvdi_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
