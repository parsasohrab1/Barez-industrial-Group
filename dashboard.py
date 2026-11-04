"""
TVDI Dashboard — داشبورد حرفه‌ای سامانه بازرسی تایر
گروه صنعتی بارز

اجرا:
    pip install streamlit plotly pandas numpy requests pillow
    streamlit run dashboard.py --server.port 8501
"""

import base64
import io
import random
from datetime import datetime, timedelta

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

API_URL_DEFAULT = "http://localhost:8000"
CLASSES = ["good", "bubble", "tread_uneven", "layer_separation"]
CLASS_LABELS_FA = {
    "good": "سالم",
    "bubble": "حباب",
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
    page_icon="🏭",
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

    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A1A 0%, #2D2D2D 100%);
        border-left: 1px solid #374151;
    }
    [data-testid="stSidebar"] * { color: #E5E7EB !important; }
    [data-testid="stSidebar"] label { color: #9CA3AF !important; font-size: .85rem !important; }

    .brand-header {
        background: linear-gradient(135deg, #1A1A1A 0%, #C8102E 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(200,16,46,.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .brand-header h1 { margin: 0; font-size: 1.75rem; font-weight: 800; color: white !important; }
    .brand-header .subtitle { color: rgba(255,255,255,.85); font-size: .9rem; margin-top: 4px; }
    .brand-header .brand-badge {
        background: rgba(255,255,255,.15);
        border: 1px solid rgba(255,255,255,.3);
        padding: 6px 16px; border-radius: 999px;
        font-size: .8rem; color: white; font-weight: 600;
    }

    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px; margin-bottom: 20px; }
    .kpi-card {
        background: white; padding: 20px; border-radius: 14px;
        border-right: 4px solid #C8102E;
        box-shadow: 0 2px 8px rgba(0,0,0,.06);
        transition: transform .2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,.1); }
    .kpi-label { font-size: .85rem; color: #6B7280; margin-bottom: 8px; font-weight: 500; }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: #1A1A1A; line-height: 1.1; }
    .kpi-delta { font-size: .78rem; margin-top: 6px; font-weight: 600; }
    .kpi-delta.up { color: #10B981; }
    .kpi-delta.down { color: #EF4444; }

    .badge { display: inline-block; padding: 4px 10px; border-radius: 999px;
             font-size: .75rem; font-weight: 600; }
    .badge-high { background: #FEE2E2; color: #991B1B; }
    .badge-medium { background: #FEF3C7; color: #92400E; }
    .badge-low { background: #D1FAE5; color: #065F46; }

    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #1A1A1A;
        margin: 20px 0 12px 0; padding-right: 12px;
        border-right: 4px solid #C8102E;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: white; padding: 8px;
        border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 10px 20px;
        font-weight: 600; font-size: .9rem; color: #6B7280;
    }
    .stTabs [aria-selected="true"] { background: #C8102E !important; color: white !important; }

    .stButton>button {
        background: #C8102E; color: white; border: none; border-radius: 8px;
        padding: 8px 20px; font-weight: 600; font-family: 'Vazirmatn', sans-serif;
        transition: all .2s;
    }
    .stButton>button:hover { background: #A00D25; color: white; transform: translateY(-1px); }

    [data-testid="stDataFrame"] {
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }

    .status-pill { display: inline-flex; align-items: center; gap: 6px;
                   padding: 6px 14px; border-radius: 999px;
                   font-size: .8rem; font-weight: 600; }
    .status-online { background: #D1FAE5; color: #065F46; }
    .status-offline { background: #FEE2E2; color: #991B1B; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; }
    .status-dot-online { background: #10B981; box-shadow: 0 0 8px #10B981; }
    .status-dot-offline { background: #EF4444; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# API HELPERS
# =============================================================================

def check_api_health(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def call_predict(url: str, image: Image.Image, metadata: dict) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "metadata": metadata,
    }
    r = requests.post(f"{url}/predict-json", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


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
        ts = now - timedelta(minutes=int(rng.integers(0, 60 * 24)))
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
        })

    return pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 20px 0;">
            <div style="font-size: 2.2rem;">🏭</div>
            <div style="font-weight: 800; font-size: 1.1rem; color: white;">TVDI</div>
            <div style="font-size: .75rem; color: #9CA3AF;">Tire Vision Defect Inspector</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ اتصال")
    api_url = st.text_input("آدرس سرویس API", value=API_URL_DEFAULT)
    api_online = check_api_health(api_url)

    if api_online:
        st.markdown(
            '<span class="status-pill status-online"><span class="status-dot status-dot-online"></span>آنلاین — API</span>',
            unsafe_allow_html=True,
        )
        data_source = "api"
    else:
        st.markdown(
            '<span class="status-pill status-offline"><span class="status-dot status-dot-offline"></span>آفلاین — داده نمایشی</span>',
            unsafe_allow_html=True,
        )
        data_source = "demo"

    st.markdown("---")
    st.markdown("### 🔍 فیلترها")

    df_all = generate_demo_dataset()

    machine_filter = st.selectbox("دستگاه ساخت", ["همه"] + sorted(df_all["machine_id"].unique()))
    batch_filter = st.selectbox("Batch", ["همه"] + sorted(df_all["batch_id"].unique()))
    grade_filter = st.selectbox("درجه کیفیت", ["همه", "A", "B", "C"])
    hours_back = st.slider("بازه (ساعت)", 1, 24, 24)

    st.markdown("---")
    st.markdown("### 🚨 آستانه هشدار")
    risk_threshold = st.slider("آستانه ریسک", 0.0, 1.0, 0.75, 0.05)

    st.markdown("---")
    if st.button("🔄 بازخوانی داده", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        '<div style="position:absolute; bottom:20px; left:20px; right:20px; '
        'color:#6B7280; font-size:.7rem; text-align:center;">'
        'گروه صنعتی بارز · پتروپالاتوس<br>v0.1.0</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# APPLY FILTERS
# =============================================================================

df = df_all.copy()
df = df[df["timestamp"] >= datetime.now() - timedelta(hours=hours_back)]
if machine_filter != "همه":
    df = df[df["machine_id"] == machine_filter]
if batch_filter != "همه":
    df = df[df["batch_id"] == batch_filter]
if grade_filter != "همه":
    df = df[df["grade"] == grade_filter]


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f"""
    <div class="brand-header">
        <div>
            <h1>سامانه هوشمند بازرسی تایر</h1>
            <div class="subtitle">پایش زنده کیفیت، ردیابی منشأ عیب و تصمیم‌یار تولید · گروه صنعتی بارز</div>
        </div>
        <div class="brand-badge">🕐 آخرین به‌روزرسانی: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# KPI CARDS
# =============================================================================

total = len(df)
defects = int((df["defect_type"] != "good").sum())
defect_rate = (defects / total * 100) if total else 0
pass_rate = 100 - defect_rate
avg_risk = df["risk_score"].mean() if total else 0
high_risk = int((df["risk_level"] == "high").sum())
grade_c = int((df["grade"] == "C").sum())

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">تعداد بازرسی</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-delta up">▲ ۱۲٪ نسبت به دیروز</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">نرخ عبور</div>
            <div class="kpi-value">{pass_rate:.1f}%</div>
            <div class="kpi-delta up">▲ ۲.۳٪</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">نرخ عیب</div>
            <div class="kpi-value">{defect_rate:.1f}%</div>
            <div class="kpi-delta down">▼ ۱.۸٪</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">میانگین ریسک</div>
            <div class="kpi-value">{avg_risk*100:.0f}%</div>
            <div class="kpi-delta down">▼ ۳٪</div>
        </div>
        <div class="kpi-card" style="border-right-color:#EF4444;">
            <div class="kpi-label">هشدار بحرانی</div>
            <div class="kpi-value" style="color:#EF4444;">{high_risk}</div>
            <div class="kpi-delta down">▲ ۲ مورد</div>
        </div>
        <div class="kpi-card" style="border-right-color:#F59E0B;">
            <div class="kpi-label">درجه C (ضایعات)</div>
            <div class="kpi-value" style="color:#F59E0B;">{grade_c}</div>
            <div class="kpi-delta up">▼ ۵ مورد</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 نمای کلی",
    "🔍 بازرسی زنده",
    "🎯 ردیابی منشأ",
    "🚨 پیش‌بینی و هشدار",
    "📋 درجه‌بندی و تصمیم",
])


# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">روند عیوب در بازه زمانی</div>', unsafe_allow_html=True)
        trend = df.copy()
        trend["hour"] = trend["timestamp"].dt.floor("H")
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
        st.plotly_chart(fig_trend, use_container_width=True)

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
        st.plotly_chart(fig_pie, use_container_width=True)

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
        st.plotly_chart(fig_m, use_container_width=True)

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
        st.plotly_chart(fig_b, use_container_width=True)


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
            if data_source == "api":
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
                        result = call_predict(api_url, img, meta)
                    st.success(f"**نوع عیب:** {CLASS_LABELS_FA.get(result['defect_type'], result['defect_type'])}")
                    c2a, c2b = st.columns(2)
                    c2a.metric("احتمال عیب", f"{result['defect_probability']*100:.1f}%")
                    c2b.metric("درجه", GRADE_FA.get(result["grade"], result["grade"]))
                    st.info(f"**توصیه:** {result['recommendation']}")
                    if result.get("suspected_causes"):
                        st.write("**منشأ محتمل:**")
                        for c in result["suspected_causes"]:
                            st.write(f"- {c['field']}: {c['value']} (Lift: {c['lift']:.2f})")
                except Exception as e:
                    st.error(f"خطا در اتصال به API: {e}")
            else:
                st.info("سرویس API در دسترس نیست. برای تست زنده، `python src.py serve` را اجرا کنید.")

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
                st.image(_make_demo_image(row["defect_type"]), use_container_width=True)
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
        st.plotly_chart(fig_lift, use_container_width=True)

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
        f"💡 **تفسیر:** Lift بالاتر از ۱ یعنی این مقدار بیشتر از میانگین خط در بروز عیب "
        f"«{CLASS_LABELS_FA[selected_defect]}» نقش دارد. اولویت اقدام اصلاحی روی موارد بالای جدول."
    )


# -----------------------------------------------------------------------------
# TAB 4: PREDICTION & ALERTS
# -----------------------------------------------------------------------------
with tab4:
    st.markdown('<div class="section-title">هشدارهای پیش‌نگر</div>', unsafe_allow_html=True)

    alerts = df[df["risk_score"] >= risk_threshold].sort_values("risk_score", ascending=False).head(20)

    if alerts.empty:
        st.success(f"✅ هیچ هشدار بحرانی در آستانه {risk_threshold*100:.0f}% وجود ندارد.")
    else:
        st.warning(f"🚨 {len(alerts)} هشدار بالای آستانه {risk_threshold*100:.0f}% یافت شد.")

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
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        fig_p = px.box(
            df, x="defect_type", y="build_pressure", color="defect_type",
            color_discrete_map=PALETTE,
            labels={"build_pressure": "فشار ساخت (bar)", "defect_type": "نوع"},
        )
        fig_p.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                            margin=dict(l=10, r=10, t=20, b=10),
                            title="فشار ساخت به تفکیک نوع عیب", showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 5: GRADING & DECISION
# -----------------------------------------------------------------------------
with tab5:
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
        st.plotly_chart(fig_g, use_container_width=True)

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
        st.plotly_chart(fig_a, use_container_width=True)

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
        "📥 دانلود گزارش CSV",
        data=csv,
        file_name=f"tvdi_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
