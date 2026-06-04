import streamlit as st
import requests
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Flood Classifier", page_icon="🌊", layout="wide")

API_URL = "http://localhost:8000/predict"

MODEL_MAP = {
    "ResNet-50":    {"id": "ResNet_50",    "tag": "Recommended", "desc": "Optimal balance of accuracy & speed"},
    "Custom CNN":   {"id": "Custom_CNN",   "tag": "Fastest",     "desc": "Minimal latency inference"},
    "Inception V3": {"id": "Inception_V3", "tag": "Robust",      "desc": "Deep feature extraction"},
    "ViT B-16":     {"id": "ViT_B-16",    "tag": "Slowest",    "desc": "Highest representational capacity"},
}

# --- GLOBAL CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── App Background & Typography ─────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0b1120 !important;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(14, 165, 233, 0.08), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.05), transparent 25%);
    font-family: 'Outfit', sans-serif;
    color: #f8fafc;
}
[data-testid="stHeader"]  { background: transparent !important; }
.block-container          { padding: 3.5rem 4rem !important; max-width: 1200px !important; }

/* ── Hide Chrome ────────────────────────────────── */
[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }

/* ── Typography Overrides ───────────────────────── */
h1, h2, h3, p { font-family: 'Outfit', sans-serif !important; }

/* ── Segmented Control (Radio) ──────────────────── */
[data-testid="stRadio"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 6px;
    display: inline-flex;
    width: fit-content;
}
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 4px !important;
}
[data-testid="stRadio"] label {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
[data-testid="stRadio"] label:hover {
    background: #1e293b !important;
    color: #38bdf8 !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(14, 165, 233, 0.15) !important;
    color: #38bdf8 !important;
}
[data-testid="stRadio"] input[type="radio"] { display: none !important; }

/* ── File Uploader ──────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0f172a !important;
    border: 1.5px dashed #334155 !important;
    border-radius: 16px !important;
    padding: 2.5rem !important;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
}
[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
}

/* ── Primary Button ─────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #0ea5e9 !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
    padding: 1.2rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5) !important;
    background: #0284c7 !important;
}

/* ── Metrics Cards ──────────────────────────────── */
[data-testid="metric-container"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}

/* ── Progress Bar ───────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: #38bdf8 !important;
}

/* ── Divider ────────────────────────────────────── */
[data-testid="stDivider"] {
    border-color: #1e293b !important;
    margin: 2.5rem 0 !important;
}

/* ── Alert Boxes ────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
}

/* ── Spinner Text ───────────────────────────────── */
[data-testid="stSpinner"] p {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem; text-align: center;">
    <div style="font-weight: 700; font-size: 2.5rem; color: #f8fafc; line-height: 1.2;">
        🌊 Flood Classification Interface
    </div>
    <div style="color: #94a3b8; font-size: 1.15rem; margin-top: 0.5rem; font-weight: 300;">
        Upload an image to determine the presence of flooding using deep learning.
    </div>
</div>
""", unsafe_allow_html=True)

# ── MODEL SELECTOR ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.8rem; text-align: center;">
    Select AI Architecture
</div>
""", unsafe_allow_html=True)

# Centering the radio buttons using columns
_, center_col, _ = st.columns([1, 4, 1])
with center_col:
    selected_display = st.radio(
        label="Model",
        options=list(MODEL_MAP.keys()),
        horizontal=True,
        label_visibility="collapsed",
        index=0,
    )

model_info = MODEL_MAP[selected_display]
st.markdown(f"""
<div style="color: #94a3b8; font-size: 0.95rem; text-align: center; margin-top: 1rem;">
    {model_info['desc']} · 
    <span style="color: #38bdf8; font-weight: 500; background: rgba(56, 189, 248, 0.1); padding: 3px 10px; border-radius: 6px;">
        {model_info['tag']}
    </span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── TWO-COLUMN LAYOUT ──────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 1rem;">
        1. Upload Image
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width='stretch')

with col2:
    st.markdown("""
    <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 1rem;">
        2. Results
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.button("Run Classification", type="primary", width='stretch'):
            with st.spinner(f"Classifying with {selected_display}..."):
                clean_model_name = model_info["id"]
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data  = {"model_name": clean_model_name}

                try:
                    response = requests.post(API_URL, files=files, data=data)
                    response.raise_for_status()
                    result = response.json()

                    pred     = result["prediction"]
                    conf     = result["confidence"]
                    inf_time = result["inference_time_ms"]

                    st.divider()

                    if pred == "Flood":
                        st.error(f"### 🚨 Result: {pred}")
                    else:
                        st.success(f"### ✅ Result: {pred}")

                    st.write("") # Spacer
                    
                    met1, met2 = st.columns(2)
                    met1.metric("Confidence", f"{conf:.2%}")
                    met2.metric("Inference Time", f"{inf_time:.1f} ms")
                    
                    st.write("") # Spacer
                    st.progress(conf)

                except Exception as e:
                    st.error(f"Failed to connect to backend API: {e}")

    else:
        st.markdown("""
        <div style="
            background: #0f172a;
            border: 1.5px dashed #334155;
            border-radius: 16px;
            padding: 4.5rem 2rem;
            text-align: center;
        ">
            <div style="color: #64748b; font-size: 1.05rem; font-weight: 500;">
                Waiting for image upload...
            </div>
        </div>
        """, unsafe_allow_html=True)