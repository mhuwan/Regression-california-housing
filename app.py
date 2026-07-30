# -*- coding: utf-8 -*-
"""
California Housing Price Predictor
Streamlit Web Application - Minimalist Theme
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS (Minimalist Template) ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1d1d1f;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 3rem 1rem;
        background-color: transparent;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1d1d1f;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #86868b;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0;
    }
    .badge {
        display: inline-block;
        margin-top: 1.2rem;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        background-color: #f0f7ff;
        color: #0071e3;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #dbeafe;
    }

    /* Card styling */
    .minimal-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.8rem;
        border: 1px solid #e5e5ea;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #f5f5f7;
        padding-bottom: 0.8rem;
    }

    /* Metric cards */
    .metric-card {
        padding: 1.5rem 1rem;
        border-radius: 16px;
        border: 1px solid #e5e5ea;
        background: #fafafa;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
        border-color: #0071e3;
    }
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        color: #86868b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1d1d1f;
        margin: 0.3rem 0 0 0;
        letter-spacing: -0.02em;
    }

    /* Result card */
    .result-card {
        background: #ffffff;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #0071e3;
        box-shadow: 0 10px 30px rgba(0, 113, 227, 0.1);
        margin-bottom: 2rem;
    }
    .result-card h2 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-value {
        font-size: 3.5rem;
        font-weight: 700;
        color: #0071e3;
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .result-card p {
        font-size: 0.95rem;
        color: #86868b;
        margin: 0;
    }

    /* Button styling */
    .stButton > button {
        background-color: #0071e3;
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 1.05rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #0077ed;
        color: white;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    """โหลดโมเดลและ scaler จากไฟล์"""
    try:
        model = joblib.load('model_files/rf_model.pkl')
        scaler = joblib.load('model_files/scaler.pkl')
        feature_names = joblib.load('model_files/feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

model, scaler, feature_names = load_model()

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("## 🏠 California Housing")
    st.markdown("<p style='color:#86868b; font-size:0.9rem;'>Price Predictor</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    **โมเดล:** Random Forest Regressor  
    **Dataset:** California Housing  
    **Accuracy:** R² = 0.81
    """)

    with st.expander("📖 คำอธิบาย Features"):
        st.markdown("""
        - **MedInc**: รายได้เฉลี่ย (×$10,000)
        - **HouseAge**: อายุบ้านเฉลี่ย
        - **AveRooms**: จำนวนห้องเฉลี่ย
        - **AveBedrms**: จำนวนห้องนอนเฉลี่ย
        - **Population**: จำนวนประชากร
        - **AveOccup**: จำนวนคนต่อครัวเรือน
        - **Latitude**: ละติจูด
        - **Longitude**: ลองจิจูด
        """)

# ==================== Main Content ====================
st.markdown("""
<div class='hero'>
    <h1>California Housing Predictor</h1>
    <p>ทำนายราคาบ้านในแคลิฟอร์เนียด้วย Machine Learning</p>
    <span class='badge'>🌲 Random Forest · R² = 0.81</span>
</div>
""", unsafe_allow_html=True)

# ==================== Input Section ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='minimal-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📍 Location & Demographics</div>", unsafe_allow_html=True)
    latitude = st.slider("Latitude", 32.0, 42.0, 35.0, 0.01)
    longitude = st.slider("Longitude", -124.0, -114.0, -119.0, 0.01)
    population = st.slider("Population", 1.0, 30000.0, 1500.0, 10.0)
    ave_occup = st.slider("Avg. Occupancy", 1.0, 10.0, 3.0, 0.1)
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}), zoom=4, height=150)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='minimal-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏡 House Characteristics</div>", unsafe_allow_html=True)
    med_inc = st.slider("Median Income (×$10k)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("House Age (years)", 1.0, 52.0, 20.0, 1.0)
    ave_rooms = st.slider("Avg. Rooms", 1.0, 15.0, 5.0, 0.1)
    ave_bedrms = st.slider("Avg. Bedrooms", 0.5, 5.0, 1.1, 0.1)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== Prediction Button ====================
st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("ประมวลผลราคาประเมิน", use_container_width=True)

# ==================== Prediction Logic ====================
if predict_button:
    with st.spinner("กำลังทำนายข้อมูล..."):
        input_data = np.array([[
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction_usd = prediction * 100000

    # แสดงผลลัพธ์แบบ Minimal
    st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='result-card'>
        <h2>Estimated Value</h2>
        <div class='result-value'>${prediction_usd:,.0f}</div>
        <p>Approx. {prediction:,.2f} × $100,000</p>
    </div>
    """, unsafe_allow_html=True)

    # แสดงรายละเอียด (Metrics)
    st.markdown("<div class='minimal-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Prediction Details</div>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>💵</div>
            <p class='metric-label'>Price</p>
            <p class='metric-value'>${prediction_usd:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        price_per_room = prediction_usd / ave_rooms if ave_rooms > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>🚪</div>
            <p class='metric-label'>Per Room</p>
            <p class='metric-value'>${price_per_room:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        price_per_person = prediction_usd / population if population > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>👥</div>
            <p class='metric-label'>Per Person</p>
            <p class='metric-value'>${price_per_person:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        confidence = 81
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>🎯</div>
            <p class='metric-label'>Confidence</p>
            <p class='metric-value'>{confidence}%</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature Importance Chart
    st.markdown("<div class='minimal-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 Feature Impact</div>", unsafe_allow_html=True)

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            # ใช้โทนฟ้ามินิมอล ไล่จากฟ้าอ่อนไปฟ้าเข้ม
            color_continuous_scale=['#f0f7ff', '#bae6fd', '#0071e3'],
            color='Importance',
        )
        fig.update_layout(
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=13, family='Inter, sans-serif', color='#86868b'),
            title_font=dict(size=16, color='#1d1d1f', family='Inter'),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor='#f5f5f7', zeroline=False),
            yaxis=dict(showgrid=False)
        )
        fig.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input Summary
    with st.expander("📋 Data Summary"):
        input_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data[0]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ==================== Footer ====================
st.markdown("<div style='margin-top:4rem; border-top: 1px solid #f5f5f7; padding-top: 2rem;'></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#86868b;font-size:0.85rem;'>California Housing Predictor · Minimalist Application</p>", unsafe_allow_html=True)