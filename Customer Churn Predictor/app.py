import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #0b0d12;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- HEADER ---------- */

    .app-header {
        padding: 0.5rem 0 1.8rem 0;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f5f7fa;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: #8f96a3;
        margin-top: 0;
    }

    .model-badge {
        display: inline-block;
        padding: 5px 12px;
        border: 1px solid #2d323c;
        border-radius: 20px;
        color: #aeb5c0;
        font-size: 0.75rem;
        margin-top: 10px;
        background: #11141a;
    }

    /* ---------- SECTION HEADERS ---------- */

    .section-title {
        font-size: 1.05rem;
        font-weight: 650;
        color: #f1f3f5;
        margin-bottom: 0.15rem;
    }

    .section-description {
        font-size: 0.75rem;
        color: #777f8c;
        margin-bottom: 1rem;
    }

    /* ---------- INPUT CONTAINERS ---------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #11141a;
        border: 1px solid #242933;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }

    /* ---------- LABELS ---------- */

    label {
        color: #aeb5c0 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }

    /* ---------- SELECTBOX ---------- */

    div[data-baseweb="select"] > div {
        background-color: #181b22;
        border: 1px solid #292e38;
        border-radius: 7px;
        color: #e7eaf0;
        min-height: 40px;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #444b58;
    }

    /* ---------- NUMBER INPUT ---------- */

    div[data-testid="stNumberInput"] input {
        background-color: #181b22;
        border: 1px solid #292e38;
        color: #e7eaf0;
        border-radius: 7px;
    }

    /* ---------- SLIDER ---------- */

    div[data-testid="stSlider"] {
        padding-top: 0.2rem;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 8px;
        border: 1px solid #ff4b4b;
        background: #ff4b4b;
        color: white;
        font-size: 0.9rem;
        font-weight: 600;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #e63f3f;
        border-color: #e63f3f;
        color: white;
    }

    /* ---------- RESULT ---------- */

    .result-label {
        color: #7f8794;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.25rem;
    }

    .result-probability {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f5f7fa;
        line-height: 1.1;
    }

    .risk-high {
        color: #ff6464;
        font-size: 1.25rem;
        font-weight: 650;
    }

    .risk-medium {
        color: #f0b84b;
        font-size: 1.25rem;
        font-weight: 650;
    }

    .risk-low {
        color: #55c58a;
        font-size: 1.25rem;
        font-weight: 650;
    }

    .result-description {
        color: #8f96a3;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    /* ---------- PROGRESS BAR ---------- */

    .risk-bar {
        width: 100%;
        height: 7px;
        background: #242933;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 15px;
    }

    .risk-fill-high {
        height: 100%;
        background: #ff4b4b;
        border-radius: 10px;
    }

    .risk-fill-medium {
        height: 100%;
        background: #f0b84b;
        border-radius: 10px;
    }

    .risk-fill-low {
        height: 100%;
        background: #55c58a;
        border-radius: 10px;
    }

    /* ---------- RECOMMENDATION ---------- */

    .recommendation-title {
        font-size: 0.9rem;
        font-weight: 650;
        color: #f1f3f5;
        margin-bottom: 0.45rem;
    }

    .recommendation-text {
        color: #9aa1ad;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #555c68;
        font-size: 0.7rem;
        padding-top: 1.8rem;
        margin-top: 2rem;
        border-top: 1px solid #20242c;
    }

    /* ---------- REMOVE EXTRA STREAMLIT SPACE ---------- */

    .stMarkdown {
        margin-bottom: 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="app-header">
    <div class="app-title">Customer Churn Predictor</div>
    <div class="app-subtitle">
        Predict customer retention risk using a machine learning model.
    </div>
    <div class="model-badge">
        Random Forest &nbsp;•&nbsp; 78% Model Accuracy
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():
    try:
        model = joblib.load("churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("feature_names.pkl")
        return model, scaler, features

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


model, scaler, features = load_models()

if model is None:
    st.stop()


# ============================================================
# CUSTOMER PROFILE
# ============================================================

with st.container(border=True):

    st.markdown("""
    <div class="section-title">Customer Profile</div>
    <div class="section-description">
        Basic information about the customer
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col2:
        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

    with col3:
        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

    with col4:
        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

    tenure = st.slider(
        "Customer Tenure (months)",
        0,
        72,
        12
    )


# ============================================================
# SERVICES
# ============================================================

st.write("")

with st.container(border=True):

    st.markdown("""
    <div class="section-title">Services & Connectivity</div>
    <div class="section-description">
        Select the services currently used by the customer
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col3:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    with col4:
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )


# ============================================================
# ENTERTAINMENT
# ============================================================

st.write("")

with st.container(border=True):

    st.markdown("""
    <div class="section-title">Streaming Services</div>
    <div class="section-description">
        Entertainment services subscribed to by the customer
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )


# ============================================================
# BILLING
# ============================================================

st.write("")

with st.container(border=True):

    st.markdown("""
    <div class="section-title">Billing & Contract</div>
    <div class="section-description">
        Contract, payment and customer billing information
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col3:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (auto)",
                "Credit card (auto)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=150.0,
            value=70.0
        )

    with col2:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=9000.0,
            value=500.0
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")
st.write("")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_clicked = st.button(
        "Predict Customer Risk",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    # --------------------------------------------------------
    # Prepare input data
    # --------------------------------------------------------

    input_data = {

        "gender":
            1 if gender == "Male" else 0,

        "SeniorCitizen":
            1 if senior_citizen == "Yes" else 0,

        "Partner":
            1 if partner == "Yes" else 0,

        "Dependents":
            1 if dependents == "Yes" else 0,

        "tenure":
            tenure,

        "PhoneService":
            1 if phone_service == "Yes" else 0,

        "MultipleLines_No":
            1 if multiple_lines == "No" else 0,

        "MultipleLines_No phone service":
            1 if multiple_lines == "No phone service" else 0,

        "MultipleLines_Yes":
            1 if multiple_lines == "Yes" else 0,

        "InternetService_Fiber optic":
            1 if internet_service == "Fiber optic" else 0,

        "InternetService_No":
            1 if internet_service == "No" else 0,

        "OnlineSecurity_No":
            1 if online_security == "No" else 0,

        "OnlineSecurity_No internet service":
            1 if online_security == "No internet service" else 0,

        "OnlineSecurity_Yes":
            1 if online_security == "Yes" else 0,

        "OnlineBackup_No":
            1 if online_backup == "No" else 0,

        "OnlineBackup_No internet service":
            1 if online_backup == "No internet service" else 0,

        "OnlineBackup_Yes":
            1 if online_backup == "Yes" else 0,

        "DeviceProtection_No":
            1 if device_protection == "No" else 0,

        "DeviceProtection_No internet service":
            1 if device_protection == "No internet service" else 0,

        "DeviceProtection_Yes":
            1 if device_protection == "Yes" else 0,

        "TechSupport_No":
            1 if tech_support == "No" else 0,

        "TechSupport_No internet service":
            1 if tech_support == "No internet service" else 0,

        "TechSupport_Yes":
            1 if tech_support == "Yes" else 0,

        "StreamingTV_No":
            1 if streaming_tv == "No" else 0,

        "StreamingTV_No internet service":
            1 if streaming_tv == "No internet service" else 0,

        "StreamingTV_Yes":
            1 if streaming_tv == "Yes" else 0,

        "StreamingMovies_No":
            1 if streaming_movies == "No" else 0,

        "StreamingMovies_No internet service":
            1 if streaming_movies == "No internet service" else 0,

        "StreamingMovies_Yes":
            1 if streaming_movies == "Yes" else 0,

        "Contract_One year":
            1 if contract == "One year" else 0,

        "Contract_Two year":
            1 if contract == "Two year" else 0,

        "PaperlessBilling":
            1 if paperless_billing == "Yes" else 0,

        "PaymentMethod_Bank transfer (auto)":
            1 if payment_method == "Bank transfer (auto)" else 0,

        "PaymentMethod_Credit card (auto)":
            1 if payment_method == "Credit card (auto)" else 0,

        "PaymentMethod_Electronic check":
            1 if payment_method == "Electronic check" else 0,

        "PaymentMethod_Mailed check":
            1 if payment_method == "Mailed check" else 0,

        "MonthlyCharges":
            monthly_charges,

        "TotalCharges":
            total_charges
    }

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])

    # Add missing columns
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep exact feature order
    input_df = input_df[features]

    # --------------------------------------------------------
    # Scale numeric columns
    # --------------------------------------------------------

    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    input_df[numeric_cols] = scaler.transform(
        input_df[numeric_cols]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prob = model.predict_proba(input_df)[0, 1]

    pred = "Churn" if prob > 0.5 else "No Churn"

    # --------------------------------------------------------
    # Determine Risk
    # --------------------------------------------------------

    if prob > 0.7:

        risk_level = "HIGH RISK"
        risk_class = "risk-high"
        bar_class = "risk-fill-high"

        description = (
            "This customer has a high probability of leaving."
        )

        recommendation = (
            "Immediate retention action is recommended. "
            "Consider a personalized discount, service upgrade, "
            "or direct customer outreach."
        )

    elif prob > 0.4:

        risk_level = "MEDIUM RISK"
        risk_class = "risk-medium"
        bar_class = "risk-fill-medium"

        description = (
            "This customer shows moderate signs of churn risk."
        )

        recommendation = (
            "Consider sending a targeted retention campaign "
            "with a personalized offer or service incentive."
        )

    else:

        risk_level = "LOW RISK"
        risk_class = "risk-low"

        bar_class = "risk-fill-low"

        description = (
            "This customer currently appears unlikely to churn."
        )

        recommendation = (
            "No immediate retention action is required. "
            "Continue monitoring the customer's engagement."
        )

    # --------------------------------------------------------
    # Result Header
    # --------------------------------------------------------

    st.write("")
    st.write("")

    st.markdown("""
    <div class="section-title">Prediction Result</div>
    <div class="section-description">
        Machine learning assessment based on the information provided
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Result Cards
    # --------------------------------------------------------

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        with st.container(border=True):

            st.markdown(
                '<div class="result-label">Churn Probability</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-probability">{prob:.1%}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="risk-bar">
                    <div class="{bar_class}"
                         style="width:{prob * 100}%;">
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with result_col2:

        with st.container(border=True):

            st.markdown(
                '<div class="result-label">Risk Assessment</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="{risk_class}">{risk_level}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-description">{description}</div>',
                unsafe_allow_html=True
            )

    with result_col3:

        with st.container(border=True):

            st.markdown(
                '<div class="result-label">Prediction</div>',
                unsafe_allow_html=True
            )

            if pred == "Churn":

                st.markdown(
                    '<div class="risk-high">Customer likely to churn</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="risk-low">Customer likely to stay</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="result-description">'
                'Based on the trained Random Forest model.'
                '</div>',
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.write("")

    with st.container(border=True):

        st.markdown(
            '<div class="recommendation-title">'
            'Recommended Action'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="recommendation-text">'
            f'{recommendation}'
            f'</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Customer Churn Prediction &nbsp;•&nbsp;
    Machine Learning Application &nbsp;•&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)