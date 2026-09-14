import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1.5rem;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    opacity: 0.75;
    margin-bottom: 25px;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Prediction button */
.stButton > button {
    width: 100%;
    height: 52px;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
}

/* Result card */
.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(128, 128, 128, 0.30);
    margin-top: 25px;
}

.result-title {
    font-size: 17px;
    opacity: 0.75;
    letter-spacing: 1px;
}

.probability {
    font-size: 52px;
    font-weight: 700;
    margin: 5px 0;
}

.risk {
    font-size: 23px;
    font-weight: 600;
    margin-bottom: 8px;
}

.description {
    font-size: 15px;
    opacity: 0.8;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    padding: 15px;
    font-size: 13px;
    opacity: 0.55;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD SAVED MODEL COMPONENTS
# ---------------------------------------------------------

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
threshold = joblib.load("threshold.pkl")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered telecom customer retention risk analysis'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------------
# CUSTOMER INFORMATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col3:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )


col1, col2 = st.columns(2)

with col1:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

with col2:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )


st.markdown("---")


# ---------------------------------------------------------
# SERVICE INFORMATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🌐 Service Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

with col3:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col2:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

with col3:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


st.markdown("---")


# ---------------------------------------------------------
# CONTRACT & BILLING
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">💳 Contract & Billing</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2, col3 = st.columns(3)

with col1:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0
    )


# ---------------------------------------------------------
# STORE USER INPUT
# ---------------------------------------------------------

customer_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}


input_df = pd.DataFrame([customer_data])


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

binary_mapping = {
    "No": 0,
    "Yes": 1
}


# Convert gender
input_df["gender"] = input_df["gender"].map({
    "Female": 0,
    "Male": 1
})


# Convert binary features
input_df["SeniorCitizen"] = input_df["SeniorCitizen"].map(binary_mapping)
input_df["Partner"] = input_df["Partner"].map(binary_mapping)
input_df["Dependents"] = input_df["Dependents"].map(binary_mapping)
input_df["PhoneService"] = input_df["PhoneService"].map(binary_mapping)
input_df["PaperlessBilling"] = input_df["PaperlessBilling"].map(binary_mapping)


# ---------------------------------------------------------
# ONE-HOT ENCODING
# ---------------------------------------------------------

input_df = pd.get_dummies(
    input_df,
    columns=[
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ],
    drop_first=False
)


# Match training feature names and order
input_df = input_df.reindex(
    columns=feature_names,
    fill_value=0
)


# Scale input data
input_scaled = scaler.transform(input_df)


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1.5, 1])

with col2:

    predict_button = st.button(
        "🔍  Predict Churn",
        use_container_width=True
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

    # Calculate churn probability
    churn_probability = model.predict_proba(
        input_scaled
    )[0][1]

    # Apply selected threshold
    prediction = int(
        churn_probability >= threshold
    )

    probability_percentage = churn_probability * 100


    # -----------------------------------------------------
    # DETERMINE RISK LEVEL
    # -----------------------------------------------------

    if probability_percentage >= 60:

        risk_level = "🔴 HIGH RISK"
        risk_color = "#ff4b4b"

        description = (
            "This customer has a high likelihood of churning. "
            "Consider targeted retention strategies."
        )

    elif probability_percentage >= 30:

        risk_level = "🟠 MEDIUM RISK"
        risk_color = "#ff9f43"

        description = (
            "This customer shows some signs of churn risk. "
            "Consider monitoring and engagement."
        )

    else:

        risk_level = "🟢 LOW RISK"
        risk_color = "#2ecc71"

        description = (
            "This customer has a relatively low likelihood "
            "of churning."
        )


    # -----------------------------------------------------
    # RESULT CARD
    # -----------------------------------------------------

    st.markdown(
        f'<div class="result-card">'
        f'<div class="result-title">CUSTOMER CHURN RISK</div>'
        f'<div class="probability">{probability_percentage:.2f}%</div>'
        f'<div class="risk" style="color:{risk_color};">'
        f'{risk_level}'
        f'</div>'
        f'<div class="description">{description}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


    # Probability bar
    st.progress(
        churn_probability,
        text=f"Churn Probability: {probability_percentage:.2f}%"
    )


    # -----------------------------------------------------
    # MODEL DECISION
    # -----------------------------------------------------

    if prediction == 1:

        st.warning(
            "⚠️ Model Decision: Customer is likely to churn."
        )

    else:

        st.success(
            "✅ Model Decision: Customer is likely to stay."
        )


# ---------------------------------------------------------
# INTERPRETATION GUIDE
# ---------------------------------------------------------

st.markdown("---")

st.markdown("### 📌 Risk Interpretation")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("🟢 **Low Risk**\n\nProbability below 30%")

with col2:
    st.warning("🟠 **Medium Risk**\n\nProbability between 30% and 60%")

with col3:
    st.error("🔴 **High Risk**\n\nProbability of 60% or above")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    '<div class="footer">'
    'Built with Python • Scikit-learn • Streamlit | '
    'Telecom Customer Churn Prediction'
    '</div>',
    unsafe_allow_html=True
)