import streamlit as st
import joblib
import numpy as np

# 1. Set up page configurations
st.set_page_config(page_title="Home Loan Predictor", page_icon="🏠", layout="centered")

# 2. Load the pre-trained model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Could not find model files. Please run 'python train_model.py' first!")
    st.stop()

# 3. Design the Web User Interface
st.title("🏠 Home Loan Eligibility Predictor")
st.write("Enter your financial parameters below to instantly calculate your loan approval probability.")
st.markdown("---")

# Layout forms cleanly using columns
col1, col2 = st.columns(2)

with col1:
    monthly_income = st.number_input("Monthly Income (INR)", min_value=0, value=50000, step=5000)
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

with col2:
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, step=10)
    existing_loan_text = st.selectbox("Any Existing Active Loans?", ["No, I don't have other loans", "Yes, I have ongoing active loans"])

# Convert dropdown string selection back to binary 0 or 1 for the ML Model
existing_loan = 1 if "Yes" in existing_loan_text else 0

st.markdown("---")

# 4. Handle Prediction Logic upon button submission
if st.button("Check Loan Status", type="primary", use_container_width=True):
    # Format and scale features
    features = np.array([[monthly_income, age, cibil_score, existing_loan]])
    scaled_features = scaler.transform(features)
    
    # Run Inference
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    confidence = round(probabilities[prediction] * 100, 2)
    
    # Render stylized output blocks dynamically without breaking UI flow
    if prediction == 1:
        st.balloons()
        st.success(f"### 🎉 Application Approved!\n\n**Model Match Confidence:** {confidence}%")
    else:
        st.error(f"### ❌ Application Rejected\n\n**Model Match Confidence:** {confidence}%\n\n*Try improving your CIBIL score or lowering active liability debt structures.*")