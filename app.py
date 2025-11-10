import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Try importing LightGBM
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Loan Payback Predictor", layout="centered")
st.title("💰 Loan Payback Probability App")

st.markdown("""
### 🎯 Project Goal
The goal of this app is to **predict the probability that a borrower will pay back their loan**.

A **loan payable** is the amount of money a borrower owes to the lender, which they must repay over time (usually with interest).  
This app helps estimate how likely it is that a borrower will successfully repay the loan.
""")

st.divider()

# -------------------------------
# Sidebar: User Inputs
# -------------------------------
st.sidebar.header("📋 Borrower Information")
st.sidebar.markdown("Please enter the following details to predict loan repayment probability.")

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700,
    help="Credit score represents a person's creditworthiness. A higher score (above 700) generally indicates good credit behavior."
)

income = st.sidebar.number_input(
    "Monthly Income (₹)",
    min_value=1000,
    value=50000,
    help="The borrower's monthly income. Higher income suggests better repayment ability."
)

loan_amount = st.sidebar.number_input(
    "Loan Amount (₹)",
    min_value=5000,
    value=200000,
    help="The total amount the borrower wants to borrow. Larger loans increase repayment risk."
)

loan_term = st.sidebar.number_input(
    "Loan Term (in months)",
    min_value=6,
    max_value=360,
    value=60,
    help="The total period for repayment. Longer terms may increase interest burden."
)

debt_to_income_ratio = st.sidebar.slider(
    "Debt-to-Income Ratio",
    0.0, 1.0, 0.3,
    help="This ratio compares total monthly debt payments to monthly income. Lower values (<0.4) are generally safer."
)

age = st.sidebar.number_input(
    "Borrower Age",
    min_value=18,
    max_value=80,
    value=30,
    help="The age of the borrower. Age can indirectly reflect financial stability."
)

employment_years = st.sidebar.number_input(
    "Years of Employment",
    min_value=0,
    max_value=40,
    value=5,
    help="Total years the borrower has been employed. More years indicate income stability."
)

st.divider()

# -------------------------------
# Create DataFrame for Input
# -------------------------------
input_data = pd.DataFrame({
    "credit_score": [credit_score],
    "income": [income],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "debt_to_income_ratio": [debt_to_income_ratio],
    "age": [age],
    "employment_years": [employment_years]
})

st.subheader("🧾 Borrower Summary")
st.dataframe(input_data)

# -------------------------------
# Build a Simple Model (for demo)
# -------------------------------
# Generate synthetic data for demonstration
np.random.seed(42)
data_size = 500
train_data = pd.DataFrame({
    "credit_score": np.random.randint(300, 900, data_size),
    "income": np.random.randint(10000, 200000, data_size),
    "loan_amount": np.random.randint(5000, 1000000, data_size),
    "loan_term": np.random.randint(6, 360, data_size),
    "debt_to_income_ratio": np.random.rand(data_size),
    "age": np.random.randint(18, 70, data_size),
    "employment_years": np.random.randint(0, 40, data_size),
})

# Create pseudo target: higher credit_score and income increase repayment probability
train_data["target"] = (
    (train_data["credit_score"] > 650).astype(int) +
    (train_data["income"] > 50000).astype(int) +
    (train_data["debt_to_income_ratio"] < 0.4).astype(int)
)
train_data["target"] = (train_data["target"] > 1).astype(int)

X = train_data.drop(columns=["target"])
y = train_data["target"]

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model selection
if LGBM_AVAILABLE:
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# Predict
scaled_input = scaler.transform(input_data)
prob = model.predict_proba(scaled_input)[:, 1][0]

# -------------------------------
# Show Prediction
# -------------------------------
st.subheader("📈 Prediction Result")

if prob >= 0.7:
    st.success(f"✅ High Probability of Loan Repayment: {prob*100:.2f}%")
elif prob >= 0.4:
    st.warning(f"⚠️ Moderate Probability of Loan Repayment: {prob*100:.2f}%")
else:
    st.error(f"❌ Low Probability of Loan Repayment: {prob*100:.2f}%")

st.markdown("""
### 🧠 Interpretation:
- **Above 70%** → Likely to repay the loan on time.  
- **40–70%** → Medium risk; depends on external factors.  
- **Below 40%** → High risk of default.  

---
### 💡 Key Insights:
- **Credit Score** shows financial discipline and repayment history.  
- **Debt-to-Income Ratio** shows how much income goes toward debt.  
- **Income & Employment Stability** reflect repayment ability.  
- **Loan Amount and Term** affect affordability and total burden.  
---
""")
