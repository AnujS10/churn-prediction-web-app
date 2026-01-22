import streamlit as st
import joblib
import numpy as np

model = joblib.load("churn_model_deploy.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn")

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

if contract == "Month-to-month":
    contract_val = 0
elif contract == "One year":
    contract_val = 1
else:
    contract_val = 2

if st.button("Predict Churn"):
    
    input_data = np.array([[tenure, monthly_charges, total_charges, contract_val]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    
    if prediction[0] == 1:
        st.error(f"⚠ Customer is likely to CHURN (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Customer is likely to STAY (Confidence: {(1-probability)*100:.2f}%)")
