import streamlit as st
import pandas as pd
import joblib
model = joblib.load("model_trained.pk3")
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")
st.title("🏦 Loan Approval Prediction")
st.write("Predict whether a loan application is approved.")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Salary", min_value=0.0, value=50000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0)

loan_term = st.selectbox(
    "Loan Term",
    ["12 months", "24 months", "36 months", "48 months"]
)

employment_status = st.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)

residence_type = st.selectbox(
    "Residence Type",
    ["Owned", "Rented", "Mortgage"]
)

previous_default = st.selectbox(
    "Previous Default",
    ["Yes", "No"]
)
if st.button("Predict"):

    input_df = pd.DataFrame({
        "Age": [age],
        "Salary": [salary],
        "Credit_Score": [credit_score],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Employment_Status": [employment_status],
        "Residence_Type": [residence_type],
        "Previous_Default": [previous_default]
    })

    prediction = model.predict(input_df)
    if prediction[0] == "Yes":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
    