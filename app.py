import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("models/model.pkl")
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")
best_model_name = joblib.load("models/best_model_name.pkl")

st.title("Customer Churn Prediction")
st.markdown("Fill in customer details below to predict churn.")

form = st.form(key="churn_form")

gender = form.selectbox("Gender", ["Female", "Male"])
senior = form.selectbox("Senior Citizen", ["No", "Yes"])
SeniorCitizen = 1 if senior == "Yes" else 0
Partner = form.selectbox("Partner", ["Yes", "No"])
Dependents = form.selectbox("Dependents", ["Yes", "No"])
tenure = form.slider("Tenure (months)", 0, 72, 12)
PhoneService = form.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = form.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = form.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = form.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = form.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = form.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = form.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = form.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = form.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = form.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = form.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = form.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = form.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = form.number_input("Total Charges", min_value=0.0, value=300.0)

submit = form.form_submit_button("Predict")

if submit:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    df_input = pd.DataFrame([input_dict])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=columns, fill_value=0)
    df_input = scaler.transform(df_input)

    prediction = model.predict(df_input)[0]
    result = "Yes (Customer will churn)" if prediction == 1 else "No (Customer will stay)"

    st.success(f"Prediction: {result}")
    st.info(f"Model used: {best_model_name}")
