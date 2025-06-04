# src/train.py
import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.preprocess import load_and_preprocess

mlflow.set_experiment("Churn_Prediction")

# Load and preprocess data
X, y, encoder, scaler = load_and_preprocess("data/customer_churn.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
feature_columns = X_train.columns.tolist()
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model, encoder, scaler
joblib.dump(model, "models/model.pkl")
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_columns, "models/columns.pkl")
# Track with MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "random forest")
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "model")

print("Model trained and saved.")
