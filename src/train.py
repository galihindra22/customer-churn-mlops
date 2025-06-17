import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocess import load_and_preprocess

# Load and preprocess dataset
X, y, encoder, scaler, input_columns = load_and_preprocess("data/customer_churn.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000)
}

best_model_name = None
best_model = None
best_accuracy = 0.0

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"Best Model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Save best model and artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(input_columns, "models/columns.pkl")
joblib.dump(best_model_name, "models/best_model_name.pkl")

print("Best model saved successfully.")