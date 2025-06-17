# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)
    
    # Encode target
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])

    y = df['Churn']
    X = df.drop('Churn', axis=1)

    # One-hot encode categorical
    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, le, scaler, X.columns.tolist()
