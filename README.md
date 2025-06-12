## Customer Churn Prediction – MLOps Pipeline

This project implements a complete **MLOps pipeline** for customer churn prediction using structured telco data. It manages the full lifecycle of a machine learning model — from preprocessing and training to deployment and inference via API.

---

## Technologies Used

| Component          | Tool                |
|--------------------|---------------------|
| Model Training     | scikit-learn        |
| API                | FastAPI             |
| Experiment Tracking| MLflow              |
| Containerization   | Docker              |
| CI/CD              | GitHub Actions + Railway |
| Hosting            | Railway (free)      |

---

## Folder Structure

```
customer-churn-mlops/
├── .github/
│   └── workflows/
│       └── main.yml         # GitHub Action
├── data/                    # Dataset (local only)
├── models/                  # Trained model files (.pkl)
├── src/
│   ├── api.py               # FastAPI app
│   ├── train.py             # Training script
│   └── preprocess.py        # Data transformation
├── Dockerfile               # Docker container definition
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Create and activate virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python -m src.train
```
- Saves: `model.pkl`, `scaler.pkl`, `encoder.pkl`, `columns.pkl`
- Logs experiment via **MLflow**

### 4. (Optional) View MLflow experiment tracking
```bash
mlflow ui
```
Open in browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 5. Run the API Server

```bash
uvicorn src.api:app --reload
```

Visit Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
Use the `/predict` endpoint to send customer data and receive churn prediction!

---

## Sample Input (for `/predict`)

```json
{
  "gender": "Female",
  "SeniorCitizen": 1,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0,
  "TotalCharges": 95.0
}
```

Expected result:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.92
}
```

---

## GitHub Actions (CI)

This project uses GitHub Actions to automatically:
- Install dependencies
- Run the training pipeline on every push to `main`
- Log the model to MLflow

You can find the workflow in `.github/workflows/main.yml`.

---

## 🌐 Deployment (CD)

- Public URL: `https://customer-churn-mlops-production.up.railway.app/docs`
- Triggered automatically on **push to GitHub**
- **No manual deploy button required** on Railway

---

## Notes

- Model files (`.pkl`) are generated after running `train.py`
- The pipeline can be extended with GitHub Actions and Docker for full CI/CD

---