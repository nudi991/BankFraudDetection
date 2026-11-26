
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class FraudPredictionRequest(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: int
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: int
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int
    device_distinct_emails_8w: int

class FraudPredictionResponse(BaseModel):
    model_name: str
    model_version: str
    fraud_score: float
    is_fraud: bool
    threshold: float

class FraudDetectionModel:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.modelArtifact = joblib.load(modelPath)
        self.model = self.modelArtifact["model"]
        self.threshold = float(self.modelArtifact["threshold"])
        self.featureColumns = self.modelArtifact["featureColumns"]

    def predict(self, features):
        inputData = pd.DataFrame([features], columns=self.featureColumns)
        fraudScore = float(self.model.predict_proba(inputData)[0][1])
        isFraud = fraudScore >= self.threshold

        return {
            "fraud_score": fraudScore,
            "is_fraud": isFraud,
            "threshold": self.threshold
        }

class FraudDetectionService:
    def __init__(self, model):
        self.model = model
        self.modelName = "baf-xgboost"
        self.modelVersion = "1.0.0"

    def predict(self, features):
        prediction = self.model.predict(features)

        return {
            "model_name": self.modelName,
            "model_version": self.modelVersion,
            **prediction
        }

class FraudDetectionApi:
    def __init__(self):
        modelPath = Path(__file__).resolve().parent / "bankFraudXgboostModel.joblib"
        self.model = FraudDetectionModel(modelPath)
        self.service = FraudDetectionService(self.model)
        self.app = FastAPI(
            title="Bank Account Fraud Detection API",
            version="1.0.0"
        )

        self.registerRoutes()

    def registerRoutes(self):
        @self.app.post("/predict", response_model=FraudPredictionResponse)
        def predict(request: FraudPredictionRequest):
            try:
                features = request.model_dump()
                return self.service.predict(features)
            except Exception:
                raise HTTPException(status_code=500, detail="Fraud prediction failed")


fraudDetectionApi = FraudDetectionApi()
app = fraudDetectionApi.app
