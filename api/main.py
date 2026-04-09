from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import URLRequest, PredictionResponse, BatchRequest
from api.predictor import PhishingPredictor

app = FastAPI(
    title="PhishGuard Browser API",
    description="Browser extension phishing URL detector",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = PhishingPredictor()


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "PhishGuard API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost",
        "roc_auc": 0.9956,
        "features": len(predictor.features),
        "threshold": predictor.threshold
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: URLRequest):
    return predictor.predict(req.url)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return {
        "results": predictor.predict_batch(req.urls)
    }