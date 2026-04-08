# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas   import URLRequest, PredictionResponse, BatchRequest
from api.predictor import PhishingPredictor

app = FastAPI(
    title       = "Phishing Domain Detection API",
    description = "Classifies URLs as phishing or legit using lexical ML features.",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Loaded once at startup — not on every request
predictor = PhishingPredictor()

# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Phishing Detection API is running"}

@app.get("/health")
def health():
    return {
        "status":    "healthy",
        "model":     "XGBoost",
        "features":  len(predictor.features),
        "threshold": predictor.threshold
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_single(req: URLRequest):
    result = predictor.predict(req.url)
    if result["prediction"] == "error":
        raise HTTPException(status_code=422,
                            detail="Could not extract features from URL")
    return result

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return predictor.predict_batch(req.urls)