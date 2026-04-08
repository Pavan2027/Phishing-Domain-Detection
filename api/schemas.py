# api/schemas.py
from pydantic import BaseModel, field_validator
from typing import List

class URLRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def must_have_scheme(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

class PredictionResponse(BaseModel):
    url:         str
    prediction:  str    # "phishing" | "legit"
    confidence:  float  # probability of phishing (0–1)
    is_phishing: bool

class BatchRequest(BaseModel):
    urls: List[str]

    @field_validator("urls")
    @classmethod
    def max_hundred(cls, v):
        if len(v) > 100:
            raise ValueError("Max 100 URLs per batch")
        return v