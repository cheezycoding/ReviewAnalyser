from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Restaurant Review Sentiment Analysis API",
    description="API for analyzing sentiment of restaurant reviews using DistilBERT",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

class ReviewRequest(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class BatchReviewRequest(BaseModel):
    reviews: List[str]

class BatchReviewResponse(BaseModel):
    predictions: List[ReviewResponse]

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer, device
    
    try:
        model_path = "./model"
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

def predict_sentiment(text: str) -> Dict[str, Any]:
    """Predict sentiment for a single review"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class indices to labels
        id2label = model.config.id2label
        sentiment = id2label[predicted_class]
        
        # Get probabilities for all classes
        prob_dict = {id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": prob_dict
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Restaurant Review Sentiment Analysis API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.post("/predict", response_model=ReviewResponse)
async def predict_single_review(request: ReviewRequest):
    """Predict sentiment for a single review"""
    result = predict_sentiment(request.text)
    
    return ReviewResponse(
        text=request.text,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        probabilities=result["probabilities"]
    )

@app.post("/predict/batch", response_model=BatchReviewResponse)
async def predict_batch_reviews(request: BatchReviewRequest):
    """Predict sentiment for multiple reviews"""
    predictions = []
    
    for review in request.reviews:
        result = predict_sentiment(review)
        predictions.append(ReviewResponse(
            text=review,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        ))
    
    return BatchReviewResponse(predictions=predictions)

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": model.config.model_type,
        "num_labels": model.config.num_labels,
        "id2label": model.config.id2label,
        "label2id": model.config.label2id,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
