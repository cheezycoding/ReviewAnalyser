# Restaurant Review Authenticity Classifier API

A FastAPI-based inference service for classifying restaurant reviews based on their authenticity and quality using a fine-tuned DistilBERT model.

## Features

- **Single Review Analysis**: Classify authenticity of individual reviews
- **Batch Processing**: Process multiple reviews at once
- **Confidence Scores**: Get confidence levels for predictions
- **Probability Distribution**: View probabilities for all authenticity classes
- **Health Checks**: Monitor API status and model health
- **Interactive Documentation**: Auto-generated API docs with Swagger UI

## Model Information

- **Base Model**: DistilBERT
- **Task**: Multi-class review authenticity classification
- **Labels**: 
  - LABEL_0: Authentic (genuine reviews)
  - LABEL_1: Fake (fake/spam reviews)
  - LABEL_2: Low Quality (poor quality reviews)
  - LABEL_3: Irrelevant (off-topic reviews)

## Setup

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Model**:
   ```bash
   python test_model.py
   ```

## Running the API

### Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health information
- `GET /model-info` - Model configuration details

### Prediction
- `POST /predict` - Analyze single review
- `POST /predict/batch` - Analyze multiple reviews

## Usage Examples

### Single Review Analysis
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This restaurant has amazing food and great service!"}'
```

### Batch Analysis
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"reviews": ["Great food!", "This place is AMAZING!!! 5 stars!!!", "The food was okay but expensive"]}'
```

### Python Client Example
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This restaurant has amazing food and great service!"}
)
result = response.json()
print(f"Classification: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"reviews": ["Great food!", "This place is AMAZING!!! 5 stars!!!", "The food was okay but expensive"]}
)
results = response.json()
for pred in results['predictions']:
    print(f"Text: {pred['text']} -> {pred['sentiment']} ({pred['confidence']:.4f})")
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Response Format

### Single Prediction
```json
{
  "text": "This restaurant has amazing food and great service!",
  "sentiment": "LABEL_0",
  "confidence": 0.9967,
  "probabilities": {
    "LABEL_0": 0.9967,
    "LABEL_1": 0.0018,
    "LABEL_2": 0.0012,
    "LABEL_3": 0.0003
  }
}
```

### Batch Prediction
```json
{
  "predictions": [
    {
      "text": "Great food!",
      "sentiment": "LABEL_0",
      "confidence": 0.8543,
      "probabilities": {...}
    },
    ...
  ]
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Successful prediction
- `422`: Validation error (invalid input)
- `500`: Internal server error (model loading/prediction issues)

## Performance

- **Model Loading**: ~2-3 seconds on startup
- **Inference Time**: ~50-100ms per prediction (CPU)
- **Memory Usage**: ~500MB for model + dependencies
- **Model Accuracy**: 90.1% test accuracy
- **Fake Review Detection**: 91.9% F1 score

## Troubleshooting

1. **Model Loading Issues**: Ensure the `model/` directory contains all required files
2. **Memory Issues**: The model requires ~500MB RAM
3. **CUDA Issues**: The API automatically falls back to CPU if CUDA is unavailable
4. **Port Conflicts**: Change the port in the uvicorn command if 8000 is occupied
5. **Deployment**: The API is live at https://review-classifier-api-370116201512.asia-southeast1.run.app

## Development

To modify the API:
1. Edit `main.py` for endpoint changes
2. Update `requirements.txt` for new dependencies
3. Test with `python test_model.py`
4. Restart the server to apply changes
