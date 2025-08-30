# Reviewstwo Backend API

A clean, production-ready backend API for Google Maps review analysis with AI-powered classification and LLM insights.

**🌐 Live API**: https://reviewstwo-backend-370116201512.asia-southeast1.run.app

## 🚀 Current Status: **PRODUCTION READY**

**✅ What's Working:**
- **Real Google Maps Scraping** via Apify integration
- **AI Review Classification** (authentic vs fake detection)
- **OpenAI LLM Analysis** of authentic reviews
- **Clean, Fast API** with proper error handling
- **Real-time Data Processing** (15-45 seconds typical)

## 🏗️ Architecture

```
Frontend → Backend API → Apify (Google Maps) → Classification Model → OpenAI LLM → Response
```

### Core Components:
1. **Apify Integration**: Real Google Maps review scraping
2. **Classification API**: DistilBERT model for authentic/fake detection
3. **OpenAI LLM**: Intelligent analysis of authentic reviews
4. **FastAPI Backend**: Clean, documented REST API

## 📡 API Endpoints

### Main Endpoint: `POST /api/analyze`
**Purpose**: Analyze Google Maps reviews end-to-end

**Request Body:**
```json
{
  "gmaps_url": "https://maps.google.com/place/...",
  "max_reviews": 20
}
```

**Response:**
```json
{
  "success": true,
  "message": "Analysis completed successfully",
  "data": {
    "restaurant_info": {
      "name": "Restaurant Name",
      "address": "Full Address",
      "current_rating": 4.2,
      "total_reviews": 1250,
      "categories": ["Italian", "Restaurant"],
      "url": "Google Maps URL"
    },
    "total_reviews": 20,
    "authentic_reviews": 15,
    "fake_reviews": 5,
    "llm_analysis": {
      "summary": "AI-generated summary of authentic reviews",
      "adjusted_rating": "3.8",
      "key_themes": ["service", "food quality", "ambiance"],
      "sentiment": "mixed",
      "raw_llm_response": "Full LLM response"
    },
    "processing_time": 18.5
  }
}
```

### Health Check: `GET /health`
**Purpose**: Check API status and configuration

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-30T16:30:00.000000",
  "classification_api": "https://review-classifier-api-370116201512.asia-southeast1.run.app",
  "openai_configured": true,
  "apify_configured": true
}
```

### Documentation: `GET /docs`
**Purpose**: Interactive API documentation (Swagger UI)

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- Apify API key
- OpenAI API key
- Classification model API access

### Environment Variables
Create a `.env` file:
```bash
# Required
APIFY_API_KEY=your_apify_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional (has defaults)
CLASSIFICATION_API_URL=https://review-classifier-api-370116201512.asia-southeast1.run.app
```

### Installation
```bash
# Clone and setup
cd reviewstwo-backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start server
source .env
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
httpx==0.25.2
pydantic==2.5.0
python-dotenv==1.0.0
openai==1.102.0
apify-client==2.0.0
```

## 🧪 Testing

### Test the API
```bash
# Health check (Live)
curl https://reviewstwo-backend-370116201512.asia-southeast1.run.app/health

# Test analysis (Live)
curl -X POST "https://reviewstwo-backend-370116201512.asia-southeast1.run.app/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"gmaps_url": "https://maps.google.com/place/...", "max_reviews": 5}'

# Local testing
curl http://localhost:8001/health
curl -X POST "http://localhost:8001/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"gmaps_url": "https://maps.google.com/place/...", "max_reviews": 5}'
```

### Test Files
- `test_apify_scraper.py`: Test Apify integration directly
- `test_apify_backend.py`: Test full backend workflow

## 🔍 How It Works

### 1. Review Scraping (Apify)
- Accepts Google Maps URL
- Scrapes real reviews using Apify Actor
- Extracts restaurant info and review data
- **Real data, no simulation**

### 2. Review Classification
- Sends review text to DistilBERT model
- Classifies as: authentic, fake, low_quality, irrelevant
- Maps numeric labels to human-readable categories
- **High accuracy detection**

### 3. LLM Analysis
- Filters to only authentic reviews
- Sends to OpenAI GPT-4 for analysis
- Generates: summary, adjusted rating, key themes, sentiment
- **Intelligent insights**

### 4. Response Assembly
- Combines all data into structured response
- Includes processing time and error handling
- **Clean, consistent format**

## 📊 Performance

- **Typical Response Time**: 15-45 seconds
- **Review Limit**: 1-100 reviews per request
- **Rate Limiting**: 10 requests per minute per IP
- **Success Rate**: >95% for valid Google Maps URLs

## 🚀 Deployment

### GCP Cloud Run (Production)
**🌐 Live Service**: https://reviewstwo-backend-370116201512.asia-southeast1.run.app

**Deploy with source:**
```bash
gcloud run deploy reviewstwo-backend --source . --region asia-southeast1 --allow-unauthenticated --port 8000 --memory 1Gi --cpu 1 --max-instances 10
```

### Docker (Local/Other Cloud)
```bash
# Build image
docker build -t reviewstwo-backend .

# Run with environment variables
docker run -p 8000:8000 \
  -e APIFY_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  reviewstwo-backend
```

### Production Considerations
- **GCP Cloud Run**: Auto-scaling, pay-per-use, managed SSL
- **Environment Variables**: Set securely in Cloud Run console
- **Memory**: 1GB recommended for LLM processing
- **CPU**: 1 vCPU sufficient for most workloads
- **Scaling**: 0-10 instances based on demand

## 🔒 Security Features

- **Rate Limiting**: Prevents API abuse
- **Input Validation**: Pydantic models ensure data integrity
- **Error Handling**: No sensitive data leakage
- **CORS Support**: Configurable for frontend integration

## 📈 Recent Improvements

### ✅ Fixed Issues
- **LLM Response Truncation**: Now returns full analysis
- **Parameter Conflicts**: Clean endpoint without naming issues
- **Data Extraction**: Proper restaurant info and review parsing
- **Classification Mapping**: Numeric labels properly mapped to categories

### 🆕 New Features
- **Real-time Apify Integration**: Live Google Maps data
- **Full LLM Analysis**: Complete OpenAI insights
- **Debug Logging**: Comprehensive request tracking
- **Error Recovery**: Graceful failure handling

## 🎯 Frontend Integration

### Simple Integration
```javascript
const analyzeReviews = async (gmapsUrl, maxReviews = 20) => {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gmaps_url: gmapsUrl, max_reviews: maxReviews })
  });
  
  const result = await response.json();
  if (result.success) {
    return result.data;
  } else {
    throw new Error(result.error);
  }
};
```

### Key Data Points
- **Restaurant Info**: Display business details
- **Classification Results**: Show authentic vs fake counts
- **LLM Insights**: Present AI-generated analysis
- **Processing Status**: Real-time feedback

## 🐛 Troubleshooting

### Common Issues
1. **"No reviews found"**: Check if URL is valid Google Maps place
2. **Classification failures**: Verify classification API is accessible
3. **LLM errors**: Check OpenAI API key and quota
4. **Slow responses**: Normal for 20+ reviews (15-45 seconds)

### Debug Mode
Enable detailed logging in server logs:
- Apify scraping progress
- Classification results
- LLM processing steps
- Performance metrics

## 📞 Support

- **API Documentation**: `/docs` endpoint
- **Health Check**: `/health` endpoint
- **Error Messages**: Detailed error responses
- **Logs**: Comprehensive server logging

## 🚀 Future Enhancements

- **Batch Processing**: Multiple URLs in one request
- **Caching**: Redis-based result caching
- **Analytics**: Usage statistics and metrics
- **Webhooks**: Real-time result notifications
- **Multi-language**: Support for non-English reviews

---

**Status**: Production Ready ✅  
**Last Updated**: August 30, 2025  
**Version**: 1.0.0
