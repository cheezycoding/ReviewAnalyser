# TikTok TechJam - Google Maps Review Quality Assessment

A machine learning system that automatically classifies Google reviews to identify authentic, fake, low-quality, and irrelevant content.

## 🚀 Project Status

### ✅ **ENHANCED MODEL DEPLOYED & LIVE**
- **Latest Model**: `distilbert_review_classifier_enhanced_20250830_090346`
- **Performance**: 90.1% accuracy, 77.3% macro F1
- **Fake Review Detection**: 91.9% F1 score (improved from 0%)
- **Production Ready**: Meets all performance thresholds
- **Live API**: https://review-classifier-api-370116201512.asia-southeast1.run.app

## 🎯 Overview

This system uses DistilBERT to classify restaurant reviews into four categories for **policy enforcement**:

### **Policy Enforcement Categories:**
- **Authentic Reviews** (92.5% F1): Genuine customer experiences - **ALLOWED**
- **Fake Reviews** (91.9% F1): Computer-generated/fraudulent content - **BLOCKED**
- **Low Quality Reviews** (89.2% F1): Poorly written/unhelpful content - **FLAGGED FOR REVIEW**
- **Irrelevant Reviews** (35.7% F1): Off-topic content - **REMOVED**

| Category | Description | Detection Rate |
|----------|-------------|----------------|
| **Authentic** | Genuine, detailed reviews | 92.5% F1 |
| **Fake** | Computer-generated/fraudulent | 91.9% F1 |
| **Low Quality** | Poorly written/unhelpful | 89.2% F1 |
| **Irrelevant** | Off-topic content | 35.7% F1 |

## 🏗️ Architecture

- **Model**: DistilBERT (distilbert-base-uncased)
- **Task**: 4-class sequence classification
- **Input**: Google review text (max 512 tokens)
- **Output**: Quality classification with confidence scores

## 📁 Project Structure

```
TikTok TechJam/
├── data-collection/                    # Data collection and preprocessing
│   ├── data/                          # Processed datasets
│   │   ├── enhanced_labeled_reviews.json    # Enhanced dataset (2,614 samples)
│   │   ├── combined_labeled_reviews.json    # Original dataset (2,140 samples)
│   │   └── raw_data/                  # Raw data files
│   │       ├── ucsd_labeled_reviews.json
│   │       ├── reviews_clean.json
│   │       ├── restaurants.parquet
│   │       └── ...
│   ├── google-reviews-scraper/        # Google Reviews scraper
│   └── scripts/                       # Data processing scripts
├── ml-pipeline/                       # Machine learning pipeline
│   ├── models/                        # Trained models
│   │   ├── distilbert_review_classifier_enhanced_20250830_090346/  # Latest
│   │   └── distilbert_review_classifier_20250830_001529/           # Original
│   ├── train_distilbert_enhanced.py  # Enhanced training script
│   ├── test_model.py                 # Model testing
│   ├── data_loader.py                # Data loading utilities
│   └── requirements.txt              # Python dependencies
└── inference-api/                     # API for model inference (✅ DEPLOYED)
```

## 🚀 Quick Start

### Option 1: Use the Live API (Recommended)
```bash
# Test the deployed API
curl -X POST "https://review-classifier-api-370116201512.asia-southeast1.run.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This restaurant has amazing food and great service!"}'

# View API documentation
open https://review-classifier-api-370116201512.asia-southeast1.run.app/docs
```

### Option 2: Local Development
```bash
# 1. Install Dependencies
cd ml-pipeline
pip install -r requirements.txt

# 2. Test the Model
python3 test_model.py

# 3. Train Enhanced Model
python3 train_distilbert_enhanced.py
```

### Option 3: Deploy Your Own API
```bash
# 1. Set up inference API
cd inference-api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Deploy to Google Cloud Run
gcloud run deploy review-classifier-api --source . --platform managed --region asia-southeast1 --allow-unauthenticated
```

## 🔄 How to Reproduce Results

### **Step 1: Data Collection & Preprocessing**
```bash
cd classification-model/data-collection

# Install dependencies
pip install -r scripts/requirements.txt

# Collect Google reviews (requires API key)
python scripts/collect_reviews.py

# Process and combine datasets
python scripts/combine_datasets.py
```

### **Step 2: Model Training**
```bash
cd classification-model/ml-pipeline

# Install dependencies
pip install -r requirements.txt

# Train the enhanced model
python train_distilbert_enhanced.py

# Expected results:
# - Test Accuracy: ~90.1%
# - Macro F1: ~77.3%
# - Fake Review F1: ~91.9%
```

### **Step 3: Model Evaluation**
```bash
# Test the trained model
python test_model.py

# Expected output:
# ✅ Model loaded successfully
# ✅ Prediction successful!
#    Sentiment: LABEL_0 (Authentic)
#    Confidence: 0.9967
```

### **Step 4: API Deployment**
```bash
cd classification-model/inference-api

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Deploy to Google Cloud Run
gcloud run deploy review-classifier-api --source . --platform managed --region asia-southeast1 --allow-unauthenticated
```

### **Expected Performance Metrics:**
- **Overall Accuracy**: 90.1%
- **Fake Review Detection**: 91.9% F1 score
- **Authentic Review Detection**: 92.5% F1 score
- **Low Quality Detection**: 89.2% F1 score

## 📊 Performance Metrics

### Enhanced Model (Latest)
- **Overall Accuracy**: 90.1%
- **Macro F1**: 77.3%
- **Weighted F1**: 89.3%

### Class-wise Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Authentic | 91.2% | 93.8% | 92.5% | 275 |
| Fake | 94.2% | 89.8% | 91.9% | 108 |
| Low Quality | 85.5% | 93.3% | 89.2% | 120 |
| Irrelevant | 71.4% | 23.8% | 35.7% | 21 |

## 🔍 Key Improvements

### Before Enhancement
- **Fake Review Detection**: 0% F1 score ❌
- **Class Imbalance**: Fake reviews only 2.9% of dataset
- **Overall Performance**: 59.0% macro F1

### After Enhancement
- **Fake Review Detection**: 91.9% F1 score ✅
- **Class Balance**: Fake reviews increased to 20.5%
- **Overall Performance**: 77.3% macro F1 (+18.3%)

## 📈 Dataset Information

### Enhanced Dataset
- **Total Samples**: 2,614
- **Authentic**: 1,371 (52.4%)
- **Fake**: 535 (20.5%) - Includes 474 Kaggle fake reviews
- **Low Quality**: 601 (23.0%)
- **Irrelevant**: 107 (4.1%)

### Data Sources
- **Original Reviews**: Google reviews from multiple sources
- **Kaggle Fake Reviews**: 474 computer-generated reviews for training
- **Labeling**: Manual annotation by domain experts

## 🎯 Use Cases & Policy Enforcement

### **Content Moderation Policies:**
1. **Fake Review Detection** (91.9% F1): **BLOCK** computer-generated/fraudulent reviews
2. **Authentic Review Validation** (92.5% F1): **ALLOW** genuine customer experiences
3. **Low Quality Filtering** (89.2% F1): **FLAG** poorly written/unhelpful content for review
4. **Irrelevant Content Removal** (35.7% F1): **REMOVE** off-topic content

### **Platform Integration:**
- **Google Maps Review Moderation**: Automatically filter low-quality content
- **Review Platform Quality Control**: Maintain review integrity and user trust
- **Real-time API Integration**: Use the live endpoint for instant classification
- **Data Cleaning**: Remove irrelevant content from review datasets

## 🔧 Technical Details

### Model Architecture
- **Base**: DistilBERT (6 layers, 768 dimensions)
- **Fine-tuning**: Sequence classification head
- **Tokenization**: WordPiece with 512 token limit
- **Optimization**: AdamW with weight decay

### Training Configuration
- **Batch Size**: 8 per device
- **Learning Rate**: 5e-5 with warmup
- **Epochs**: 5 with early stopping
- **Evaluation**: Every 100 steps

## 🚨 Important Notes

1. **Model Reliability**: Fake review detection (91.9% F1) is highly reliable
2. **Data Quality**: Enhanced with high-quality Kaggle fake reviews
3. **Production Use**: Model meets performance thresholds for deployment
4. **External Validation**: Recommended before full production use

## 🤝 Contributing

1. **Test on External Data**: Validate performance on different review sources
2. **Improve Data Quality**: Add more diverse review samples
3. **Model Architecture**: Experiment with different transformer models
4. **Data Augmentation**: Implement techniques for minority classes

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer models
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

## 🔮 Future Enhancements

1. **Multi-language Support**: Extend to other languages
2. **Real-time API**: ✅ **COMPLETED** - REST API deployed on Google Cloud Run
3. **Active Learning**: Continuous model improvement
4. **Ensemble Methods**: Combine multiple models
5. **Cross-domain Adaptation**: Adapt to different review types
6. **Model Optimization**: Quantization and compression for faster inference
7. **Monitoring & Analytics**: Add usage tracking and performance monitoring

## 📄 License

This project is part of the TikTok TechJam review quality assessment system.

## 🌐 Live API Endpoints

### Base URL
```
https://review-classifier-api-370116201512.asia-southeast1.run.app
```

### Available Endpoints
- **Health Check**: `GET /health`
- **Single Prediction**: `POST /predict`
- **Batch Prediction**: `POST /predict/batch`
- **Model Info**: `GET /model-info`
- **API Documentation**: `GET /docs`

### Example Usage
```bash
# Single review classification
curl -X POST "https://review-classifier-api-370116201512.asia-southeast1.run.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This restaurant has amazing food and great service!"}'

# Batch classification
curl -X POST "https://review-classifier-api-370116201512.asia-southeast1.run.app/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"reviews": ["Great food!", "This place is AMAZING!!! 5 stars!!!", "The food was okay but expensive"]}'
```

---

**Status**: ✅ **Production Ready** - Enhanced model successfully deployed with 91.9% fake review detection accuracy and live API available.