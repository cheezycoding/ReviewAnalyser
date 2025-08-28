# Filtering the Noise: ML for Trustworthy Location Reviews

A comprehensive machine learning system to evaluate the quality and relevancy of Google location reviews, built for the TikTok TechJam hackathon.

## 🎯 Project Vision
Build an end-to-end ML system that automatically assesses review authenticity and relevancy, helping platforms filter out spam, advertisements, fake reviews, and irrelevant content to improve user trust and experience.

## 🏗️ System Architecture
```
TikTok TechJam/
├── data-collection/             # ✅ COMPLETE
│   ├── data/                   # 1,140 Singapore restaurant reviews
│   ├── scripts/                # Collection pipeline (Places API + scraper)
│   └── google-reviews-scraper/ # Third-party scraper integration
├── ml-pipeline/                # 🔄 IN DEVELOPMENT
│   ├── data-analysis/          # EDA and data exploration
│   ├── annotation/             # Labeling tools and schemas
│   ├── features/               # Text embeddings and feature engineering
│   ├── models/                 # Baseline and transformer models
│   └── evaluation/             # Metrics and validation
├── inference-api/              # 🔮 PLANNED
│   ├── api/                    # FastAPI service
│   ├── models/                 # Trained model artifacts
│   └── monitoring/             # Performance tracking
└── README.md                   # This file
```

## 📊 Current Status

### ✅ Phase 1: Data Collection (COMPLETE)
- **1,140 authentic reviews** collected from 15 diverse Singapore restaurants
- **100% text population** - no empty review fields
- **Geographic diversity** - Multiple Singapore districts covered
- **Cuisine variety** - Chinese, Indian, Italian, Japanese, Cafes, Hawkers
- **Quality verified** - All reviews contain actual customer feedback

### 🔄 Phase 2: ML Pipeline (IN PROGRESS)
- **Policy Definition** - Define spam/fake/irrelevant categories
- **Data Annotation** - Create labeled training dataset
- **Feature Engineering** - Text embeddings + metadata features
- **Model Development** - Baseline classifiers → Fine-tuned transformers
- **Evaluation Framework** - Metrics, validation, error analysis

### 🔮 Phase 3: Production System (PLANNED)
- **Inference API** - Real-time review scoring service
- **Monitoring Dashboard** - Performance tracking and drift detection
- **Human-in-the-Loop** - Review moderation interface
- **Deployment** - Scalable cloud infrastructure

## 🎯 ML Objectives

### Primary Tasks
1. **Authenticity Detection**: Identify fake/spam reviews
2. **Relevancy Assessment**: Determine if content relates to the location
3. **Quality Scoring**: Rate review helpfulness and informativeness

### Policy Categories
- **Spam**: Promotional content, repeated text patterns
- **Fake**: AI-generated or incentivized reviews  
- **Irrelevant**: Off-topic content unrelated to the venue
- **Low Quality**: Uninformative, extremely short, or unhelpful reviews

## 🚀 Quick Start

### 1. Explore the Dataset
```bash
cd data-collection
# See data-collection/README.md for detailed instructions
python -c "import json; data=json.load(open('data/reviews_clean.json')); print(f'Reviews: {len(data)}')"
```

### 2. Start ML Development
```bash
# Coming soon - ML pipeline setup
cd ml-pipeline
# python setup.py install
```

### 3. Run Inference API
```bash
# Coming soon - API service
cd inference-api
# docker-compose up
```

## 📋 Data Overview

### Restaurant Coverage
- **15 venues** across Singapore with reviews
- **25 venues** in metadata (ready for expansion)
- **Balanced representation** across cuisine types and venue categories

### Review Characteristics
- **Average length**: 284 characters
- **Rating distribution**: 4.0-5.0 stars (authentic range)
- **Language**: Primarily English
- **Temporal range**: Recent reviews from active venues

## 🔬 Technical Approach

### Feature Engineering
- **Text Features**: TF-IDF, word embeddings, semantic similarity
- **Metadata Features**: Rating patterns, temporal signals, user behavior
- **Location Features**: Geographic consistency, venue-specific context

### Model Architecture
- **Baseline Models**: Logistic Regression, SVM for interpretability
- **Advanced Models**: DistilBERT, RoBERTa for contextual understanding
- **Ensemble Methods**: Multi-task learning for combined objectives

### Evaluation Strategy
- **Metrics**: Precision, Recall, F1-score, AUC-ROC
- **Validation**: Stratified cross-validation, temporal splits
- **Error Analysis**: Confusion matrices, failure case studies

## ⚠️ Ethical Considerations
- **Educational Purpose**: Hackathon project for learning and research
- **Data Privacy**: No personal information stored or shared
- **Responsible AI**: Bias detection and fairness evaluation
- **Platform Benefit**: Improving user experience, not harming businesses

## 🎯 Success Metrics
- **Data Collection**: ✅ 1,140+ quality reviews collected
- **Model Performance**: Target >85% accuracy on test set
- **Inference Speed**: <100ms response time for real-time scoring
- **System Reliability**: >99% uptime for production API

## 🤝 Contributing
This is a hackathon project focused on demonstrating ML capabilities for review quality assessment. The system is designed to be educational and showcase best practices in ML engineering.

---

**🚀 Ready to build trustworthy review systems!** The foundation is solid with quality data and a clear technical roadmap.