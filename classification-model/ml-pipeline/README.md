# ML Pipeline - Google Review Quality Classifier

This directory contains the machine learning pipeline for training and evaluating a DistilBERT-based classifier that identifies the quality of Google reviews.

## 🎯 Project Overview

The system classifies Google reviews into four categories:
- **Authentic** (0): Genuine, high-quality reviews
- **Fake** (1): Computer-generated or fraudulent reviews  
- **Low Quality** (2): Poorly written or unhelpful reviews
- **Irrelevant** (3): Reviews not related to food/service quality

## 🚀 Current Status

### Enhanced Model Performance (Latest)
- **Model**: `distilbert_review_classifier_enhanced_20250830_090346`
- **Dataset**: Enhanced dataset with 2,614 samples (including 474 Kaggle fake reviews)
- **Test Accuracy**: 90.1%
- **Test Macro F1**: 77.3%
- **Fake Review Detection F1**: 91.9% ✅

### Key Improvements
- **Fake Review Detection**: Improved from 0% to 91.9% F1 score
- **Class Balance**: Fake reviews increased from 2.9% to 20.5% of dataset
- **Overall Performance**: Macro F1 improved by 18.3 percentage points

## 📁 Directory Structure

```
ml-pipeline/
├── models/                                    # Trained model checkpoints
│   ├── distilbert_review_classifier_enhanced_20250830_090346/  # Latest enhanced model
│   └── distilbert_review_classifier_20250830_001529/           # Original model
├── data_loader.py                            # Data loading and preprocessing
├── train_distilbert_enhanced.py             # Enhanced training script
├── test_model.py                             # Model testing and inference
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

## 🏗️ Architecture

### Model
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Architecture**: 6 transformer layers, 768 hidden dimensions
- **Task**: 4-class sequence classification
- **Max Sequence Length**: 512 tokens

### Training Configuration
- **Batch Size**: 8 per device
- **Learning Rate**: 5e-5 with warmup
- **Optimization**: AdamW with weight decay 0.01
- **Early Stopping**: Patience of 3 evaluation steps
- **Evaluation**: Every 100 steps

## 📊 Dataset Information

### Enhanced Dataset (`enhanced_labeled_reviews.json`)
- **Total Samples**: 2,614
- **Authentic**: 1,371 (52.4%)
- **Fake**: 535 (20.5%) - Includes 474 Kaggle fake reviews
- **Low Quality**: 601 (23.0%)
- **Irrelevant**: 107 (4.1%)

### Original Dataset (`combined_labeled_reviews.json`)
- **Total Samples**: 2,140
- **Authentic**: 1,371 (64.1%)
- **Fake**: 61 (2.9%) - Severely imbalanced
- **Low Quality**: 601 (28.1%)
- **Irrelevant**: 107 (5.0%)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Enhanced Model
```bash
python3 train_distilbert_enhanced.py
```

### 3. Test Model
```bash
python3 test_model.py
```

## 📈 Performance Metrics

### Enhanced Model Results
| Metric | Value | Status |
|--------|-------|--------|
| Test Accuracy | 90.1% | ✅ Excellent |
| Macro F1 | 77.3% | ✅ Good |
| Weighted F1 | 89.3% | ✅ Excellent |
| Fake Review F1 | 91.9% | ✅ Excellent |

### Class-wise Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Authentic | 91.2% | 93.8% | 92.5% | 275 |
| Fake | 94.2% | 89.8% | 91.9% | 108 |
| Low Quality | 85.5% | 93.3% | 89.2% | 120 |
| Irrelevant | 71.4% | 23.8% | 35.7% | 21 |

## 🔍 Model Reliability

### High Reliability (✅)
- **Authentic**: 276 test samples
- **Fake**: 108 test samples  
- **Low Quality**: 121 test samples

### Low Reliability (⚠️)
- **Irrelevant**: 21 test samples (too few for reliable F1)

## 📝 Usage Examples

### Single Prediction
```python
from test_model import SimpleModelTester

tester = SimpleModelTester()
result = tester.predict("This place has amazing food and great service!")
print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
```

### Batch Evaluation
```python
from data_loader import ReviewDataLoader
from train_distilbert_enhanced import DistilBertReviewClassifier

# Load data
loader = ReviewDataLoader()
train_dataset, val_dataset, test_dataset = loader.prepare_dataset()

# Evaluate model
classifier = DistilBertReviewClassifier()
results = classifier.evaluate(test_dataset, trainer)
```

## 🎯 Key Features

1. **Enhanced Fake Review Detection**: 91.9% F1 score vs 0% in original model
2. **Balanced Dataset**: Proper class distribution for training
3. **Robust Evaluation**: Stratified sampling and comprehensive metrics
4. **Production Ready**: Meets performance thresholds for deployment
5. **Easy Integration**: Simple API for predictions and evaluation

## 🔧 Customization

### Hyperparameters
- Modify `train_distilbert_enhanced.py` to adjust:
  - Learning rate, batch size, epochs
  - Early stopping patience
  - Model architecture parameters

### Data
- Add new labeled reviews to enhance dataset
- Adjust class weights for imbalanced scenarios
- Implement data augmentation techniques

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer models
- **Scikit-learn**: Machine learning utilities
- **Datasets**: Hugging Face dataset handling
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

## 🚨 Important Notes

1. **Model Performance**: The enhanced model significantly outperforms the original
2. **Data Quality**: Kaggle fake reviews improved fake review detection dramatically
3. **Validation**: Consider external validation for production deployment
4. **Monitoring**: Track performance on real-world data continuously

## 🤝 Contributing

1. Test on external datasets
2. Improve data quality and labeling
3. Experiment with different model architectures
4. Add data augmentation techniques
5. Implement cross-validation strategies

## 📄 License

This project is part of the TikTok TechJam review quality assessment system.
