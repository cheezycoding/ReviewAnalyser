import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os
from datetime import datetime
from data_loader import ReviewDataLoader
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

class DistilBertReviewClassifier:
    """DistilBERT-based classifier for restaurant review quality assessment"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 4):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def tokenize_function(self, examples):
        """Tokenize the input texts"""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        report = classification_report(
            labels, predictions, 
            target_names=["authentic", "fake", "low_quality", "irrelevant"],
            output_dict=True
        )
        
        return {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "authentic_f1": report["authentic"]["f1-score"],
            "fake_f1": report["fake"]["f1-score"],
            "low_quality_f1": report["low_quality"]["f1-score"],
            "irrelevant_f1": report["irrelevant"]["f1-score"]
        }
    
    def train(self, train_dataset, val_dataset, output_dir: str = "./distilbert_review_classifier"):
        """Train the DistilBERT model"""
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to=None,  # Disable WandB for now
            dataloader_pin_memory=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def evaluate(self, test_dataset, trainer):
        """Evaluate the trained model on test set"""
        print("Evaluating on test set...")
        
        # Tokenize test dataset
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Generate detailed report
        report = classification_report(
            true_labels, pred_labels,
            target_names=["authentic", "fake", "low_quality", "irrelevant"],
            output_dict=True
        )
        
        # Save evaluation results
        results = {
            "test_accuracy": report["accuracy"],
            "test_macro_f1": report["macro avg"]["f1-score"],
            "test_weighted_f1": report["weighted avg"]["f1-score"],
            "class_metrics": {
                "authentic": report["authentic"],
                "fake": report["fake"],
                "low_quality": report["low_quality"],
                "irrelevant": report["irrelevant"]
            },
            "confusion_matrix": confusion_matrix(true_labels, pred_labels).tolist()
        }
        
        return results
    
    def predict_single(self, text: str) -> dict:
        """Predict label for a single review text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            confidence = probabilities[0][predicted_label].item()
        
        label_names = ["authentic", "fake", "low_quality", "irrelevant"]
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "predicted_label": predicted_label,
            "predicted_class": label_names[predicted_label],
            "confidence": confidence,
            "probabilities": {
                label_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }

class EnhancedDataLoader(ReviewDataLoader):
    """Enhanced data loader that can switch between datasets"""
    
    def __init__(self, enhanced_data_path: str = "../data-collection/data/enhanced_labeled_reviews.json",
                 fallback_data_path: str = "../data-collection/data/combined_labeled_reviews.json"):
        self.enhanced_data_path = enhanced_data_path
        self.fallback_data_path = fallback_data_path
        self.label_mapping = {
            0: "authentic",
            1: "fake", 
            2: "low_quality",
            3: "irrelevant"
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def load_enhanced_data(self) -> list:
        """Load enhanced dataset with additional fake reviews"""
        try:
            with open(self.enhanced_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded enhanced dataset with {len(data)} samples")
            return data
        except FileNotFoundError:
            print(f"Enhanced dataset not found at {self.enhanced_data_path}")
            return None
    
    def load_fallback_data(self) -> list:
        """Load fallback dataset (original combined dataset)"""
        with open(self.fallback_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded fallback dataset with {len(data)} samples")
        return data
    
    def prepare_dataset(self, use_enhanced: bool = True, test_size: float = 0.2, 
                       val_size: float = 0.1, random_state: int = 42):
        """Prepare train/validation/test splits"""
        
        # Try enhanced dataset first
        if use_enhanced:
            data = self.load_enhanced_data()
            if data is None:
                print("Falling back to original dataset...")
                data = self.load_fallback_data()
                use_enhanced = False
        else:
            data = self.load_fallback_data()
        
        # Show label distribution
        label_dist = self.get_label_distribution(data)
        print("Label distribution:")
        for label, count in label_dist.items():
            print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")
        
        # Convert to DataFrame for easier splitting
        df = pd.DataFrame(data)
        
        # Split into train and temp (test + val)
        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size, 
            random_state=random_state, stratify=df['label']
        )
        
        # Split temp into validation and test
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=1-val_ratio,
            random_state=random_state, stratify=temp_df['label']
        )
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        print(f"Train: {len(train_dataset)} samples")
        print(f"Validation: {len(val_dataset)} samples") 
        print(f"Test: {len(test_dataset)} samples")
        print(f"Dataset: {'Enhanced' if use_enhanced else 'Fallback'}")
        
        return train_dataset, val_dataset, test_dataset, use_enhanced

def main():
    """Main training pipeline with enhanced dataset"""
    
    # Initialize enhanced data loader
    print("Initializing enhanced data loader...")
    loader = EnhancedDataLoader()
    
    # Try enhanced dataset first
    print("Attempting to use enhanced dataset...")
    train_dataset, val_dataset, test_dataset, used_enhanced = loader.prepare_dataset(use_enhanced=True)
    
    # Initialize model
    print("Initializing DistilBERT model...")
    classifier = DistilBertReviewClassifier()
    
    # Create output directory with timestamp and dataset info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_type = "enhanced" if used_enhanced else "fallback"
    output_dir = f"./models/distilbert_review_classifier_{dataset_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the model
    print(f"Starting training pipeline with {dataset_type} dataset...")
    trainer = classifier.train(train_dataset, val_dataset, output_dir)
    
    # Evaluate on test set
    results = classifier.evaluate(test_dataset, trainer)
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Dataset used: {dataset_type.upper()}")
    print(f"Model saved to: {output_dir}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Macro F1: {results['test_macro_f1']:.4f}")
    print(f"Test Weighted F1: {results['test_weighted_f1']:.4f}")
    
    # Check if performance is acceptable
    macro_f1_threshold = 0.70  # Minimum acceptable macro F1
    fake_f1_threshold = 0.60   # Minimum acceptable fake review F1
    
    print(f"\nPerformance Analysis:")
    print(f"Macro F1: {results['test_macro_f1']:.4f} (threshold: {macro_f1_threshold})")
    print(f"Fake Review F1: {results['class_metrics']['fake']['f1-score']:.4f} (threshold: {fake_f1_threshold})")
    
    if results['test_macro_f1'] < macro_f1_threshold or results['class_metrics']['fake']['f1-score'] < fake_f1_threshold:
        print(f"\n⚠️  Performance below threshold! Consider:")
        print(f"   - Retraining with fallback dataset")
        print(f"   - Adjusting model hyperparameters")
        print(f"   - Using different model architecture")
    else:
        print(f"\n✅ Performance meets expectations!")
    
    # Test single prediction
    print("\nTesting single prediction...")
    test_text = "This restaurant has amazing food and great service!"
    prediction = classifier.predict_single(test_text)
    print(f"Text: {prediction['text']}")
    print(f"Predicted: {prediction['predicted_class']} (confidence: {prediction['confidence']:.3f})")

if __name__ == "__main__":
    main()
