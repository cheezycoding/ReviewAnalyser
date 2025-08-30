#!/usr/bin/env python3
"""
Simple test script for the trained DistilBERT model
This script avoids device compatibility issues by using CPU only
"""

import torch
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

class SimpleModelTester:
    def __init__(self, model_path: str = "./models/distilbert_review_classifier_20250830_001529"):
        self.model_path = model_path
        self.label_names = ["authentic", "fake", "low_quality", "irrelevant"]
        
        # Force CPU usage to avoid device issues
        self.device = torch.device("cpu")
        
        print(f"Loading model from: {model_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict(self, text: str) -> dict:
        """Predict label for a single review text"""
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move inputs to CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            confidence = probabilities[0][predicted_label].item()
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "predicted_label": predicted_label,
            "predicted_class": self.label_names[predicted_label],
            "confidence": confidence,
            "probabilities": {
                self.label_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }

def main():
    """Test the model with various example reviews"""
    
    # Initialize tester
    tester = SimpleModelTester()
    
    # Test examples
    test_reviews = [
        "This restaurant has amazing food and great service! The staff was friendly and the atmosphere was perfect.",
        "Great place",
        "I had an incredible dining experience here. The chef personally came to our table and the food was absolutely delicious. Highly recommend!",
        "The food was okay but nothing special. Service was slow and the prices were too high for what you get.",
        "This is a fake review written by the restaurant owner to boost ratings.",
        "The weather was nice today and I went for a walk in the park.",
        "Absolutely terrible experience. The food was cold, the service was rude, and the place was dirty. Never coming back!",
        "The restaurant is located near the mall and has plenty of parking available.",
        "I love this place! The pasta is homemade and the wine selection is excellent. The owner is so friendly and remembers our names every time we visit.",
        "Good food, good service, good price."
    ]
    
    print("\n" + "="*60)
    print("MODEL TESTING RESULTS")
    print("="*60)
    
    for i, review in enumerate(test_reviews, 1):
        result = tester.predict(review)
        print(f"\n{i}. {result['text']}")
        print(f"   Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
        print(f"   Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"     {label}: {prob:.3f}")
    
    # Interactive testing
    print("\n" + "="*60)
    print("INTERACTIVE TESTING")
    print("="*60)
    print("Enter your own reviews to test (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter a review: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input:
            result = tester.predict(user_input)
            print(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
            print("Class probabilities:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.3f}")

if __name__ == "__main__":
    main()
