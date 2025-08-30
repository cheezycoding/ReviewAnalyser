#!/usr/bin/env python3
"""
Test script to verify the model loads and works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def test_model_loading():
    """Test if the model loads correctly"""
    try:
        print("Testing model loading...")
        
        model_path = "./model"
        if not os.path.exists(model_path):
            print(f"❌ Model path {model_path} does not exist!")
            return False
        
        # Load tokenizer and model
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded successfully")
        
        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Model loaded successfully")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"✅ Model moved to {device}")
        
        # Test prediction
        test_text = "This restaurant has amazing food and great service!"
        print(f"\nTesting prediction with: '{test_text}'")
        
        inputs = tokenizer(
            test_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class indices to labels
        id2label = model.config.id2label
        sentiment = id2label[predicted_class]
        
        print(f"✅ Prediction successful!")
        print(f"   Sentiment: {sentiment}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Available labels: {list(id2label.values())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 All tests passed! The model is ready for deployment.")
    else:
        print("\n💥 Tests failed! Please check the model files and dependencies.")
        sys.exit(1)
