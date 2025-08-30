import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import numpy as np

class ReviewDataLoader:
    """Data loader for labeled restaurant reviews"""
    
    def __init__(self, data_path: str = "../data-collection/data/combined_labeled_reviews.json"):
        self.data_path = data_path
        self.label_mapping = {
            0: "authentic",
            1: "fake", 
            2: "low_quality",
            3: "irrelevant"
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def load_data(self) -> List[Dict]:
        """Load labeled reviews from JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def get_label_distribution(self, data: List[Dict]) -> Dict:
        """Get distribution of labels in the dataset"""
        label_counts = {}
        for item in data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            self.label_mapping[label]: count 
            for label, count in label_counts.items()
        }
    
    def prepare_dataset(self, test_size: float = 0.2, val_size: float = 0.1, 
                       random_state: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare train/validation/test splits for DistilBERT training"""
        
        # Load data
        data = self.load_data()
        print(f"Loaded {len(data)} labeled reviews")
        
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
        
        return train_dataset, val_dataset, test_dataset
    
    def get_class_weights(self, train_dataset: Dataset) -> List[float]:
        """Calculate class weights for imbalanced dataset"""
        labels = train_dataset['label']
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Calculate weights inversely proportional to class frequencies
        total_samples = len(labels)
        class_weights = total_samples / (len(unique_labels) * counts)
        
        return class_weights.tolist()

if __name__ == "__main__":
    # Test the data loader with combined dataset
    loader = ReviewDataLoader()
    train_ds, val_ds, test_ds = loader.prepare_dataset()
    
    # Show some examples
    print("\nSample training examples:")
    for i in range(3):
        example = train_ds[i]
        print(f"Text: {example['text'][:100]}...")
        print(f"Label: {example['label']} ({loader.label_mapping[example['label']]})")
        print()
