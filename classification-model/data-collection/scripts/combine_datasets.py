#!/usr/bin/env python3
"""
Script to combine labeled_reviews.json and ucsd_labeled_reviews.json into a single file.
This script preserves both original files and creates a new combined dataset.
"""

import json
import os
from pathlib import Path

def combine_datasets():
    # Define file paths
    data_dir = Path("../data")
    labeled_reviews_path = data_dir / "labeled_reviews.json"
    ucsd_labeled_path = data_dir / "ucsd_labeled_reviews.json"
    combined_output_path = data_dir / "combined_labeled_reviews.json"
    
    # Check if input files exist
    if not labeled_reviews_path.exists():
        print(f"Error: {labeled_reviews_path} not found!")
        return
    
    if not ucsd_labeled_path.exists():
        print(f"Error: {ucsd_labeled_path} not found!")
        return
    
    print("Loading labeled_reviews.json...")
    with open(labeled_reviews_path, 'r', encoding='utf-8') as f:
        labeled_reviews = json.load(f)
    
    print("Loading ucsd_labeled_reviews.json...")
    with open(ucsd_labeled_path, 'r', encoding='utf-8') as f:
        ucsd_labeled_reviews = json.load(f)
    
    # Combine the datasets
    combined_reviews = labeled_reviews + ucsd_labeled_reviews
    
    # Save the combined dataset
    print(f"Saving combined dataset to {combined_output_path}...")
    with open(combined_output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_reviews, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  labeled_reviews.json: {len(labeled_reviews)} reviews")
    print(f"  ucsd_labeled_reviews.json: {len(ucsd_labeled_reviews)} reviews")
    print(f"  Combined dataset: {len(combined_reviews)} reviews")
    print(f"\nCombined file saved as: {combined_output_path}")
    print("Original files preserved!")

if __name__ == "__main__":
    combine_datasets()
