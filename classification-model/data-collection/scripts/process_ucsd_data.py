#!/usr/bin/env python3
"""
UCSD Google Local Data Processor - Targeted Low Quality Sampling
Process District of Columbia data and sample reviews with focus on low quality detection
"""

import json
import gzip
import random
from pathlib import Path

def parse_gzip_json(file_path: str):
    """Parse gzipped JSON file"""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.strip())

def sample_targeted_reviews(file_path: str, target_samples: int = 1000):
    """Sample reviews with focus on low quality (short) reviews"""
    print(f"📊 Sampling {target_samples} reviews from {file_path}...")
    
    # Separate reviews by length for targeted sampling
    short_reviews = []      # 5-50 chars (likely low quality)
    medium_reviews = []     # 51-200 chars (mixed)
    long_reviews = []       # 201+ chars (likely authentic)
    
    for review in parse_gzip_json(file_path):
        # Handle None values
        if review is None:
            continue
            
        text = review.get('text', '')
        if text is None:
            continue
            
        text = text.strip()
        
        # Skip if no text or too short
        if not text or len(text) < 5:
            continue
            
        # Skip if too long (likely spam/irrelevant)
        if len(text) > 2000:
            continue
            
        # Skip obvious spam
        spam_indicators = ['click here', 'visit our website', 'call now', 'http://', 'https://']
        if any(indicator in text.lower() for indicator in spam_indicators):
            continue
        
        review_data = {
            'text': text,
            'rating': review.get('rating', 0),
            'user_id': review.get('user_id', ''),
            'business_id': review.get('gmap_id', ''),
            'timestamp': review.get('time', 0),
            'length': len(text)
        }
        
        # Categorize by length
        if len(text) <= 50:
            short_reviews.append(review_data)
        elif len(text) <= 200:
            medium_reviews.append(review_data)
        else:
            long_reviews.append(review_data)
        
        # Stop when we have enough samples
        if len(short_reviews) >= target_samples * 0.4 and \
           len(medium_reviews) >= target_samples * 0.3 and \
           len(long_reviews) >= target_samples * 0.3:
            break
    
    print(f"📊 Found reviews by length:")
    print(f"  Short (5-50 chars): {len(short_reviews)}")
    print(f"  Medium (51-200 chars): {len(medium_reviews)}")
    print(f"  Long (201+ chars): {len(long_reviews)}")
    
    # Sample proportionally to get balanced dataset
    target_short = min(int(target_samples * 0.4), len(short_reviews))
    target_medium = min(int(target_samples * 0.3), len(medium_reviews))
    target_long = min(int(target_samples * 0.3), len(long_reviews))
    
    sampled_reviews = []
    
    if short_reviews:
        sampled_short = random.sample(short_reviews, target_short)
        sampled_reviews.extend(sampled_short)
        print(f"✅ Sampled {len(sampled_short)} short reviews")
    
    if medium_reviews:
        sampled_medium = random.sample(medium_reviews, target_medium)
        sampled_reviews.extend(sampled_medium)
        print(f"✅ Sampled {len(sampled_medium)} medium reviews")
    
    if long_reviews:
        sampled_long = random.sample(long_reviews, target_long)
        sampled_reviews.extend(sampled_long)
        print(f"✅ Sampled {len(sampled_long)} long reviews")
    
    # Shuffle the final dataset
    random.shuffle(sampled_reviews)
    
    print(f"📊 Final sample: {len(sampled_reviews)} reviews")
    print(f"  Short: {len([r for r in sampled_reviews if r['length'] <= 50])}")
    print(f"  Medium: {len([r for r in sampled_reviews if 51 <= r['length'] <= 200])}")
    print(f"  Long: {len([r for r in sampled_reviews if r['length'] > 200])}")
    
    return sampled_reviews



def save_for_labeling(reviews, output_file: str = "../data/ucsd_reviews_for_labeling.json"):
    """Save reviews for labeling"""
    # Remove length field for cleaner output
    clean_reviews = []
    for review in reviews:
        clean_review = {k: v for k, v in review.items() if k != 'length'}
        clean_reviews.append(clean_review)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_reviews, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(clean_reviews)} reviews to {output_file}")

def show_sample_reviews(reviews, num_samples: int = 10):
    """Show sample reviews by length category"""
    print(f"\n📝 Sample Reviews:")
    print("-" * 50)
    
    short_samples = [r for r in reviews if r['length'] <= 50][:3]
    medium_samples = [r for r in reviews if 51 <= r['length'] <= 200][:4]
    long_samples = [r for r in reviews if r['length'] > 200][:3]
    
    print("Short reviews (likely low quality):")
    for i, review in enumerate(short_samples, 1):
        print(f"  {i}. ({review['length']} chars): \"{review['text']}\"")
    
    print("\nMedium reviews (mixed quality):")
    for i, review in enumerate(medium_samples, 1):
        print(f"  {i}. ({review['length']} chars): \"{review['text'][:100]}...\"")
    
    print("\nLong reviews (likely authentic):")
    for i, review in enumerate(long_samples, 1):
        print(f"  {i}. ({review['length']} chars): \"{review['text'][:100]}...\"")

def main():
    """Main function"""
    print("🚀 UCSD Google Local Data Processor - Low Quality Focus")
    print("="*60)
    
    # File path - using District of Columbia
    file_path = "../data/review-District_of_Columbia_10.json.gz"
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        print("Please ensure the District of Columbia file is in the data folder")
        return
    
    # Sample 1000 reviews with focus on low quality
    reviews = sample_targeted_reviews(file_path, target_samples=1000)
    
    # Show sample reviews
    show_sample_reviews(reviews)
    
    # Save for labeling
    save_for_labeling(reviews)
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Use data/ucsd_reviews_for_labeling.json for manual labeling")
    print(f"2. Use existing Singapore labeling script for LLM labeling")
    print(f"3. Focus on labeling short reviews as LOW_QUALITY (2)")
    print(f"4. Combine with existing Singapore data for training")

if __name__ == "__main__":
    main()
