#!/usr/bin/env python3
"""
UCSD Review Labeling Script - Fast Processing
Label all UCSD reviews using general business classification
"""

import json
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import openai
from tqdm import tqdm
from config import get_api_key, check_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LabelingResult:
    """Result of labeling a single review"""
    review_id: str
    review_text: str
    predicted_label: int
    confidence: float
    reasoning: str
    processing_time: float

class UCSDReviewLabeler:
    """Fast LLM-based review classifier for UCSD data"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        
    def _create_prompt(self, review_text: str) -> str:
        """Create the classification prompt with general business focus"""
        
        prompt = f"""You are an expert at classifying business reviews. Classify the given review into one of these categories:

## Classification Guidelines:

**0 - AUTHENTIC**: Personal, natural language, specific details about service/experience, uses "I" pronouns, casual expressions, natural imperfections in writing, mentions specific interactions or services.

**1 - FAKE**: Marketing-style language, overly polished, generic promotional buzzwords, lacks personal voice, template-like structure, could apply to any business, uses clichéd phrases.

**2 - LOW_QUALITY**: Very brief (under 50 characters), generic statements, no specific information, unhelpful to customers, just basic praise without details.

**3 - IRRELEVANT**: Not about business experience - delivery issues, parking problems, construction complaints, personal stories unrelated to the business, nostalgic stories about the area.

## Review to Classify:
Business Review: {review_text}

Please respond with ONLY a JSON object in this exact format:
{{
    "label": 0,
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this classification was chosen"
}}

Label should be 0, 1, 2, or 3. Confidence should be between 0.0 and 1.0, where 1.0 is completely certain."""
        
        return prompt
    
    def classify_review(self, review_text: str, review_id: str) -> LabelingResult:
        """Classify a single review"""
        start_time = time.time()
        
        try:
            prompt = self._create_prompt(review_text)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a business review classifier. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=200    # Shorter response for speed
            )
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content)
            
            processing_time = time.time() - start_time
            
            return LabelingResult(
                review_id=review_id,
                review_text=review_text,
                predicted_label=result["label"],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error classifying review {review_id}: {e}")
            # Return a fallback result
            return LabelingResult(
                review_id=review_id,
                review_text=review_text,
                predicted_label=2,  # Default to low quality
                confidence=0.0,
                reasoning=f"Error during classification: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def classify_batch_fast(self, reviews: List[Dict], batch_size: int = 20) -> List[LabelingResult]:
        """Classify reviews in fast batches with minimal delays"""
        results = []
        
        for i in tqdm(range(0, len(reviews), batch_size), desc="Classifying reviews"):
            batch = reviews[i:i + batch_size]
            
            for review in batch:
                result = self.classify_review(
                    review_text=review["text"],
                    review_id=review.get("user_id", f"ucsd_{i}")
                )
                results.append(result)
                
                # Minimal delay - just 0.1 seconds between requests
                time.sleep(0.1)
            
            # Save progress every batch
            self._save_progress(results, f"ucsd_labeling_progress_{i//batch_size}.json")
            
            # Small batch delay
            time.sleep(0.5)
        
        return results
    
    def _save_progress(self, results: List[LabelingResult], filename: str):
        """Save progress to file"""
        progress_data = []
        for result in results:
            progress_data.append({
                "review_id": result.review_id,
                "text": result.review_text,
                "label": result.predicted_label,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "processing_time": result.processing_time
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)

def load_ucsd_reviews(file_path: str = "../data/ucsd_reviews_for_labeling.json") -> List[Dict]:
    """Load UCSD reviews for labeling"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    print(f"📊 Loaded {len(reviews)} UCSD reviews for labeling")
    return reviews

def save_labeled_reviews(results: List[LabelingResult], output_file: str = "../data/ucsd_labeled_reviews.json"):
    """Save labeled reviews in the same format as Singapore data"""
    labeled_data = []
    
    for result in results:
        labeled_data.append({
            "text": result.review_text,
            "label": result.predicted_label
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(labeled_data)} labeled reviews to {output_file}")

def analyze_results(results: List[LabelingResult]):
    """Analyze labeling results"""
    label_counts = {}
    total_confidence = 0
    total_time = 0
    
    for result in results:
        label = result.predicted_label
        label_counts[label] = label_counts.get(label, 0) + 1
        total_confidence += result.confidence
        total_time += result.processing_time
    
    print(f"\n📊 Labeling Results:")
    print("-" * 40)
    
    label_names = {0: "Authentic", 1: "Fake", 2: "Low Quality", 3: "Irrelevant"}
    for label, count in sorted(label_counts.items()):
        percentage = count / len(results) * 100
        print(f"  {label_names[label]}: {count} ({percentage:.1f}%)")
    
    print(f"\n📈 Performance:")
    print(f"  Average confidence: {total_confidence/len(results):.3f}")
    print(f"  Total processing time: {total_time:.1f}s")
    print(f"  Average time per review: {total_time/len(results):.2f}s")

def main():
    """Main labeling function"""
    print("🚀 UCSD Review Labeling - Fast Processing")
    print("="*50)
    
    # Check API key
    api_key = get_api_key()
    if not api_key:
        print("❌ OpenAI API key not found. Please check your configuration.")
        return
    
    # Load reviews
    reviews = load_ucsd_reviews()
    
    # Initialize labeler
    labeler = UCSDReviewLabeler(api_key)
    
    # Classify all reviews
    print(f"\n🔥 Starting fast classification of {len(reviews)} reviews...")
    results = labeler.classify_batch_fast(reviews, batch_size=20)
    
    # Analyze results
    analyze_results(results)
    
    # Save labeled reviews
    save_labeled_reviews(results)
    
    print(f"\n🎉 Labeling complete!")
    print(f"✅ Processed {len(results)} reviews")
    print(f"✅ Results saved to data/ucsd_labeled_reviews.json")

if __name__ == "__main__":
    main()
