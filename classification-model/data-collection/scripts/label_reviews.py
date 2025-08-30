#!/usr/bin/env python3
"""
Restaurant Review Labeling Script
Uses LLMs to classify reviews based on seed examples for:
- authentic: Personal, natural language, specific details
- fake: Marketing-style, overly polished, generic promotional language
- low_quality: Very brief, generic, unhelpful reviews
- irrelevant: Not about restaurant experience (delivery, parking, etc.)
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import openai
import google.generativeai as genai
from tqdm import tqdm
import argparse
from config import get_api_key, check_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LabelingResult:
    """Result of labeling a single review"""
    review_id: str
    restaurant_name: str
    review_text: str
    predicted_label: str
    confidence: float
    reasoning: str
    processing_time: float

class ReviewLabeler:
    """LLM-based review classifier using few-shot learning"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview", api_type: str = "openai"):
        self.api_type = api_type
        self.api_key = api_key
        self.model = model
        
        if api_type == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        else:  # gemini
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        
        self.seed_examples = self._load_seed_examples()
        
    def _load_seed_examples(self) -> Dict[str, List[Dict]]:
        """Load and organize seed examples by category"""
        seed_path = Path(__file__).parent.parent / "data" / "seed_examples.json"
        with open(seed_path, "r") as f:
            data = json.load(f)
        
        # Organize examples by label
        examples_by_label = {}
        for example in data["seed_examples"]:
            label = example["label"]
            if label not in examples_by_label:
                examples_by_label[label] = []
            examples_by_label[label].append(example)
        
        return examples_by_label
    
    def _create_prompt(self, review_text: str) -> str:
        """Create the classification prompt with few-shot examples"""
        
        # Build examples section
        examples_text = ""
        for label, examples in self.seed_examples.items():
            label_name = {0: "AUTHENTIC", 1: "FAKE", 2: "LOW_QUALITY", 3: "IRRELEVANT"}.get(label, f"LABEL_{label}")
            examples_text += f"\n## {label} - {label_name} Examples:\n"
            for i, example in enumerate(examples[:2], 1):  # Use 2 examples per category
                examples_text += f"{i}. Restaurant: {example['restaurant_name']}\n"
                examples_text += f"   Review: {example['review_text'][:200]}...\n"
                if 'notes' in example:
                    examples_text += f"   Notes: {example['notes']}\n\n"
                else:
                    examples_text += f"   Label: {example['label']}\n\n"
        
        prompt = f"""You are an expert at classifying restaurant reviews. Based on the following examples, classify the given review into one of these categories:

{examples_text}

## Classification Guidelines:

**0 - AUTHENTIC**: Personal, natural language, specific details about food/experience, uses "I" pronouns, casual expressions, natural imperfections in writing, mentions specific interactions or dishes.

**1 - FAKE**: Marketing-style language, overly polished, generic promotional buzzwords, lacks personal voice, template-like structure, could apply to any restaurant, uses clichéd phrases.

**2 - LOW_QUALITY**: Very brief (under 50 characters), generic statements, no specific information, unhelpful to customers, just basic praise without details.

**3 - IRRELEVANT**: Not about restaurant experience - delivery issues, parking problems, construction complaints, personal stories unrelated to dining, nostalgic stories about the area.

## Review to Classify:
Restaurant Review: {review_text}

Please respond with ONLY a JSON object in this exact format:
{{
    "label": 0,
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this classification was chosen"
}}

Label should be 0, 1, 2, or 3. Confidence should be between 0.0 and 1.0, where 1.0 is completely certain."""
        
        return prompt
    
    def classify_review(self, review_text: str, review_id: str, restaurant_name: str) -> LabelingResult:
        """Classify a single review"""
        start_time = time.time()
        
        try:
            prompt = self._create_prompt(review_text)
            
            if self.api_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a restaurant review classifier. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=500
                )
                content = response.choices[0].message.content.strip()
            else:  # gemini
                response = self.client.generate_content(prompt)
                content = response.text.strip()
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content)
            
            processing_time = time.time() - start_time
            
            return LabelingResult(
                review_id=review_id,
                restaurant_name=restaurant_name,
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
                restaurant_name=restaurant_name,
                review_text=review_text,
                predicted_label="low_quality",  # Safe fallback
                confidence=0.0,
                reasoning=f"Error during classification: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def classify_batch(self, reviews: List[Dict], batch_size: int = 10) -> List[LabelingResult]:
        """Classify a batch of reviews with progress tracking and incremental saving"""
        results = []
        
        for i in tqdm(range(0, len(reviews), batch_size), desc="Classifying reviews"):
            batch = reviews[i:i + batch_size]
            
            for review in batch:
                result = self.classify_review(
                    review_text=review["review_text"],
                    review_id=review["review_id"],
                    restaurant_name=review["restaurant_name"]
                )
                results.append(result)
                
                # Small delay to avoid rate limiting
                time.sleep(2)  # Increased from 0.1 to 2 seconds
            
            # Save progress after each batch to prevent data loss
            if results:
                self._save_progress(results, f"../data/labeled_reviews_batch_{i//batch_size}.json")
        
        return results
    
    def classify_batch_resume(self, reviews: List[Dict], batch_size: int = 10, batch_prefix: str = "resume_batch", start_batch_num: int = 0) -> List[LabelingResult]:
        """Classify a batch of reviews with progress tracking and incremental saving (for resume)"""
        results = []
        
        for i in tqdm(range(0, len(reviews), batch_size), desc="Classifying reviews"):
            batch = reviews[i:i + batch_size]
            
            for review in batch:
                result = self.classify_review(
                    review_text=review["review_text"],
                    review_id=review["review_id"],
                    restaurant_name=review["restaurant_name"]
                )
                results.append(result)
                
                # Small delay to avoid rate limiting
                time.sleep(2)  # Increased from 0.1 to 2 seconds
            
            # Save progress after each batch to prevent data loss
            # Use start_batch_num to avoid overwriting existing files
            batch_num = start_batch_num + (i // batch_size)
            if results:
                self._save_progress_resume(results, f"../data/{batch_prefix}_{batch_num}.json")
        
        return results
    
    def _save_progress(self, results: List[LabelingResult], output_path: str):
        """Save intermediate results to prevent data loss"""
        try:
            output_data = {
                "metadata": {
                    "total_reviews": len(results),
                    "label_distribution": {},
                    "average_confidence": 0.0,
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "is_incremental": True
                },
                "results": []
            }
            
            # Calculate statistics
            label_counts = {}
            total_confidence = 0.0
            
            for result in results:
                output_data["results"].append({
                    "review_id": result.review_id,
                    "restaurant_name": result.restaurant_name,
                    "review_text": result.review_text,
                    "predicted_label": result.predicted_label,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "processing_time": result.processing_time
                })
                
                label_counts[result.predicted_label] = label_counts.get(result.predicted_label, 0) + 1
                total_confidence += result.confidence
            
            output_data["metadata"]["label_distribution"] = label_counts
            output_data["metadata"]["average_confidence"] = total_confidence / len(results) if results else 0.0
            
            # Save to file
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Progress saved to {output_path} ({len(results)} reviews)")
            
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    def _save_progress_resume(self, results: List[LabelingResult], output_path: str):
        """Save intermediate results to prevent data loss (for resume)"""
        try:
            output_data = {
                "metadata": {
                    "total_reviews": len(results),
                    "label_distribution": {},
                    "average_confidence": 0.0,
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "is_incremental": True,
                    "is_resume": True
                },
                "results": []
            }
            
            # Calculate statistics
            label_counts = {}
            total_confidence = 0.0
            
            for result in results:
                output_data["results"].append({
                    "review_id": result.review_id,
                    "restaurant_name": result.restaurant_name,
                    "review_text": result.review_text,
                    "predicted_label": result.predicted_label,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "processing_time": result.processing_time
                })
                
                label_counts[result.predicted_label] = label_counts.get(result.predicted_label, 0) + 1
                total_confidence += result.confidence
            
            output_data["metadata"]["label_distribution"] = label_counts
            output_data["metadata"]["average_confidence"] = total_confidence / len(results) if results else 0.0
            
            # Save to file
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Resume progress saved to {output_path} ({len(results)} reviews)")
            
        except Exception as e:
            logger.error(f"Error saving resume progress: {e}")

def save_results(results: List[LabelingResult], output_path: str):
    """Save labeling results in DistilBERT format"""
    
    # Convert to DistilBERT format
    output_data = []
    label_counts = {}
    
    for result in results:
        # Convert string label to numeric if needed
        if isinstance(result.predicted_label, str):
            label_mapping = {
                "authentic": 0,
                "fake": 1,
                "low_quality": 2,
                "irrelevant": 3
            }
            numeric_label = label_mapping.get(result.predicted_label.lower(), 2)
        else:
            numeric_label = result.predicted_label
        
        # Add to results in DistilBERT format
        output_data.append({
            "text": result.review_text,
            "label": numeric_label
        })
        
        # Update statistics
        label_counts[numeric_label] = label_counts.get(numeric_label, 0) + 1
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Label distribution: {label_counts}")
    logger.info(f"Total examples: {len(output_data)}")

def main():
    parser = argparse.ArgumentParser(description="Label restaurant reviews using LLMs")
    parser.add_argument("--api-key", help="API key (optional if configured)")
    parser.add_argument("--api-type", choices=["openai", "gemini"], default="openai", help="Type of API to use")
    parser.add_argument("--model", default="gpt-4-turbo-preview", help="Model to use")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--output", default="../data/labeled_reviews.json", help="Output file path")
    parser.add_argument("--sample-size", type=int, help="Number of reviews to sample (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing batch files")
    
    args = parser.parse_args()
    
    # Set default model based on API type
    if args.api_type == "openai" and args.model == "gpt-4-turbo-preview":
        args.model = "gpt-4o-mini"  # Use cheaper model for cost efficiency
    
    # Get API key
    api_key = args.api_key or get_api_key(args.api_type)
    if not api_key:
        print(f"❌ {args.api_type.upper()} API key not found!")
        print(f"Please run: python setup_api_key.py --api-type {args.api_type}")
        return
    
    # Load reviews
    logger.info("Loading reviews...")
    reviews_path = Path(__file__).parent.parent / "data" / "reviews_clean.json"
    with open(reviews_path, "r") as f:
        reviews = json.load(f)
    
    if args.sample_size:
        import random
        random.seed(42)  # For reproducible sampling
        reviews = random.sample(reviews, min(args.sample_size, len(reviews)))
        logger.info(f"Sampled {len(reviews)} reviews for testing")
    
    # Check for existing batch files if resuming
    existing_results = []
    if args.resume:
        existing_results = check_existing_batches()
        if existing_results:
            logger.info(f"Found {len(existing_results)} existing labeled reviews, resuming...")
            # Remove already processed reviews
            processed_ids = {r.review_id for r in existing_results}
            reviews = [r for r in reviews if r["review_id"] not in processed_ids]
            logger.info(f"Remaining reviews to process: {len(reviews)}")
    
    # Initialize labeler
    labeler = ReviewLabeler(api_key=api_key, model=args.model, api_type=args.api_type)
    
    # Classify reviews
    logger.info(f"Starting classification of {len(reviews)} reviews...")
    new_results = labeler.classify_batch(reviews, batch_size=args.batch_size)
    
    # Combine existing and new results
    all_results = existing_results + new_results
    
    # Save final results
    save_results(all_results, args.output)
    
    logger.info("Classification complete!")

def check_existing_batches():
    """Check for existing batch files and load them"""
    batch_files = list(Path(__file__).parent.parent / "data").glob("labeled_reviews_batch_*.json")
    if not batch_files:
        return []
    
    all_results = []
    for batch_file in sorted(batch_files):
        try:
            with open(batch_file, "r") as f:
                data = json.load(f)
                for result_data in data["results"]:
                    result = LabelingResult(
                        review_id=result_data["review_id"],
                        restaurant_name=result_data["restaurant_name"],
                        review_text=result_data["review_text"],
                        predicted_label=result_data["predicted_label"],
                        confidence=result_data["confidence"],
                        reasoning=result_data["reasoning"],
                        processing_time=result_data["processing_time"]
                    )
                    all_results.append(result)
            logger.info(f"Loaded {len(data['results'])} reviews from {batch_file.name}")
        except Exception as e:
            logger.error(f"Error loading {batch_file}: {e}")
    
    return all_results

if __name__ == "__main__":
    main()
