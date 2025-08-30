#!/usr/bin/env python3
"""
Reviewstwo Backend - Clean, Simple Google Maps Review Analysis
"""

import os
import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Reviewstwo Backend API",
    description="Clean backend API for Google Maps review analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CLASSIFICATION_API_URL = os.getenv("CLASSIFICATION_API_URL", "https://review-classifier-api-370116201512.asia-southeast1.run.app")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APIFY_API_KEY = os.getenv("APIFY_API_KEY")

# Pydantic models
class ReviewRequest(BaseModel):
    gmaps_url: str
    max_reviews: int = 20
    
    @validator('max_reviews')
    def validate_max_reviews(cls, v):
        if v < 1 or v > 100:
            raise ValueError('max_reviews must be between 1 and 100')
        return v

class ReviewResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Apify client for Google Maps scraping
async def scrape_gmaps_reviews(url: str, max_reviews: int) -> Dict[str, Any]:
    """Scrape reviews using Apify Google Maps Reviews Scraper"""
    try:
        from apify_client import ApifyClient
        
        client = ApifyClient(APIFY_API_KEY)
        
        run_input = {
            "startUrls": [{"url": url}],
            "maxReviews": max_reviews,
            "reviewsSort": "newest",
            "language": "en"
        }
        
        # Run the Actor
        run = client.actor("Xb8osYTtOjlsgI6k9").call(run_input=run_input)
        
        # Get results
        results = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            results.append(item)
        
        # Debug: Log the actual data structure
        logger.info(f"Apify returned {len(results)} items")
        if results:
            logger.info(f"First item keys: {list(results[0].keys())}")
            logger.info(f"First item sample: {str(results[0])[:500]}...")
        
        # Process results
        if not results:
            return {"restaurant_info": {}, "reviews": []}
        
        # Extract restaurant info from first result (each item is a review with restaurant info)
        first_result = results[0]
        restaurant_info = {
            "name": first_result.get("title", first_result.get("name", "Unknown")),
            "address": first_result.get("address", "Unknown"),
            "current_rating": first_result.get("totalScore", first_result.get("rating", 0.0)),
            "total_reviews": first_result.get("reviewsCount", 0),
            "categories": first_result.get("categories", []),
            "url": url
        }
        
        logger.info(f"Extracted restaurant info: {restaurant_info}")
        
        # Extract reviews - handle different possible data structures
        reviews = []
        for i, item in enumerate(results):
            logger.info(f"Processing item {i}: keys = {list(item.keys())}")
            
            # Check if item has reviews directly
            if "reviews" in item and item["reviews"]:
                logger.info(f"Item {i} has {len(item['reviews'])} reviews")
                for review in item["reviews"]:
                    reviews.append({
                        "text": review.get("text", ""),
                        "rating": review.get("rating", 0),
                        "timestamp": review.get("publishedAtDate", ""),
                        "reviewer_name": item.get("name", ""),
                        "is_local_guide": item.get("isLocalGuide", False)
                    })
            # Check if item IS a review
            elif "text" in item and item.get("text"):
                logger.info(f"Item {i} is a review with text: {item['text'][:100]}...")
                reviews.append({
                    "text": item.get("text", ""),
                    "rating": item.get("stars", item.get("rating", 0)),
                    "timestamp": item.get("publishedAtDate", ""),
                    "reviewer_name": item.get("reviewerName", ""),
                    "is_local_guide": item.get("reviewerIsLocalGuide", False)
                })
            else:
                logger.info(f"Item {i} doesn't match review structure")
        
        logger.info(f"Extracted {len(reviews)} reviews from Apify results")
        
        return {
            "restaurant_info": restaurant_info,
            "reviews": reviews[:max_reviews]
        }
        
    except Exception as e:
        logger.error(f"Apify scraping failed: {e}")
        return {"restaurant_info": {}, "reviews": []}

# Classification API call
async def classify_reviews(review_texts: List[str]) -> List[Dict[str, Any]]:
    """Send reviews to classification model"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CLASSIFICATION_API_URL}/predict/batch",
                json={"reviews": review_texts},
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()["predictions"]
            else:
                logger.error(f"Classification API error: {response.status_code}")
                return []
                
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return []

# OpenAI LLM analysis
async def analyze_with_llm(authentic_reviews: List[Dict], restaurant_info: Dict) -> Dict[str, Any]:
    """Analyze authentic reviews with OpenAI"""
    try:
        import openai
        
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare review data for LLM
        reviews_text = ""
        for review in authentic_reviews:
            reviews_text += f"Rating: {review['rating']}/5\n"
            reviews_text += f"Review: {review['text']}\n\n"
        
        prompt = f"""
        Analyze these authentic reviews for {restaurant_info['name']}:
        
        {reviews_text}
        
        Provide a JSON response with:
        - summary: Brief summary of the reviews
        - adjusted_rating: Adjusted rating based on authentic reviews
        - key_themes: List of main themes mentioned
        - sentiment: Overall sentiment (positive/negative/neutral)
        """
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # Parse the full LLM response
        llm_content = response.choices[0].message.content
        
        try:
            # Try to parse as JSON first
            import json
            parsed_response = json.loads(llm_content)
            
            return {
                "summary": parsed_response.get("summary", llm_content),
                "adjusted_rating": parsed_response.get("adjusted_rating", restaurant_info.get("current_rating", 0.0)),
                "key_themes": parsed_response.get("key_themes", []),
                "sentiment": parsed_response.get("sentiment", "neutral"),
                "raw_llm_response": llm_content  # Keep full response for debugging
            }
        except json.JSONDecodeError:
            # If not JSON, return the full text
            return {
                "summary": llm_content,
                "adjusted_rating": restaurant_info.get("current_rating", 0.0),
                "key_themes": [],
                "sentiment": "neutral",
                "raw_llm_response": llm_content
            }
        
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return {
            "summary": "LLM analysis unavailable",
            "adjusted_rating": restaurant_info.get("current_rating", 0.0),
            "key_themes": [],
            "sentiment": "neutral"
        }

# Main endpoint
@app.post("/api/analyze", response_model=ReviewResponse)
async def analyze_reviews(request: ReviewRequest):
    """Main endpoint to analyze Google Maps reviews"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting analysis for: {request.gmaps_url}")
        
        # Step 1: Scrape reviews
        scraped_data = await scrape_gmaps_reviews(request.gmaps_url, request.max_reviews)
        restaurant_info = scraped_data["restaurant_info"]
        reviews = scraped_data["reviews"]
        
        if not reviews:
            return ReviewResponse(
                success=False,
                message="No reviews found",
                error="Failed to scrape reviews from the provided URL"
            )
        
        # Step 2: Classify reviews
        review_texts = [review["text"] for review in reviews if review["text"]]
        logger.info(f"Sending {len(review_texts)} reviews to classification model")
        logger.info(f"Sample review texts: {[text[:100] + '...' for text in review_texts[:3]]}")
        
        classification_results = await classify_reviews(review_texts)
        
        # Step 3: Process results
        logger.info(f"Received {len(classification_results)} classification results")
        logger.info(f"Sample classification result: {classification_results[0] if classification_results else 'None'}")
        
        authentic_reviews = []
        fake_reviews = []
        
        # Map numeric classification labels to human-readable labels
        label_mapping = {
            "LABEL_0": "authentic",
            "LABEL_1": "fake", 
            "LABEL_2": "low_quality",
            "LABEL_3": "irrelevant"
        }
        
        for i, result in enumerate(classification_results):
            if i < len(reviews):
                review = reviews[i]
                classification = result.get("sentiment", "")
                mapped_classification = label_mapping.get(classification, classification)
                
                logger.info(f"Review {i}: {classification} -> {mapped_classification} (confidence: {result.get('confidence', 0):.3f})")
                
                if mapped_classification == "authentic":
                    authentic_reviews.append(review)
                else:
                    fake_reviews.append(review)
        
        # Step 4: LLM analysis (if we have authentic reviews)
        llm_analysis = None
        if authentic_reviews and OPENAI_API_KEY:
            llm_analysis = await analyze_with_llm(authentic_reviews, restaurant_info)
        
        # Step 5: Prepare response
        processing_time = time.time() - start_time
        
        response_data = {
            "restaurant_info": restaurant_info,
            "total_reviews": len(reviews),
            "authentic_reviews": len(authentic_reviews),
            "fake_reviews": len(fake_reviews),
            "llm_analysis": llm_analysis,
            "processing_time": processing_time
        }
        
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        return ReviewResponse(
            success=True,
            message="Analysis completed successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return ReviewResponse(
            success=False,
            message="Analysis failed",
            error=str(e)
        )

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "classification_api": CLASSIFICATION_API_URL,
        "openai_configured": bool(OPENAI_API_KEY),
        "apify_configured": bool(APIFY_API_KEY)
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Reviewstwo Backend API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
