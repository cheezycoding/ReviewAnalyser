#!/usr/bin/env python3
"""
Restaurant Metadata Fetcher
============================

Clean script to fetch restaurant details from Google Places API
and generate Google Maps URLs for review collection.
"""

import os
import time
import json
import argparse
import urllib.parse
from typing import Dict, List, Optional
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# Approximate center of Singapore and default radius
SG_CENTER = (1.3521, 103.8198)
DEFAULT_RADIUS_METERS = 5000  # 5km; adjust per query

# Places API v1 endpoints
PLACES_SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_DETAILS_URL_TMPL = "https://places.googleapis.com/v1/places/{place_id}"

def load_api_key() -> str:
    """Load Google Maps API key from environment"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY not set. Please put it in .env")
    return api_key

def text_search(api_key: str, query: str, location: Optional[str] = None, 
                radius: Optional[int] = None, max_pages: int = 1) -> List[Dict]:
    """Search for businesses using Google Places API text search"""
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.types,nextPageToken",
        "Content-Type": "application/json",
    }
    
    body = {"textQuery": query}
    if location and radius:
        lat, lng = map(float, location.split(","))
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius),
            }
        }

    results = []
    next_page_token = None
    page = 0
    
    while True:
        if next_page_token:
            body["pageToken"] = next_page_token
            
        resp = requests.post(PLACES_SEARCH_TEXT_URL, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            print(f"Text search HTTP {resp.status_code}: {resp.text[:200]}")
            break
            
        data = resp.json()
        batch = data.get("places", [])
        results.extend(batch)
        next_page_token = data.get("nextPageToken")
        page += 1
        
        if not next_page_token or page >= max_pages:
            break
        time.sleep(1.0)  # Polite delay
        
    return results

def get_business_details(api_key: str, place_id: str) -> Dict:
    """Get detailed business information including Google Maps URL"""
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "id,displayName,formattedAddress,location,rating,userRatingCount,types,websiteUri,nationalPhoneNumber,googleMapsUri",
        "Content-Type": "application/json",
    }
    
    url = PLACES_DETAILS_URL_TMPL.format(place_id=place_id)
    resp = requests.get(url, headers=headers, timeout=30)
    
    if resp.status_code != 200:
        print(f"Details HTTP {resp.status_code}: {resp.text[:200]}")
        return {}
        
    return resp.json()

def construct_google_maps_url(place_id: str) -> str:
    """Construct Google Maps URL from place ID"""
    return f"https://maps.google.com/?cid={place_id}"

def search_restaurants(api_key: str, queries: List[str], max_restaurants: int = 10) -> List[Dict]:
    """Search for restaurants using multiple queries"""
    all_restaurants = []
    
    for query in queries:
        print(f"🔍 Searching for: {query}")
        
        # Search in different areas of Singapore
        locations = [
            "1.3521,103.8198",  # Central
            "1.3000,103.8000",  # South
            "1.4000,103.8000",  # North
            "1.3500,103.9000",  # East
            "1.3500,103.7000",  # West
        ]
        
        for location in locations:
            if len(all_restaurants) >= max_restaurants:
                break
                
            results = text_search(api_key, query, location, DEFAULT_RADIUS_METERS, max_pages=1)
            
            for place in results:
                if len(all_restaurants) >= max_restaurants:
                    break
                    
                # Get full details
                details = get_business_details(api_key, place["id"])
                if not details:
                    continue
                
                # Construct restaurant data
                restaurant = {
                    "place_id": details.get("id", ""),
                    "name": details.get("displayName", {}).get("text", ""),
                    "formatted_address": details.get("formattedAddress", ""),
                    "lat": details.get("location", {}).get("latitude", 0),
                    "lng": details.get("location", {}).get("longitude", 0),
                    "rating": details.get("rating", 0.0),
                    "user_ratings_total": details.get("userRatingCount", 0),
                    "types": details.get("types", []),
                    "phone": details.get("nationalPhoneNumber", ""),
                    "website": details.get("websiteUri", ""),
                    "google_maps_url": details.get("googleMapsUri") or construct_google_maps_url(details.get("id", "")),
                }
                
                # Only add if it's a restaurant and not already added
                if any("restaurant" in t.lower() or "food" in t.lower() for t in restaurant["types"]):
                    if not any(r["place_id"] == restaurant["place_id"] for r in all_restaurants):
                        all_restaurants.append(restaurant)
                        print(f"  ✅ Added: {restaurant['name']} ({restaurant['rating']}⭐)")
                
                time.sleep(0.5)  # Polite delay between API calls
                
            if len(all_restaurants) >= max_restaurants:
                break
                
        if len(all_restaurants) >= max_restaurants:
            break
    
    return all_restaurants

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fetch restaurant metadata from Google Places API")
    parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    parser.add_argument("--max-restaurants", type=int, default=10, help="Maximum restaurants to fetch (default: 10)")
    parser.add_argument("--output", default="../data/restaurants.parquet", help="Output file path")
    parser.add_argument("--query", help="Custom search query (e.g., 'hawker centers Singapore')")
    parser.add_argument("--append", action="store_true", help="Append to existing data instead of overwriting")
    
    args = parser.parse_args()
    
    # Load API key
    api_key = args.api_key or load_api_key()
    
    # Restaurant search queries
    # Use custom query if provided, otherwise default queries
    if args.query:
        queries = [args.query]
        print(f"🔍 Custom query: {args.query}")
    else:
        queries = [
            "restaurant",
            "cafe", 
            "food court",
            "hawker center",
            "fine dining",
            "casual dining"
        ]
    
    print("🏪 Singapore Restaurant Metadata Fetcher")
    print("=" * 50)
    
    # Fetch restaurants
    restaurants = search_restaurants(api_key, queries, args.max_restaurants)
    
    if not restaurants:
        print("❌ No restaurants found")
        return
    
    # Save to parquet
    new_df = pd.DataFrame(restaurants)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle append mode
    if args.append and output_path.exists():
        existing_df = pd.read_parquet(output_path)
        # Merge and deduplicate by place_id
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        df = combined_df.drop_duplicates(subset=['place_id'], keep='last')
        print(f"📈 Appended {len(new_df)} new, total: {len(df)} unique restaurants")
    else:
        df = new_df
    
    df.to_parquet(output_path, index=False)
    
    print(f"\n🎯 Successfully fetched {len(restaurants)} restaurants")
    print(f"💾 Saved to: {output_path}")
    print(f"📊 Data shape: {df.shape}")
    
    # Show sample
    print(f"\n📋 Sample restaurants:")
    for i, restaurant in enumerate(restaurants[:3], 1):
        print(f"  {i}. {restaurant['name']} ({restaurant['rating']}⭐)")
        print(f"     Address: {restaurant['formatted_address']}")
        print(f"     Maps: {restaurant['google_maps_url']}")
        print()

if __name__ == "__main__":
    main()
