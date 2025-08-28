#!/usr/bin/env python3
"""
Review Collector
===============

Clean script to collect reviews from Google Maps URLs
and output in ML-ready format for authenticity detection.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

def load_restaurants(restaurants_file: str) -> List[Dict]:
    """Load restaurant data from parquet file"""
    try:
        df = pd.read_parquet(restaurants_file)
        restaurants = df.to_dict('records')
        print(f"📊 Loaded {len(restaurants)} restaurants from {restaurants_file}")
        return restaurants
    except Exception as e:
        print(f"❌ Error loading restaurants: {e}")
        return []

def collect_reviews_for_restaurant(restaurant: Dict[str, Any], scraper_dir: str, restaurant_index: int) -> bool:
    """Collect reviews for a single restaurant"""
    
    if not restaurant.get('google_maps_url'):
        print(f"⚠️ No Google Maps URL for {restaurant['name']}, skipping...")
        return False
    
    print(f"\n🚀 Collecting reviews for: {restaurant['name']}")
    print(f"   URL: {restaurant['google_maps_url']}")
    print(f"   Rating: {restaurant['rating']}")
    
    # Prepare custom parameters for this restaurant
    custom_params = {
        "restaurant_name": restaurant['name'],
        "restaurant_id": restaurant['place_id'],
        "restaurant_category": "restaurant",
        "restaurant_rating": restaurant['rating'],
        "restaurant_address": restaurant['formatted_address']
    }
    
    # Convert to JSON string for command line
    custom_params_json = json.dumps(custom_params)
    
    # Create unique config file for this restaurant
    unique_config = f"config_temp_{restaurant_index}.yaml"
    unique_json_path = f"../data/reviews_temp_{restaurant_index}.json"
    unique_ids_path = f"../data/reviews_seen_ids_temp_{restaurant_index}.txt"
    
    # Create the unique config file by copying and modifying the original
    config_path = os.path.join(scraper_dir, "config.yaml")
    unique_config_path = os.path.join(scraper_dir, unique_config)
    
    try:
        # Read original config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace paths with unique ones
        config_content = config_content.replace(
            'json_path: "../data/reviews_with_metadata.json"',
            f'json_path: "../data/reviews_temp_{restaurant_index}.json"'
        )
        config_content = config_content.replace(
            'seen_ids_path: "../data/reviews_seen_ids.txt"',
            f'seen_ids_path: "../data/reviews_seen_ids_temp_{restaurant_index}.txt"'
        )
        
        # Write unique config
        with open(unique_config_path, 'w') as f:
            f.write(config_content)
    
    except Exception as e:
        print(f"❌ Error creating config for {restaurant['name']}: {e}")
        return False
    
    # Prepare scraper command with unique config
    cmd = [
        "bash", "-c",
        f"cd {scraper_dir} && "
        f"source .venv311/bin/activate && "
        f"export CHROME_BIN='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' && "
        f"export CHROMEDRIVER_PATH='/Users/wazir/.wdm/drivers/chromedriver/mac64/139.0.7258.154/chromedriver-mac-arm64/chromedriver' && "
        f"python start.py "
        f"--url '{restaurant['google_maps_url']}' "
        f"--sort newest "
        f"--custom-params '{custom_params_json}' "
        f"--config {unique_config}"
    ]
    
    try:
        # Run the scraper with proper environment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully collected reviews for {restaurant['name']}")
            success = True
        else:
            print(f"❌ Scraper failed for {restaurant['name']}")
            print(f"   Error: {result.stderr}")
            success = False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Scraper timed out for {restaurant['name']}")
        success = False
    except Exception as e:
        print(f"❌ Error running scraper for {restaurant['name']}: {e}")
        success = False
    
    finally:
        # Clean up the temporary config file
        try:
            if os.path.exists(unique_config_path):
                os.remove(unique_config_path)
        except Exception:
            pass  # Ignore cleanup errors
    
    return success

def merge_review_files(temp_files: List[str], output_file: str):
    """Merge multiple temporary review files into one final file"""
    all_reviews = []
    all_seen_ids = set()
    
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    reviews = json.load(f)
                    for review in reviews:
                        review_id = review.get('review_id')
                        if review_id and review_id not in all_seen_ids:
                            all_reviews.append(review)
                            all_seen_ids.add(review_id)
                
                # Clean up temp file
                os.remove(temp_file)
                print(f"   ✅ Merged {temp_file}")
            else:
                print(f"   ⚠️ Temp file not found: {temp_file}")
        except Exception as e:
            print(f"   ❌ Error merging {temp_file}: {e}")
    
    # Save merged reviews
    if all_reviews:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_reviews, f, ensure_ascii=False, indent=2)
        print(f"   💾 Final file: {output_file} ({len(all_reviews)} reviews)")
    
    # Clean up temporary seen_ids files
    for i in range(1, len(temp_files) + 1):
        temp_ids_file = f"../data/reviews_seen_ids_temp_{i}.txt"
        if os.path.exists(temp_ids_file):
            os.remove(temp_ids_file)

def main():
    """Main function to collect reviews from all restaurants"""
    
    print("🍽️ Restaurant Review Collector")
    print("=" * 40)
    
    # Configuration
    restaurants_file = "data/restaurants.parquet"
    scraper_dir = "google-reviews-scraper"
    output_file = "data/reviews_clean.json"
    
    # Check if restaurants file exists
    if not Path(restaurants_file).exists():
        print(f"❌ Restaurants file not found: {restaurants_file}")
        print("   Please run fetch_restaurants.py first")
        return
    
    # Load restaurants
    restaurants = load_restaurants(restaurants_file)
    
    if not restaurants:
        print("❌ No restaurants found, exiting...")
        return
    
    # Show restaurants to be processed
    print(f"\n📋 Restaurants to process:")
    for i, restaurant in enumerate(restaurants, 1):
        print(f"   {i}. {restaurant['name']} ({restaurant['rating']}⭐)")
    
    # Ask for confirmation
    response = input(f"\n🤔 Process all {len(restaurants)} restaurants? (y/n): ").lower()
    if response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return
    
    # Process each restaurant with unique files
    successful = 0
    failed = 0
    temp_files = []
    
    for i, restaurant in enumerate(restaurants, 1):
        print(f"\n🔄 Processing {i}/{len(restaurants)}: {restaurant['name']}")
        
        if collect_reviews_for_restaurant(restaurant, scraper_dir, i):
            successful += 1
            temp_files.append(f"../data/reviews_temp_{i}.json")
        else:
            failed += 1
        
        # Small delay between restaurants
        if i < len(restaurants):
            print("   ⏳ Waiting 5 seconds before next restaurant...")
            time.sleep(5)
    
    # Debug: Check what temp files actually exist
    print(f"\n🔍 Debug: Checking for temp files...")
    actual_temp_files = []
    for i in range(1, len(restaurants) + 1):
        temp_file = f"../data/reviews_temp_{i}.json"
        if os.path.exists(temp_file):
            actual_temp_files.append(temp_file)
            print(f"   ✅ Found: {temp_file}")
        else:
            print(f"   ❌ Missing: {temp_file}")
    
    # Merge all temporary files into final output
    if actual_temp_files:
        print(f"\n🔄 Merging {len(actual_temp_files)} temporary files...")
        merge_review_files(actual_temp_files, output_file)
    else:
        print(f"\n❌ No temp files found to merge!")
    
    # Summary
    print(f"\n🎯 Review Collection Complete!")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Total: {len(restaurants)}")
    
    if successful > 0:
        print(f"\n💾 Reviews saved to: {output_file}")
        print(f"   Format: Clean ML-ready JSON for DistilBERT training")
        print(f"\n📋 Next steps:")
        print(f"   1. Review the collected data")
        print(f"   2. Label reviews for authenticity (authentic/fake)")
        print(f"   3. Train your DistilBERT model")

if __name__ == "__main__":
    main()
