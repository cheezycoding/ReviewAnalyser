#!/usr/bin/env python3
"""
Setup script for configuring API keys
"""

import getpass
import argparse
from config import set_api_key, check_api_key

def main():
    parser = argparse.ArgumentParser(description="Setup API key")
    parser.add_argument("--api-type", choices=["openai", "gemini"], default="openai", 
                       help="Type of API key to configure")
    args = parser.parse_args()
    
    api_type = args.api_type
    
    print(f"🔑 {api_type.upper()} API Key Setup")
    print("=" * 40)
    
    # Check if already configured
    if check_api_key(api_type):
        print(f"\n{api_type.upper()} API key is already configured!")
        response = input("Do you want to update it? (y/N): ").strip().lower()
        if response != 'y':
            return
    
    if api_type == "openai":
        print(f"\nPlease enter your OpenAI API key:")
        print("(You can get one from: https://platform.openai.com/api-keys)")
        expected_prefix = "sk-"
    else:  # gemini
        print(f"\nPlease enter your Google Gemini API key:")
        print("(You can get one from: https://makersuite.google.com/app/apikey)")
        expected_prefix = "AI"
    
    print()
    
    # Use getpass for secure input (won't show the key as you type)
    api_key = getpass.getpass(f"{api_type.upper()} API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided!")
        return
    
    if not api_key.startswith(expected_prefix):
        print(f"❌ Invalid API key format! Should start with '{expected_prefix}'")
        return
    
    # Save the API key
    set_api_key(api_key, api_type)
    
    print(f"\n✅ {api_type.upper()} API key configured successfully!")
    print("You can now run the labeling scripts.")

if __name__ == "__main__":
    main()
