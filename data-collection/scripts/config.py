"""
Configuration file for the review labeling system
"""

import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Configuration file path - look in parent directory (data-collection)
CONFIG_FILE = SCRIPT_DIR.parent / ".env"

def get_api_key(api_type="openai"):
    """
    Get API key from environment variable or config file
    """
    env_var = f"{api_type.upper()}_API_KEY"
    
    # First try environment variable
    api_key = os.getenv(env_var)
    if api_key:
        return api_key
    
    # Then try config file
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            for line in f:
                if line.startswith(f"{env_var}="):
                    return line.split("=", 1)[1].strip()
    
    return None

def set_api_key(api_key: str, api_type="openai"):
    """
    Save API key to config file
    """
    env_var = f"{api_type.upper()}_API_KEY"
    
    # Read existing config
    existing_lines = []
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            existing_lines = f.readlines()
    
    # Update or add the API key
    key_found = False
    for i, line in enumerate(existing_lines):
        if line.startswith(f"{env_var}="):
            existing_lines[i] = f"{env_var}={api_key}\n"
            key_found = True
            break
    
    if not key_found:
        existing_lines.append(f"{env_var}={api_key}\n")
    
    # Write back to file
    with open(CONFIG_FILE, "w") as f:
        f.writelines(existing_lines)
    
    # Set file permissions to be readable only by owner
    os.chmod(CONFIG_FILE, 0o600)
    print(f"{api_type.upper()} API key saved to {CONFIG_FILE}")

def check_api_key(api_type="openai"):
    """
    Check if API key is configured
    """
    api_key = get_api_key(api_type)
    if not api_key:
        print(f"❌ {api_type.upper()} API key not found!")
        print(f"\nTo set up your API key, you can:")
        print(f"1. Set environment variable: export {api_type.upper()}_API_KEY='your-key-here'")
        print(f"2. Or run: python setup_api_key.py --api-type {api_type}")
        return False
    
    print(f"✅ {api_type.upper()} API key found")
    return True
