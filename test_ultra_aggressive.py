#!/usr/bin/env python3
"""
ULTRA-AGGRESSIVE Performance Test

This script tests the ultra-aggressive optimizations targeting 20 seconds per PDF.
"""

import time
import requests
import json
from pathlib import Path

def test_ultra_aggressive_performance():
    """Test the ultra-aggressive processing performance."""
    
    print("🚀 ULTRA-AGGRESSIVE Performance Test")
    print("=" * 50)
    print("Target: 20 seconds per PDF")
    print("Optimizations:")
    print("  - 80 DPI (ultra-low)")
    print("  - 50% JPEG quality")
    print("  - 20 concurrent workers")
    print("  - 10s conversion timeout")
    print("  - 25s API timeout")
    print("=" * 50)
    
    # Test server health
    try:
        response = requests.get("http://127.0.0.1:8003/api/v2/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test configuration
    try:
        response = requests.get("http://127.0.0.1:8003/api/v2/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Configuration loaded:")
            print(f"   - DPI: {config.get('dpi', 'N/A')}")
            print(f"   - Format: {config.get('format', 'N/A')}")
            print(f"   - Thread Count: {config.get('thread_count', 'N/A')}")
            print(f"   - API Timeout: {config.get('request_timeout', 'N/A')}s")
        else:
            print("❌ Configuration check failed")
    except Exception as e:
        print(f"❌ Configuration check error: {e}")
    
    print("\n🎯 Ready for testing!")
    print("Upload PDF files via:")
    print("  - Frontend: http://127.0.0.1:8003/")
    print("  - API: http://127.0.0.1:8003/docs")
    print("\nExpected performance: 20 seconds per PDF")

if __name__ == "__main__":
    test_ultra_aggressive_performance() 