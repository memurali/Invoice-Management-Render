#!/usr/bin/env python3
"""
ULTRA-FAST Invoice Processing API Server

This script starts the FastAPI server with ULTRA-FAST optimizations:
- 150 DPI for faster image processing
- JPEG format for smaller files
- 8 concurrent threads
- 45-second API timeout
- Maximum performance settings

Usage:
    python start_fastapi.py
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

def main():
    """Start the ULTRA-FAST invoice processing server."""
    print("🚀 Starting ULTRA-FAST Invoice Processing API Server")
    print("=" * 60)
    
    # Verify environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "FIREBASE_SERVICE_ACCOUNT_KEY",
        "FIREBASE_STORAGE_BUCKET"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing environment variables for full functionality:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 To set these variables:")
        print("   1. Create a .env file in the project root")
        print("   2. Add the following variables:")
        print("      OPENAI_API_KEY=your_openai_api_key_here")
        print("      FIREBASE_SERVICE_ACCOUNT_KEY=your_firebase_json_string")
        print("      FIREBASE_STORAGE_BUCKET=your_bucket_name")
        print("\n🚀 Starting server in TEST MODE (some features may not work)...")
        print("   - API endpoints will be available")
        print("   - Health checks will work")
        print("   - Configuration will be displayed")
        print("   - Invoice processing will require valid credentials")
    
    print("\n🔧 ULTRA-AGGRESSIVE Configuration (20s target):")
    print("   - DPI: 80 (ultra-low for maximum speed)")
    print("   - Format: JPEG (smallest files)")
    print("   - Thread Count: 20 (maximum concurrency)")
    print("   - API Timeout: 25 seconds")
    print("   - Target: 20 seconds per invoice")
    print("   - Unit field: Removed from commodity details")
    
    # Start the server with optimized settings
    print("\n🌐 Starting server on http://127.0.0.1:8003")
    print("   - API Documentation: http://127.0.0.1:8003/docs")
    print("   - Health Check: http://127.0.0.1:8003/api/v2/health")
    print("   - Configuration: http://127.0.0.1:8003/api/v2/config")
    print("   - Frontend: http://127.0.0.1:8003/")
    
    print("\n" + "=" * 60)
    print("🚀 ULTRA-FAST SERVER STARTING...")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8003,
            reload=False,  # Disable reload for production performance
            workers=1,     # Single worker for maximum stability
            log_level="info",
            access_log=True,
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 