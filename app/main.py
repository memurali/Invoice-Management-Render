"""
FastAPI Invoice Parser - Professional Main Application

Entry point for the FastAPI invoice parsing microservice with professional
configuration management, comprehensive error handling, and production-ready features.

This application provides:
- Professional API documentation and versioning
- Comprehensive logging and monitoring
- Proper error handling and validation
- Firebase integration with validation
- CORS configuration for production use
"""

import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn
import firebase_admin
from firebase_admin import credentials, storage

# Load environment variables first
load_dotenv()

from .config import settings, ConfigurationError
from .api import router

# Configure professional logging
def setup_logging():
    """Setup professional logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO if not settings.DEBUG_MODE else logging.DEBUG,
        format=log_format,
        handlers=[
            logging.FileHandler("invoice_parser_api.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set specific log levels for different modules
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration completed")
    return logger

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Professional application lifespan management.
    
    Handles startup and shutdown events with proper resource initialization
    and cleanup.
    """
    # Startup
    logger.info("Invoice Parser API starting up...")
    
    try:
        # Initialize Firebase Admin SDK
        initialize_firebase()
        
        # Validate all connections
        validate_service_connections()
        
        logger.info("Invoice Parser API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Invoice Parser API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Invoice Parser API shutting down...")
    
    try:
        # Clean up Firebase connection
        if firebase_admin._apps:
            firebase_admin.delete_app(firebase_admin.get_app())
            logger.info("Firebase Admin SDK cleaned up")
        
        # Clean up temporary configuration files
        settings.cleanup()
        
        logger.info("Invoice Parser API shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Initialize FastAPI app with professional configuration
app = FastAPI(
    title="Invoice Parser API",
    description="""
    **Professional AI-Powered Invoice Processing Service**
    
    This API provides comprehensive PDF invoice parsing capabilities using:
    
    - 🔍 **OpenAI Vision API** for advanced OCR and text extraction
    - 🏗️ **Structured Output** for reliable, type-safe data extraction
    - ☁️ **Firebase Storage** integration for scalable file management
    - ⚡ **High-Performance** batch processing with optimization
    - 🛡️ **Professional** error handling and validation
    
    ## Features
    
    - **Single Invoice Processing**: Process individual invoices from Firebase Storage
    - **Batch Processing**: Process multiple invoices with optimized performance
    - **Streaming Processing**: Real-time processing updates for large batches
    - **Health Monitoring**: Comprehensive health checks and monitoring
    
    ## Authentication
    
    This API requires proper Firebase and OpenAI API credentials configured via environment variables.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Invoice Parser API Support",
        "url": "https://github.com/your-org/invoice-parser",
        "email": "support@yourcompany.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Professional CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:8080",  # Vue development server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        # Vercel deployment domains
        "https://*.vercel.app",
        "https://vercel.app",
        # Add your specific Vercel domain here:
        # "https://your-app-name.vercel.app",
        # Allow all origins for development (remove in production)
        "*"
    ],
    allow_credentials=False,  # Set to False when using "*" for origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With", "User-Agent"],
    expose_headers=["X-Request-ID", "X-Processing-Time"]
)


def initialize_firebase():
    """
    Initialize Firebase Admin SDK with proper error handling.
    
    For testing purposes, Firebase initialization is optional.
    """
    try:
        if firebase_admin._apps:
            logger.info("Firebase Admin SDK already initialized")
            return
        
        # Check if Firebase credentials are available
        if not settings.FIREBASE_SERVICE_ACCOUNT_KEY_FILE or not settings.FIREBASE_STORAGE_BUCKET:
            logger.warning("Firebase credentials not configured - Firebase features will be disabled")
            logger.warning("To enable Firebase features, set FIREBASE_SERVICE_ACCOUNT_KEY and FIREBASE_STORAGE_BUCKET")
            return
        
        # Initialize with service account credentials from temporary file
        cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT_KEY_FILE)
        firebase_admin.initialize_app(cred, {
            'storageBucket': settings.FIREBASE_STORAGE_BUCKET
        })
        
        logger.info(f"Firebase Admin SDK initialized successfully (bucket: {settings.FIREBASE_STORAGE_BUCKET})")
        
    except Exception as e:
        logger.warning(f"Failed to initialize Firebase Admin SDK: {e}")
        logger.warning("Firebase features will be disabled")
        # Don't raise exception for testing purposes


def validate_service_connections():
    """
    Validate all external service connections.
    
    Raises:
        ConfigurationError: If any service validation fails
    """
    logger.info("Validating service connections...")
    
    # Validate OpenAI connection
    if not settings.validate_openai_connection():
        raise ConfigurationError("OpenAI API connection validation failed")
    
    # Validate Firebase connection
    if not settings.validate_firebase_connection():
        raise ConfigurationError("Firebase connection validation failed")
    
    logger.info("All service connections validated successfully")


# Add professional middleware for request tracking
@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    """Add request tracking and performance monitoring."""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    # Add request ID to logs
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = f"{process_time:.4f}s"
    
    logger.info(f"[{request_id}] Completed in {process_time:.4f}s - Status: {response.status_code}")
    
    return response


# Professional error handlers
@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    """Handle configuration errors professionally."""
    logger.error(f"Configuration error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Configuration Error",
            "message": "Service configuration is invalid. Please check server logs.",
            "type": "configuration_error"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "message": exc.detail,
            "type": "http_error"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors professionally."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please contact support.",
            "type": "internal_error"
        }
    )


# Include API routes
app.include_router(router)

# Serve the frontend HTML file
@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the main frontend HTML file."""
    from fastapi.responses import FileResponse
    import os
    
    # Get the path to the HTML file in the project root
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "invoice_uploader.html")
    
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Frontend file not found")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns:
        dict: Health status of all services
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "services": {
                "openai": settings.validate_openai_connection(),
                "firebase": settings.validate_firebase_connection()
            },
            "configuration": {
                "model": settings.OPENAI_MODEL,
                "dpi": settings.DPI,
                "format": settings.FORMAT,
                "debug_mode": settings.DEBUG_MODE
            }
        }
        
        # Check if any service is unhealthy
        if not all(health_status["services"].values()):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


# API info endpoint (moved to /info since / serves frontend)
@app.get("/info", tags=["Info"])
async def api_info():
    """
    API information endpoint.
    
    Returns:
        dict: API information and links
    """
    return {
        "service": "Invoice Parser API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "api_prefix": "/api/v2",
        "frontend": "/"
    }


# Production server runner
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8003,
        reload=settings.DEBUG_MODE,
        log_level="info" if not settings.DEBUG_MODE else "debug",
        access_log=True
    ) 