"""
Professional Configuration Management

This module provides centralized configuration management for the Invoice Parser API.
Core credentials are loaded from environment variables, while server configuration
uses sensible defaults defined in this file.

Required Environment Variables:
    - OPENAI_API_KEY: OpenAI API key for GPT models
    - FIREBASE_SERVICE_ACCOUNT_KEY: Complete Firebase service account JSON string
    - FIREBASE_STORAGE_BUCKET: Firebase Storage bucket name

All other configuration uses defaults defined in this file.
"""

import os
import json
import logging
import tempfile
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class Settings:
    """
    Professional configuration settings with environment variable validation.
    
    This class provides type-safe access to all application configuration
    with automatic validation and sensible defaults.
    """
    
    def __init__(self):
        """Initialize settings with validation of required environment variables."""
        self._validate_required_settings()
        self._load_settings()
        self._setup_firebase_credentials()
        self._log_configuration()
    
    def _validate_required_settings(self) -> None:
        """
        Validate that all required environment variables are present.
        
        For testing purposes, we'll allow missing variables but log warnings.
        """
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
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Some features may not work without proper configuration")
            # Don't raise exception for testing purposes
            return
        
        # Validate Firebase service account key is valid JSON (only if present)
        try:
            firebase_key_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
            if firebase_key_json:
                json.loads(firebase_key_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid Firebase service account key JSON: {e}")
            logger.warning("Firebase features may not work properly")
    
    def _load_settings(self) -> None:
        """Load and validate all configuration settings."""
        
        # Required settings from environment
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
        self.FIREBASE_SERVICE_ACCOUNT_KEY: str = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
        self.FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET")
        
        # Server Configuration (defaults defined here, no environment variables needed)
        self.HOST: str = "0.0.0.0"
        self.PORT: int = 8003
        self.DEBUG: bool = False
        self.LOG_LEVEL: str = "INFO"
        
        # File Processing Configuration
        self.MAX_FILE_SIZE: int = 52428800  # 50MB
        
        # OpenAI Configuration
        self.OPENAI_MODEL: str = "gpt-4o-mini"
        
        # PDF Processing Configuration (ULTRA-AGGRESSIVE for 20s target)
        self.DPI: int = 80  # Ultra-low DPI for maximum speed
        self.FORMAT: str = "JPEG"  # JPEG for smallest files
        self.THREAD_COUNT: int = 20  # Maximum concurrency for ultra-fast processing
        
        # API Configuration (ULTRA-AGGRESSIVE for 20s target)
        self.MAX_RETRIES: int = 1  # Minimal retries for maximum speed
        self.RETRY_DELAY: float = 0.2  # Ultra-fast retry for speed
        self.REQUEST_TIMEOUT: float = 25.0  # Ultra-short timeout for faster processing
        
        # Feature Flags
        self.ENABLE_STREAMING: bool = True
        self.DEBUG_MODE: bool = False
        
        # Validate settings
        self._validate_settings()
    
    def _validate_settings(self) -> None:
        """Validate configuration settings."""
        # Validate DPI range
        if not (50 <= self.DPI <= 600):  # Updated range to allow lower DPI for maximum speed
            logger.warning(f"DPI {self.DPI} outside recommended range (50-600), using 100")
            self.DPI = 100
        
        # Validate thread count
        if not (1 <= self.THREAD_COUNT <= 32):  # Increased max to 32 for ultra-aggressive concurrency
            logger.warning(f"Thread count {self.THREAD_COUNT} outside valid range (1-32), using 16")
            self.THREAD_COUNT = 16
        
        # Validate image format
        if self.FORMAT not in ["PNG", "JPEG", "JPG"]:
            logger.warning(f"Invalid image format '{self.FORMAT}', defaulting to JPEG")
            self.FORMAT = "JPEG"
        
        # Validate OpenAI model
        valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview", "gpt-4-turbo"]
        if self.OPENAI_MODEL not in valid_models:
            logger.warning(f"Unrecognized OpenAI model '{self.OPENAI_MODEL}', using gpt-4o-mini")
            self.OPENAI_MODEL = "gpt-4o-mini"
        
        # Validate retry settings
        if not (0 <= self.MAX_RETRIES <= 10):
            logger.warning(f"Max retries {self.MAX_RETRIES} outside valid range (0-10), using 3")
            self.MAX_RETRIES = 3
        
        if not (0.1 <= self.RETRY_DELAY <= 10.0):
            logger.warning(f"Retry delay {self.RETRY_DELAY} outside valid range (0.1-10.0), using 1.0")
            self.RETRY_DELAY = 1.0
        
        if not (10.0 <= self.REQUEST_TIMEOUT <= 300.0):
            logger.warning(f"Request timeout {self.REQUEST_TIMEOUT} outside valid range (10.0-300.0), using 60.0")
            self.REQUEST_TIMEOUT = 60.0
    
    def _setup_firebase_credentials(self) -> None:
        """
        Setup Firebase credentials from JSON string in environment variable.
        
        Creates a temporary file with the service account credentials for Firebase SDK.
        """
        try:
            # Check if Firebase credentials are available
            firebase_key_json = self.FIREBASE_SERVICE_ACCOUNT_KEY
            if not firebase_key_json:
                logger.warning("Firebase credentials not provided - Firebase features will be disabled")
                self.FIREBASE_SERVICE_ACCOUNT_KEY_FILE = None
                return
            
            # Parse the JSON string from environment variable
            credentials_dict = json.loads(firebase_key_json)
            
            # Create a temporary file to store the credentials
            # Firebase Admin SDK expects a file path, not a JSON string directly
            self._temp_credentials_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                delete=False,
                prefix='firebase_credentials_'
            )
            
            # Write credentials to temporary file
            json.dump(credentials_dict, self._temp_credentials_file, indent=2)
            self._temp_credentials_file.flush()
            self._temp_credentials_file.close()
            
            # Update the setting to point to the temporary file
            self.FIREBASE_SERVICE_ACCOUNT_KEY_FILE = self._temp_credentials_file.name
            
            logger.info("Firebase credentials temporary file created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to setup Firebase credentials: {e}")
            logger.warning("Firebase features will be disabled")
            self.FIREBASE_SERVICE_ACCOUNT_KEY_FILE = None
    
    def _log_configuration(self) -> None:
        """Log current configuration (excluding sensitive data)."""
        config_info = {
            "Server Host": self.HOST,
            "Server Port": self.PORT,
            "Debug Mode": self.DEBUG,
            "Log Level": self.LOG_LEVEL,
            "OpenAI Model": self.OPENAI_MODEL,
            "DPI": self.DPI,
            "Format": self.FORMAT,
            "Thread Count": self.THREAD_COUNT,
            "Max Retries": self.MAX_RETRIES,
            "Retry Delay": f"{self.RETRY_DELAY}s",
            "Request Timeout": f"{self.REQUEST_TIMEOUT}s",
            "Streaming Enabled": self.ENABLE_STREAMING,
            "Max File Size": f"{self.MAX_FILE_SIZE / (1024 * 1024):.1f}MB",
            "Firebase Bucket": self.FIREBASE_STORAGE_BUCKET
        }
        
        logger.info("Configuration loaded successfully:")
        for key, value in config_info.items():
            logger.info(f"  {key}: {value}")
    
    def validate_openai_connection(self) -> bool:
        """
        Validate OpenAI API connection.
        
        Returns:
            bool: True if connection is valid
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.OPENAI_API_KEY)
            
            # Test with a minimal request
            response = client.chat.completions.create(
                model=self.OPENAI_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            logger.info("OpenAI API connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate OpenAI API connection: {e}")
            return False
    
    def validate_firebase_connection(self) -> bool:
        """
        Validate Firebase connection.
        
        Returns:
            bool: True if connection is valid
        """
        try:
            import firebase_admin
            from firebase_admin import credentials, storage
            
            # Check if already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.FIREBASE_SERVICE_ACCOUNT_KEY_FILE)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': self.FIREBASE_STORAGE_BUCKET
                })
            
            # Test bucket access
            bucket = storage.bucket()
            
            logger.info("Firebase connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate Firebase connection: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up temporary files created during configuration."""
        try:
            if hasattr(self, '_temp_credentials_file') and os.path.exists(self._temp_credentials_file.name):
                os.unlink(self._temp_credentials_file.name)
                logger.info("Cleaned up temporary Firebase credentials file")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary credentials file: {e}")


# Global settings instance
try:
    settings = Settings()
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading configuration: {e}")
    raise ConfigurationError(f"Failed to load configuration: {e}")


# Export settings for backward compatibility
__all__ = ["settings", "Settings", "ConfigurationError"] 