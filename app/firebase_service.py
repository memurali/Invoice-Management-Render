"""
Firebase Storage service for handling file operations.
"""

import logging
import tempfile
import os
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Optional, BinaryIO, List, Dict, Tuple
import urllib.parse
import requests
import firebase_admin
from firebase_admin import storage
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import settings

logger = logging.getLogger(__name__)


class FirebaseStorageService:
    """Service for handling Firebase Storage operations."""
    
    def __init__(self):
        self.bucket_name = settings.FIREBASE_STORAGE_BUCKET
        self.max_concurrent_downloads = 10  # Maximum concurrent downloads
        self.download_timeout = 120  # Timeout for downloads in seconds
        
    def get_bucket(self):
        """Get Firebase Storage bucket."""
        try:
            if not firebase_admin._apps:
                raise ValueError("Firebase not initialized")
            return storage.bucket(self.bucket_name)
        except Exception as e:
            logger.error(f"Failed to get Firebase bucket: {e}")
            raise
    
    async def download_file_from_url_async(self, storage_url: str, session: aiohttp.ClientSession) -> Tuple[str, str]:
        """
        Download a file from Firebase Storage URL asynchronously.
        
        Args:
            storage_url: Firebase Storage download URL
            session: aiohttp session for async requests
            
        Returns:
            Tuple[str, str]: (original_url, temp_file_path)
        """
        try:
            logger.info(f"Downloading file asynchronously from: {storage_url[:100]}...")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix="firebase_invoice_"
            )
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Download the file
            timeout = aiohttp.ClientTimeout(total=self.download_timeout)
            async with session.get(storage_url, timeout=timeout) as response:
                response.raise_for_status()
                
                with open(temp_file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
            
            logger.info(f"Downloaded file to: {temp_file_path}")
            return (storage_url, temp_file_path)
            
        except Exception as e:
            logger.error(f"Failed to download file from Firebase Storage: {e}")
            # Clean up temp file if it was created
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            raise Exception(f"Firebase download failed: {str(e)}")
    
    async def download_multiple_files_async(self, storage_urls: List[str]) -> Dict[str, str]:
        """
        Download multiple files from Firebase Storage URLs in parallel.
        
        Args:
            storage_urls: List of Firebase Storage download URLs
            
        Returns:
            Dict[str, str]: Mapping of original URLs to temp file paths
        """
        if not storage_urls:
            return {}
            
        logger.info(f"Starting parallel download of {len(storage_urls)} files")
        start_time = time.time()
        
        # Limit concurrent downloads
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        async def download_with_semaphore(url: str, session: aiohttp.ClientSession):
            async with semaphore:
                return await self.download_file_from_url_async(url, session)
        
        try:
            # Create aiohttp session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent_downloads,
                limit_per_host=self.max_concurrent_downloads
            )
            
            async with aiohttp.ClientSession(connector=connector) as session:
                # Create tasks for all downloads
                tasks = [download_with_semaphore(url, session) for url in storage_urls]
                
                # Execute downloads concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                url_to_path = {}
                successful_downloads = 0
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to download {storage_urls[i]}: {result}")
                        # Don't add to results, but don't fail the entire batch
                    else:
                        url, temp_path = result
                        url_to_path[url] = temp_path
                        successful_downloads += 1
                
                download_time = time.time() - start_time
                logger.info(f"Parallel download completed: {successful_downloads}/{len(storage_urls)} files in {download_time:.2f}s")
                
                return url_to_path
                
        except Exception as e:
            logger.error(f"Failed to download multiple files: {e}")
            raise Exception(f"Batch download failed: {str(e)}")
    
    def download_file_from_url(self, storage_url: str) -> str:
        """
        Download a file from Firebase Storage URL to a temporary file.
        
        Args:
            storage_url: Firebase Storage download URL
            
        Returns:
            str: Path to the downloaded temporary file
        """
        try:
            logger.info(f"Downloading file from Firebase Storage URL: {storage_url[:100]}...")
            
            # Download the file with improved error handling
            response = requests.get(storage_url, stream=True, timeout=self.download_timeout)
            response.raise_for_status()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix="firebase_invoice_"
            )
            
            # Write content to temporary file with progress tracking
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress for large files
                    if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Every MB
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Download progress: {progress:.1f}%")
            
            temp_file.close()
            
            logger.info(f"Downloaded file to: {temp_file.name} ({downloaded} bytes)")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to download file from Firebase Storage: {e}")
            raise Exception(f"Firebase download failed: {str(e)}")
    
    def download_file_from_path(self, storage_path: str) -> str:
        """
        Download a file from Firebase Storage path to a temporary file.
        
        Args:
            storage_path: Firebase Storage path (e.g., 'invoices/file.pdf')
            
        Returns:
            str: Path to the downloaded temporary file
        """
        try:
            logger.info(f"Downloading file from Firebase Storage path: {storage_path}")
            
            bucket = self.get_bucket()
            blob = bucket.blob(storage_path)
            
            if not blob.exists():
                raise FileNotFoundError(f"File not found in Firebase Storage: {storage_path}")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                prefix="firebase_invoice_"
            )
            
            # Download to temporary file
            blob.download_to_filename(temp_file.name)
            temp_file.close()
            
            logger.info(f"Downloaded file from path {storage_path} to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to download file from Firebase Storage path: {e}")
            raise Exception(f"Firebase path download failed: {str(e)}")
    
    def download_file_content(self, storage_path: str) -> bytes:
        """
        Download file content from Firebase Storage path as bytes.
        
        Args:
            storage_path: Firebase Storage path (e.g., 'invoices/file.pdf')
            
        Returns:
            bytes: File content as bytes
        """
        try:
            logger.info(f"Downloading file content from Firebase Storage path: {storage_path}")
            
            bucket = self.get_bucket()
            blob = bucket.blob(storage_path)
            
            if not blob.exists():
                raise FileNotFoundError(f"File not found in Firebase Storage: {storage_path}")
            
            # Download content as bytes
            content = blob.download_as_bytes()
            
            logger.info(f"Downloaded {len(content)} bytes from path: {storage_path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to download file content from Firebase Storage path: {e}")
            raise Exception(f"Firebase content download failed: {str(e)}")
    
    def upload_file(self, file_path: str, storage_path: str) -> str:
        """
        Upload a file to Firebase Storage.
        
        Args:
            file_path: Local file path to upload
            storage_path: Firebase Storage path where to store the file
            
        Returns:
            str: Firebase Storage download URL
        """
        try:
            logger.info(f"Uploading file to Firebase Storage: {storage_path}")
            
            bucket = self.get_bucket()
            blob = bucket.blob(storage_path)
            
            # Upload file
            blob.upload_from_filename(file_path)
            
            # Make the blob publicly readable (optional, depending on your needs)
            # blob.make_public()
            
            # Get download URL
            download_url = blob.generate_signed_url(
                expiration=datetime.utcnow() + timedelta(hours=1)
            )
            
            logger.info(f"File uploaded successfully: {storage_path}")
            return download_url
            
        except Exception as e:
            logger.error(f"Failed to upload file to Firebase Storage: {e}")
            raise Exception(f"Firebase upload failed: {str(e)}")
    
    def list_files(self, prefix: str = "", limit: int = 50) -> list:
        """
        List files in Firebase Storage.
        
        Args:
            prefix: Prefix to filter files (e.g., 'invoices/')
            limit: Maximum number of files to return
            
        Returns:
            list: List of file information dictionaries
        """
        try:
            logger.info(f"Listing Firebase Storage files with prefix: {prefix}")
            
            bucket = self.get_bucket()
            blobs = bucket.list_blobs(prefix=prefix, max_results=limit)
            
            files = []
            for blob in blobs:
                files.append({
                    "name": blob.name,
                    "size": blob.size,
                    "created": blob.time_created.isoformat() if blob.time_created else None,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "content_type": blob.content_type
                })
            
            logger.info(f"Found {len(files)} files")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list Firebase Storage files: {e}")
            raise Exception(f"Firebase list failed: {str(e)}")
    
    def delete_file(self, storage_path: str) -> bool:
        """
        Delete a file from Firebase Storage.
        
        Args:
            storage_path: Firebase Storage path of file to delete
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Deleting file from Firebase Storage: {storage_path}")
            
            bucket = self.get_bucket()
            blob = bucket.blob(storage_path)
            
            if blob.exists():
                blob.delete()
                logger.info(f"File deleted successfully: {storage_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {storage_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file from Firebase Storage: {e}")
            raise Exception(f"Firebase delete failed: {str(e)}")
    
    def parse_storage_url(self, storage_url: str) -> str:
        """
        Parse Firebase Storage URL to extract the file path.
        
        Args:
            storage_url: Firebase Storage download URL
            
        Returns:
            str: Extracted file path
        """
        try:
            if 'firebasestorage.googleapis.com' in storage_url:
                # Format: https://firebasestorage.googleapis.com/v0/b/bucket/o/path%2Fto%2Ffile.pdf
                parts = storage_url.split('/o/')
                if len(parts) > 1:
                    # Decode URL-encoded path
                    path = urllib.parse.unquote(parts[1].split('?')[0])
                    return path
            return storage_url
        except Exception as e:
            logger.error(f"Failed to parse storage URL: {e}")
            return storage_url 