"""
Firebase PDF Upload Service with Concurrent Processing and Batch Optimization.

This service provides optimized PDF upload capabilities to Firebase Storage with:
- Single and batch PDF uploads
- Concurrent processing with intelligent batch sizing
- Runtime optimization based on file sizes and system resources
- Comprehensive error handling and progress tracking
"""

import asyncio
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from fastapi import UploadFile
import logging

from .firebase_service import FirebaseStorageService
from .utils import cleanup_temp_file

logger = logging.getLogger(__name__)


class UploadResult:
    """Result of a single file upload operation."""
    def __init__(self, filename: str, success: bool, firebase_path: str = None, 
                 error: str = None, file_size: int = 0, upload_time: float = 0.0):
        self.filename = filename
        self.success = success
        self.firebase_path = firebase_path
        self.error = error
        self.file_size = file_size
        self.upload_time = upload_time


class BatchUploadResult:
    """Result of a batch upload operation."""
    def __init__(self):
        self.total_files = 0
        self.successful_uploads = 0
        self.failed_uploads = 0
        self.upload_results: List[UploadResult] = []
        self.total_upload_time = 0.0
        self.total_file_size = 0
        self.batch_size_used = 0
        self.concurrent_workers = 0


class OptimizedFirebaseUploader:
    """
    Optimized Firebase uploader with concurrent processing and intelligent batch sizing.
    
    Features:
    - Runtime batch size optimization
    - Concurrent uploads with resource management
    - File size-based optimization
    - Progress tracking and comprehensive error handling
    """
    
    def __init__(self):
        self.firebase_service = FirebaseStorageService()
        self.base_upload_path = "invoices"
        
    def _calculate_optimal_batch_size(self, file_sizes: List[int], max_workers: int = None) -> Tuple[int, int]:
        """
        Calculate optimal batch size and worker count based on file sizes and system resources.
        
        Args:
            file_sizes: List of file sizes in bytes
            max_workers: Maximum number of workers (optional)
            
        Returns:
            Tuple of (batch_size, worker_count)
        """
        total_files = len(file_sizes)
        total_size_mb = sum(file_sizes) / (1024 * 1024)
        avg_file_size_mb = total_size_mb / total_files if total_files > 0 else 0
        
        # Base calculations
        if max_workers is None:
            # Dynamic worker calculation based on file characteristics
            if avg_file_size_mb < 1:  # Small files
                max_workers = min(8, total_files)
            elif avg_file_size_mb < 5:  # Medium files  
                max_workers = min(6, total_files)
            else:  # Large files
                max_workers = min(4, total_files)
        
        # Batch size optimization
        if total_files <= 5:
            batch_size = total_files
        elif total_size_mb < 50:  # Small total size
            batch_size = min(10, total_files)
        elif total_size_mb < 200:  # Medium total size
            batch_size = min(8, total_files)
        else:  # Large total size
            batch_size = min(6, total_files)
            
        # Ensure sensible limits
        max_workers = max(1, min(max_workers, total_files))
        batch_size = max(1, min(batch_size, total_files))
        
        logger.info(f"Optimized batch configuration: {total_files} files, "
                   f"{total_size_mb:.2f}MB total, batch_size={batch_size}, workers={max_workers}")
        
        return batch_size, max_workers
    
    def _generate_firebase_path(self, filename: str) -> str:
        """Generate organized Firebase storage path."""
        from datetime import datetime
        
        # Create organized path: invoices/YYYY/MM/DD/filename
        now = datetime.now()
        path = f"{self.base_upload_path}/{now.year:04d}/{now.month:02d}/{now.day:02d}/{filename}"
        return path
    
    async def upload_single_file(self, file: UploadFile, custom_path: str = None) -> UploadResult:
        """
        Upload a single PDF file to Firebase Storage.
        
        Args:
            file: FastAPI UploadFile object
            custom_path: Optional custom Firebase path
            
        Returns:
            UploadResult with upload details
        """
        start_time = time.time()
        temp_path = None
        
        try:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                return UploadResult(
                    filename=file.filename,
                    success=False,
                    error="Only PDF files are allowed"
                )
            
            # Read file content
            file_content = await file.read()
            file_size = len(file_content)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            # Generate Firebase path
            firebase_path = custom_path or self._generate_firebase_path(file.filename)
            
            # Upload to Firebase
            logger.info(f"Uploading {file.filename} ({file_size} bytes) to {firebase_path}")
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                await loop.run_in_executor(
                    executor,
                    self.firebase_service.upload_file,
                    temp_path,
                    firebase_path
                )
            
            upload_time = time.time() - start_time
            logger.info(f"Successfully uploaded {file.filename} in {upload_time:.2f}s")
            
            return UploadResult(
                filename=file.filename,
                success=True,
                firebase_path=firebase_path,
                file_size=file_size,
                upload_time=upload_time
            )
            
        except Exception as e:
            upload_time = time.time() - start_time
            logger.error(f"Failed to upload {file.filename}: {e}")
            
            return UploadResult(
                filename=file.filename,
                success=False,
                error=str(e),
                upload_time=upload_time
            )
            
        finally:
            # Cleanup temporary file
            if temp_path:
                cleanup_temp_file(temp_path)
    
    async def upload_multiple_files(self, files: List[UploadFile], 
                                  batch_size: int = None, max_workers: int = None) -> BatchUploadResult:
        """
        Upload multiple PDF files to Firebase Storage with optimized concurrent processing.
        
        Args:
            files: List of FastAPI UploadFile objects
            batch_size: Optional batch size (auto-calculated if not provided)
            max_workers: Optional max workers (auto-calculated if not provided)
            
        Returns:
            BatchUploadResult with comprehensive upload details
        """
        start_time = time.time()
        result = BatchUploadResult()
        result.total_files = len(files)
        
        if not files:
            logger.warning("No files provided for upload")
            return result
        
        try:
            # Pre-read file sizes for optimization
            logger.info(f"Analyzing {len(files)} files for batch optimization...")
            file_data = []
            
            for file in files:
                content = await file.read()
                file_data.append({
                    'file': file,
                    'content': content,
                    'size': len(content)
                })
                # Reset file position for potential re-use
                await file.seek(0)
            
            file_sizes = [data['size'] for data in file_data]
            result.total_file_size = sum(file_sizes)
            
            # Calculate optimal batch configuration
            if batch_size is None or max_workers is None:
                calc_batch_size, calc_workers = self._calculate_optimal_batch_size(file_sizes, max_workers)
                batch_size = batch_size or calc_batch_size
                max_workers = max_workers or calc_workers
            
            result.batch_size_used = batch_size
            result.concurrent_workers = max_workers
            
            logger.info(f"Starting batch upload: {len(files)} files, "
                       f"batch_size={batch_size}, workers={max_workers}")
            
            # Process files in batches
            semaphore = asyncio.Semaphore(max_workers)
            
            async def upload_single_with_semaphore(file_info: dict) -> UploadResult:
                async with semaphore:
                    # Create temporary UploadFile-like object from pre-read data
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    temp_file.write(file_info['content'])
                    temp_file.close()
                    
                    try:
                        # Create mock UploadFile for compatibility
                        mock_file = type('MockFile', (), {
                            'filename': file_info['file'].filename,
                            'read': lambda: asyncio.create_task(asyncio.coroutine(lambda: file_info['content'])())
                        })()
                        
                        # Generate Firebase path
                        firebase_path = self._generate_firebase_path(file_info['file'].filename)
                        
                        # Upload directly using Firebase service
                        upload_start = time.time()
                        self.firebase_service.upload_file(temp_file.name, firebase_path)
                        upload_time = time.time() - upload_start
                        
                        return UploadResult(
                            filename=file_info['file'].filename,
                            success=True,
                            firebase_path=firebase_path,
                            file_size=file_info['size'],
                            upload_time=upload_time
                        )
                        
                    except Exception as e:
                        return UploadResult(
                            filename=file_info['file'].filename,
                            success=False,
                            error=str(e),
                            file_size=file_info['size']
                        )
                    finally:
                        cleanup_temp_file(temp_file.name)
            
            # Execute batch uploads
            tasks = [upload_single_with_semaphore(data) for data in file_data]
            upload_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for upload_result in upload_results:
                if isinstance(upload_result, Exception):
                    logger.error(f"Upload task failed with exception: {upload_result}")
                    result.failed_uploads += 1
                    result.upload_results.append(UploadResult(
                        filename="unknown",
                        success=False,
                        error=str(upload_result)
                    ))
                else:
                    result.upload_results.append(upload_result)
                    if upload_result.success:
                        result.successful_uploads += 1
                    else:
                        result.failed_uploads += 1
            
            result.total_upload_time = time.time() - start_time
            
            logger.info(f"Batch upload completed: {result.successful_uploads}/{result.total_files} successful "
                       f"in {result.total_upload_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            result.total_upload_time = time.time() - start_time
            result.failed_uploads = result.total_files
            
            return result 