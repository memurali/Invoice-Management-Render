"""
Utility functions for file handling, logging, and common operations.
"""

import os
import uuid
import logging
import tempfile
import asyncio
from typing import Optional, List, Tuple
from pathlib import Path
from fastapi import UploadFile, HTTPException
import secrets
import string
import hashlib
import re
try:
    import psutil  # For system resource monitoring
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Fallback values for when psutil is not available
    class FallbackPsutil:
        @staticmethod
        def cpu_count(logical=True):
            return 4  # Default to 4 cores
        
        @staticmethod
        def virtual_memory():
            class Memory:
                available = 8 * 1024**3  # Default to 8GB
            return Memory()
        
        @staticmethod
        def cpu_percent(interval=0.1):
            return 50.0  # Default to 50% usage
    
    psutil = FallbackPsutil()

import time
import math

from .config import settings

logger = logging.getLogger(__name__)

# Performance tracking for batch optimization
_batch_performance_history = []
_last_optimization_time = 0
OPTIMIZATION_INTERVAL = 300  # Re-optimize every 5 minutes


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def validate_pdf_file(file: UploadFile) -> bool:
    """
    Validate that the uploaded file is a PDF.
    
    Args:
        file: The uploaded file
        
    Returns:
        bool: True if valid PDF, False otherwise
        
    Raises:
        HTTPException: If file validation fails
    """
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
        )
    
    # Check file extension
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF"
        )
    
    # Check content type
    if file.content_type and not file.content_type.startswith('application/pdf'):
        logger.warning(f"Content type is {file.content_type}, expected application/pdf")
    
    return True


def validate_multiple_pdf_files(files: List[UploadFile]) -> bool:
    """
    Validate multiple uploaded PDF files.
    
    Args:
        files: List of uploaded files
        
    Returns:
        bool: True if all files are valid PDFs
        
    Raises:
        HTTPException: If any file validation fails
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded"
        )
    
    if len(files) > 20:  # Set a reasonable limit
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files allowed per request"
        )
    
    for i, file in enumerate(files):
        try:
            validate_pdf_file(file)
        except HTTPException as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=f"File {i+1} ({file.filename}): {e.detail}"
            )
    
    return True


async def save_uploaded_file(file: UploadFile, request_id: str) -> str:
    """
    Save uploaded file to temporary directory.
    
    Args:
        file: The uploaded file
        request_id: Unique request identifier
        
    Returns:
        str: Path to the saved file
        
    Raises:
        HTTPException: If file saving fails
    """
    try:
        # Reset file position to beginning
        await file.seek(0)
        
        # Create temporary file
        suffix = f"_{request_id}.pdf"
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix="invoice_"
        )
        
        # Write file contents
        contents = await file.read()
        if not contents:
            raise ValueError(f"File {file.filename} is empty or could not be read")
        
        temp_file.write(contents)
        temp_file.flush()
        temp_file.close()
        
        logger.info(f"Saved uploaded file to {temp_file.name} (size: {len(contents)} bytes)")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )


async def save_multiple_uploaded_files(files: List[UploadFile], request_id: str) -> List[Tuple[str, str]]:
    """
    Save multiple uploaded files to temporary directory.
    
    Args:
        files: List of uploaded files
        request_id: Unique request identifier
        
    Returns:
        List[Tuple[str, str]]: List of (filename, temp_path) tuples
        
    Raises:
        HTTPException: If file saving fails
    """
    saved_files = []
    
    try:
        for i, file in enumerate(files):
            # Reset file position to beginning
            await file.seek(0)
            
            # Create unique suffix for each file
            suffix = f"_{request_id}_{i}.pdf"
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                prefix="invoice_"
            )
            
            # Write file contents
            contents = await file.read()
            if not contents:
                raise ValueError(f"File {file.filename} is empty or could not be read")
            
            temp_file.write(contents)
            temp_file.flush()
            temp_file.close()
            
            saved_files.append((file.filename or f"file_{i}.pdf", temp_file.name))
            logger.info(f"Saved file {file.filename} to {temp_file.name} (size: {len(contents)} bytes)")
        
        return saved_files
        
    except Exception as e:
        # Clean up any files that were saved before the error
        for _, temp_path in saved_files:
            cleanup_temp_file(temp_path)
        
        logger.error(f"Failed to save uploaded files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded files: {str(e)}"
        )


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file.
    
    Args:
        file_path: Path to the temporary file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        else:
            logger.warning(f"Temporary file not found for cleanup: {file_path}")
    except Exception as e:
        logger.error(f"Failed to clean up temporary file {file_path}: {e}")


def save_temp_file(file_bytes: bytes, filename: str) -> str:
    """
    Save bytes to a temporary file.
    
    Args:
        file_bytes: File content as bytes
        filename: Original filename for logging/reference
        
    Returns:
        str: Path to the saved temporary file
        
    Raises:
        Exception: If file saving fails
    """
    try:
        # Create temporary file with PDF extension
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
            prefix="firebase_invoice_"
        )
        
        # Write file contents
        temp_file.write(file_bytes)
        temp_file.flush()
        temp_file.close()
        
        logger.info(f"Saved Firebase file {filename} to {temp_file.name} (size: {len(file_bytes)} bytes)")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Failed to save Firebase file {filename}: {e}")
        raise Exception(f"Failed to save Firebase file: {str(e)}")


def cleanup_multiple_temp_files(file_paths: List[str]) -> None:
    """
    Clean up multiple temporary files.
    
    Args:
        file_paths: List of paths to temporary files
    """
    for file_path in file_paths:
        cleanup_temp_file(file_path)


def create_batches(items: List, batch_size: int = 4) -> List[List]:
    """
    Create batches from a list of items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch (default: 4)
        
    Returns:
        List[List]: List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("api.log"),
            logging.StreamHandler()
        ]
    )


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB.
    
    Args:
        file_path: Path to the file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


def parse_firebase_storage_url(url: str) -> dict:
    """
    Parse Firebase Storage URL to extract bucket and path information.
    
    Args:
        url: Firebase Storage URL
        
    Returns:
        dict: Parsed URL information with bucket, path, and token
    """
    import urllib.parse
    
    result = {
        'bucket': None,
        'path': None,
        'token': None,
        'is_firebase_url': False
    }
    
    if not url or 'firebasestorage.googleapis.com' not in url:
        return result
    
    result['is_firebase_url'] = True
    
    try:
        # Parse the URL
        parsed = urllib.parse.urlparse(url)
        
        # Extract bucket from path: /v0/b/bucket-name/o/...
        path_parts = parsed.path.split('/')
        if len(path_parts) >= 4 and path_parts[1] == 'v0' and path_parts[2] == 'b':
            result['bucket'] = path_parts[3]
        
        # Extract file path from after /o/
        o_index = None
        for i, part in enumerate(path_parts):
            if part == 'o':
                o_index = i
                break
        
        if o_index is not None and len(path_parts) > o_index + 1:
            # Join remaining parts and decode
            encoded_path = '/'.join(path_parts[o_index + 1:])
            result['path'] = urllib.parse.unquote(encoded_path)
        
        # Extract token from query parameters
        query_params = urllib.parse.parse_qs(parsed.query)
        if 'token' in query_params:
            result['token'] = query_params['token'][0]
        
    except Exception as e:
        logger.error(f"Failed to parse Firebase Storage URL: {e}")
    
    return result


def find_firebase_file_by_name(bucket, filename: str) -> Optional[str]:
    """
    Find a file in Firebase Storage by searching common folder patterns.
    
    Args:
        bucket: Firebase Storage bucket object
        filename: Filename to search for
        
    Returns:
        str: Found file path, or None if not found
    """
    # Common folder patterns to search
    search_patterns = [
        filename,  # Root level
        f"invoices/{filename}",
        f"uploads/{filename}",
        f"documents/{filename}",
        f"files/{filename}",
        f"pdfs/{filename}",
        f"data/{filename}",
        f"storage/{filename}",
    ]
    
    # Also search by date patterns
    from datetime import datetime
    now = datetime.now()
    date_patterns = [
        f"{now.year}/{filename}",
        f"{now.year}/{now.month:02d}/{filename}",
        f"{now.year}/{now.month:02d}/{now.day:02d}/{filename}",
        f"uploads/{now.year}/{filename}",
        f"documents/{now.year}/{filename}",
        f"invoices/{now.year}/{filename}",
        f"invoices/{now.year}/{now.month:02d}/{filename}",
    ]
    
    search_patterns.extend(date_patterns)
    
    for pattern in search_patterns:
        try:
            blob = bucket.blob(pattern)
            if blob.exists():
                logger.info(f"Found file at path: {pattern}")
                return pattern
        except Exception as e:
            logger.debug(f"Error checking path {pattern}: {e}")
            continue
    
    return None


def list_firebase_files_by_prefix(bucket, prefix: str = "", limit: int = 100) -> List[str]:
    """
    List files in Firebase Storage with optional prefix.
    
    Args:
        bucket: Firebase Storage bucket object
        prefix: Prefix to filter files
        limit: Maximum number of files to return
        
    Returns:
        List[str]: List of file paths
    """
    try:
        blobs = bucket.list_blobs(prefix=prefix, max_results=limit)
        return [blob.name for blob in blobs if blob.name.endswith('.pdf')]
    except Exception as e:
        logger.error(f"Failed to list Firebase files: {e}")
        return [] 


def calculate_optimal_batch_size(
    file_count: int, 
    file_sizes: List[int] = None, 
    target_completion_time: float = 60.0
) -> int:
    """
    Intelligently calculate optimal batch size based on multiple factors.
    
    Args:
        file_count: Total number of files to process
        file_sizes: List of file sizes in bytes (optional)
        target_completion_time: Target completion time in seconds
        
    Returns:
        int: Optimal batch size (minimum 1, maximum 20)
    """
    global _batch_performance_history, _last_optimization_time
    
    try:
        # Base calculations
        avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 2_000_000  # 2MB default
        total_size_mb = (sum(file_sizes) if file_sizes else file_count * avg_file_size) / (1024 * 1024)
        
        # System resource factors
        cpu_count = psutil.cpu_count(logical=True)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory per file estimate (processing overhead)
        memory_per_file_mb = max(50, avg_file_size / (1024 * 1024) * 2)  # ~2x file size + 50MB base
        max_concurrent_by_memory = max(1, int((available_memory_gb * 1024 * 0.7) / memory_per_file_mb))
        
        # Performance history analysis
        recent_performance = _batch_performance_history[-10:] if _batch_performance_history else []
        avg_processing_time = sum(p['time_per_file'] for p in recent_performance) / len(recent_performance) if recent_performance else 3.0
        
        # Smart batch size calculation
        if file_count <= 2:
            # Very small batches - process all at once
            optimal_size = file_count
        elif file_count <= 8:
            # Small batches - consider CPU cores and memory
            optimal_size = min(file_count, cpu_count, max_concurrent_by_memory)
        else:
            # Large batches - optimize for throughput
            # Calculate based on target completion time
            estimated_batches_needed = math.ceil(file_count * avg_processing_time / target_completion_time)
            optimal_size = max(2, min(20, file_count // max(1, estimated_batches_needed)))
            
            # Adjust based on system load
            if cpu_usage > 80:
                optimal_size = max(1, optimal_size // 2)  # Reduce load if system is busy
            elif cpu_usage < 30:
                optimal_size = min(20, optimal_size * 2)  # Increase if system has capacity
            
            # Memory constraint
            optimal_size = min(optimal_size, max_concurrent_by_memory)
        
        # File size considerations
        if avg_file_size > 5_000_000:  # Large files > 5MB
            optimal_size = max(1, optimal_size // 2)
        elif avg_file_size < 500_000:  # Small files < 500KB
            optimal_size = min(20, optimal_size * 2)
        
        # Final constraints
        optimal_size = max(1, min(20, optimal_size, file_count))
        
        logger.info(f"Batch optimization: {file_count} files, {total_size_mb:.1f}MB total, "
                   f"CPU: {cpu_count} cores ({cpu_usage:.1f}% usage), "
                   f"Memory: {available_memory_gb:.1f}GB available, "
                   f"Optimal batch size: {optimal_size}"
                   f"{' (using fallback values)' if not PSUTIL_AVAILABLE else ''}")
        
        return optimal_size
        
    except Exception as e:
        logger.warning(f"Batch size calculation failed, using fallback: {e}")
        # Fallback logic
        if file_count <= 4:
            return file_count
        elif file_count <= 12:
            return 4
        else:
            return 6


def track_batch_performance(batch_size: int, processing_time: float, success_rate: float):
    """
    Track batch processing performance for future optimization.
    
    Args:
        batch_size: Size of the batch that was processed
        processing_time: Total time taken to process the batch
        success_rate: Percentage of successful processing (0.0 to 1.0)
    """
    global _batch_performance_history
    
    time_per_file = processing_time / batch_size if batch_size > 0 else processing_time
    
    performance_record = {
        'batch_size': batch_size,
        'processing_time': processing_time,
        'time_per_file': time_per_file,
        'success_rate': success_rate,
        'timestamp': time.time(),
        'efficiency_score': success_rate / max(0.1, time_per_file)  # Success per second
    }
    
    _batch_performance_history.append(performance_record)
    
    # Keep only recent history (last 50 records)
    if len(_batch_performance_history) > 50:
        _batch_performance_history = _batch_performance_history[-50:]
    
    logger.debug(f"Performance tracked: batch_size={batch_size}, time_per_file={time_per_file:.2f}s, "
                f"success_rate={success_rate:.2f}, efficiency={performance_record['efficiency_score']:.3f}")


def create_intelligent_batches(files: List[Tuple], target_completion_time: float = 60.0) -> List[List[Tuple]]:
    """
    Create batches with intelligent sizing for optimal performance.
    
    Args:
        files: List of (filename, file_path_or_data) tuples
        target_completion_time: Target completion time in seconds
        
    Returns:
        List[List[Tuple]]: List of batches, each containing file tuples
    """
    if not files:
        return []
    
    file_count = len(files)
    
    # Estimate file sizes if available (for local files)
    file_sizes = []
    for filename, file_path in files:
        try:
            if isinstance(file_path, str) and os.path.exists(file_path):
                file_sizes.append(os.path.getsize(file_path))
            else:
                # Estimate based on filename or use average
                file_sizes.append(2_000_000)  # 2MB default
        except:
            file_sizes.append(2_000_000)
    
    # Calculate optimal batch size
    optimal_batch_size = calculate_optimal_batch_size(file_count, file_sizes, target_completion_time)
    
    # Create batches
    batches = []
    for i in range(0, file_count, optimal_batch_size):
        batch = files[i:i + optimal_batch_size]
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} intelligent batches from {file_count} files "
               f"with optimal batch size {optimal_batch_size}")
    
    return batches 