# 🚀 ULTRA-AGGRESSIVE Invoice Parser API Documentation

## Overview

The **ULTRA-AGGRESSIVE Invoice Parser API** is a high-performance AI-powered service that extracts structured data from PDF invoices using OpenAI's Vision API and Firebase Storage integration. This API provides lightning-fast invoice processing capabilities with **20-second target per PDF** and maximum parallel processing.

**Base URL**: `http://127.0.0.1:8003`  
**API Version**: 2.0.0  
**Documentation**: `/docs` (Swagger UI) or `/redoc` (ReDoc)

## 🎯 ULTRA-AGGRESSIVE PERFORMANCE FEATURES

- **⚡ 20-Second Target**: Ultra-optimized processing targeting 20 seconds per PDF
- **🔄 Maximum Parallelism**: 20 concurrent workers for PDF conversion
- **📉 Ultra-Low DPI**: 80 DPI for maximum speed (vs 300 DPI standard)
- **🗜️ Ultra-Compressed**: 50% JPEG quality for smallest files
- **⚡ Ultra-Fast Timeouts**: 10s conversion, 25s API timeout
- **🚀 Single API Call**: 1 API call per PDF (83% reduction)
- **🧹 Aggressive Memory Management**: Immediate cleanup for speed

## Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [ULTRA-AGGRESSIVE Endpoints](#ultra-aggressive-endpoints)
4. [Firebase Storage Endpoints](#firebase-storage-endpoints)
5. [Upload & Process Endpoints](#upload--process-endpoints)
6. [System Endpoints](#system-endpoints)
7. [Data Models](#data-models)
8. [Error Handling](#error-handling)
9. [Performance Monitoring](#performance-monitoring)
10. [Examples](#examples)

---

## Authentication

The API requires the following environment variables to be configured:

- `OPENAI_API_KEY`: Your OpenAI API key for invoice processing
- `FIREBASE_SERVICE_ACCOUNT_KEY`: Firebase service account JSON string
- `FIREBASE_STORAGE_BUCKET`: Firebase Storage bucket name

---

## Core Endpoints

### 1. Parse Single Invoice (ULTRA-AGGRESSIVE)

**Endpoint**: `POST /api/v2/parse-invoice/`  
**Description**: Process a single PDF invoice with ULTRA-AGGRESSIVE optimizations (20s target)

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file`: PDF file (required)

**Response**:
```json
{
  "success": true,
  "message": "Invoice processed successfully with ULTRA-AGGRESSIVE optimizations",
  "request_id": "req_123456789",
  "data": {
    "invoice_metadata": {
      "invoice_number": "INV-2024-001",
      "invoice_date": "2024-01-15",
      "due_date": "2024-02-15",
      "po_number": "PO-2024-001",
      "terms": "Net 30"
    },
    "vendor_information": {
      "company_name": "ABC Corporation",
      "contact_info": {
        "name": "John Doe",
        "address": "123 Business St, City, State 12345",
        "phone": "+1-555-0123",
        "email": "contact@abccorp.com"
      },
      "tax_id": "12-3456789",
      "vendor_id": "VEND001"
    },
    "customer_information": {
      "company_name": "XYZ Company",
      "contact_info": {
        "name": "Jane Smith",
        "address": "456 Customer Ave, City, State 67890",
        "phone": "+1-555-0456",
        "email": "accounts@xyzcompany.com"
      },
      "customer_id": "CUST001"
    },
    "delivery_information": {
      "delivery_address": "456 Customer Ave, City, State 67890",
      "delivery_date": "2024-01-10",
      "delivery_instructions": "Leave at front desk"
    },
    "parcel_information": {
      "tracking_number": "1Z999AA1234567890",
      "shipping_method": "Ground",
      "parcel_weight": 5.5,
      "parcel_dimensions": "12x8x4 inches",
      "package_count": 1,
      "shipping_cost": 15.99,
      "pickup_date": "2024-01-08",
      "delivery_date": "2024-01-10",
      "service_type": "Standard",
      "shipping_notes": "Signature required"
    },
    "commodity_details": {
      "items": [
        {
          "description": "Office Supplies",
          "quantity": 10,
          "unit_price": 25.00,
          "amount": 250.00,
          "item": "Office Supplies",
          "invoice_line_number": 1,
          "notes": "Premium quality supplies"
        }
      ],
      "total_items": 1,
      "total_tonnage": 0.0,
      "total_weight": 0.0,
      "total_volume": 0.0,
      "service_locations_count": 1,
      "service_date_range": "2024-01-15",
      "commodity_types": ["Office Supplies"],
      "container_types": [],
      "service_types": ["Standard"]
    },
    "financial_summary": {
      "subtotal": 250.00,
      "tax_amount": 20.00,
      "tax_rate": 8.0,
      "total_amount": 270.00,
      "currency": "USD",
      "payment_method": "Credit Card"
    },
    "additional_information": {
      "notes": "Thank you for your business",
      "special_instructions": "Please include invoice number on check",
      "reference_numbers": {
        "order_ref": "ORD-2024-001",
        "contract_ref": "CON-2024-001"
      }
    },
    "processing_metadata": {
      "processed_at": "2024-01-15T10:30:00Z",
      "processing_time_seconds": 18.5,
      "api_version": "2.0.0",
      "confidence_score": 0.95,
      "target_achieved": true,
      "optimization_level": "ULTRA-AGGRESSIVE"
    }
  }
}
```

### 2. Parse Multiple Invoices (ULTRA-AGGRESSIVE)

**Endpoint**: `POST /api/v2/parse-multiple-invoices/`  
**Description**: Process multiple PDF invoices with ULTRA-AGGRESSIVE batch processing

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `files`: Array of PDF files (required)

**Response**:
```json
{
  "success": true,
  "message": "ULTRA-AGGRESSIVE batch processing completed",
  "request_id": "req_123456789",
  "results": [
    {
      "filename": "invoice1.pdf",
      "success": true,
      "data": { /* ParsedInvoiceData object */ },
      "processing_time_seconds": 18.2,
      "target_achieved": true
    },
    {
      "filename": "invoice2.pdf",
      "success": true,
      "data": { /* ParsedInvoiceData object */ },
      "processing_time_seconds": 19.8,
      "target_achieved": true
    }
  ],
  "batch_metadata": {
    "total_files": 2,
    "successful_files": 2,
    "failed_files": 0,
    "total_processing_time_seconds": 38.0,
    "average_processing_time_seconds": 19.0,
    "target_achieved": true,
    "optimization_level": "ULTRA-AGGRESSIVE",
    "processed_at": "2024-01-15T10:30:00Z",
    "api_version": "2.0.0"
  }
}
```

---

## ULTRA-AGGRESSIVE Endpoints

### 1. Ultra-Fast Single Invoice Processing

**Endpoint**: `POST /api/v2/parse-invoice-ultra-fast/`  
**Description**: Process single invoice with ULTRA-FAST optimizations (30s target)

### 2. Ultra-Aggressive Single Invoice Processing

**Endpoint**: `POST /api/v2/parse-invoice-ultra-aggressive/`  
**Description**: Process single invoice with ULTRA-AGGRESSIVE optimizations (20s target)

### 3. Ultra-Optimized Batch Processing

**Endpoint**: `POST /api/v2/parse-multiple-invoices-ultra-optimized/`  
**Description**: Process multiple invoices with maximum parallel processing

---

## Firebase Storage Endpoints

### 1. Parse Firebase Storage Invoice (ULTRA-AGGRESSIVE)

**Endpoint**: `POST /api/v2/parse-firebase-storage-urls-ultra-optimized/`  
**Description**: Process invoices from Firebase Storage with ULTRA-AGGRESSIVE optimizations

**Request**:
```json
{
  "files": [
    {
      "storage_path": "invoices/invoice1.pdf"
    },
    {
      "storage_url": "https://firebasestorage.googleapis.com/v0/b/bucket/o/invoices%2Finvoice2.pdf"
    }
  ]
}
```

### 2. Upload and Process Multiple Invoices

**Endpoint**: `POST /api/v2/upload-and-process-multiple-invoices/`  
**Description**: Upload multiple PDFs to Firebase and process with ULTRA-AGGRESSIVE optimizations

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `files`: Array of PDF files (required)

---

## System Endpoints

### 1. Health Check

**Endpoint**: `GET /api/v2/health`  
**Description**: Check API health and status

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "optimization_level": "ULTRA-AGGRESSIVE",
  "target_processing_time": 20
}
```

### 2. Configuration

**Endpoint**: `GET /api/v2/config`  
**Description**: Get current API configuration

**Response**:
```json
{
  "dpi": 80,
  "format": "JPEG",
  "thread_count": 20,
  "request_timeout": 25.0,
  "max_retries": 1,
  "retry_delay": 0.2,
  "target_processing_time": 20,
  "optimization_level": "ULTRA-AGGRESSIVE"
}
```

### 3. Performance Metrics

**Endpoint**: `GET /api/v2/performance`  
**Description**: Get performance statistics and metrics

**Response**:
```json
{
  "total_requests": 150,
  "successful_requests": 145,
  "failed_requests": 5,
  "average_processing_time": 18.5,
  "target_achievement_rate": 0.92,
  "optimization_level": "ULTRA-AGGRESSIVE",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

---

## Performance Monitoring

### Real-Time Metrics

The API provides real-time performance monitoring:

- **Processing Time**: Actual vs target (20s)
- **Target Achievement**: Percentage of requests meeting 20s target
- **Concurrent Workers**: Number of active parallel processors
- **API Call Reduction**: 83% fewer API calls (1 per PDF vs 5-6)
- **Memory Usage**: Aggressive cleanup metrics

### Performance Targets

- **Single PDF**: 15-20 seconds
- **Multiple PDFs**: 20-25 seconds each (parallel)
- **API Calls**: 1 per PDF
- **File Size**: 60-70% smaller images
- **Concurrency**: 20 workers maximum

---

## Error Handling

### Common Error Responses

```json
{
  "success": false,
  "error": "PDF_CONVERSION_FAILED",
  "message": "Failed to convert PDF to images",
  "details": "Timeout after 10 seconds",
  "request_id": "req_123456789"
}
```

### Error Codes

- `PDF_CONVERSION_FAILED`: PDF to image conversion failed
- `API_PROCESSING_FAILED`: OpenAI API processing failed
- `INVALID_FILE_FORMAT`: File is not a valid PDF
- `FILE_TOO_LARGE`: File exceeds size limit
- `TIMEOUT`: Processing timeout exceeded

---

## Examples

### cURL Examples

**Single Invoice Processing**:
```bash
curl -X POST "http://127.0.0.1:8003/api/v2/parse-invoice/" \
  -F "file=@invoice.pdf"
```

**Multiple Invoice Processing**:
```bash
curl -X POST "http://127.0.0.1:8003/api/v2/parse-multiple-invoices/" \
  -F "files=@invoice1.pdf" \
  -F "files=@invoice2.pdf" \
  -F "files=@invoice3.pdf"
```

**Health Check**:
```bash
curl "http://127.0.0.1:8003/api/v2/health"
```

### Python Examples

```python
import requests

# Single invoice processing
with open('invoice.pdf', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8003/api/v2/parse-invoice/',
        files={'file': f}
    )
    result = response.json()
    print(f"Processing time: {result['data']['processing_metadata']['processing_time_seconds']}s")
    print(f"Target achieved: {result['data']['processing_metadata']['target_achieved']}")

# Multiple invoice processing
files = [
    ('files', open('invoice1.pdf', 'rb')),
    ('files', open('invoice2.pdf', 'rb'))
]
response = requests.post(
    'http://127.0.0.1:8003/api/v2/parse-multiple-invoices/',
    files=files
)
result = response.json()
print(f"Batch processing time: {result['batch_metadata']['total_processing_time_seconds']}s")
```

---

## Frontend Integration

Access the web interface at: `http://127.0.0.1:8003/`

Features:
- Drag & drop PDF upload
- Real-time progress tracking
- Performance metrics display
- Batch processing support
- Results download

---

## Rate Limits

- **Single Invoice**: No limit (20s processing time)
- **Batch Processing**: Up to 10 files per request
- **Concurrent Requests**: Limited by server resources
- **API Calls**: 1 per PDF (optimized)

---

## Support

For technical support or questions about ULTRA-AGGRESSIVE optimizations:

- **Documentation**: `/docs` (Swagger UI)
- **Health Check**: `/api/v2/health`
- **Configuration**: `/api/v2/config`
- **Performance**: `/api/v2/performance`

---

**🚀 ULTRA-AGGRESSIVE Invoice Parser API v2.0.0**  
**Target: 20 seconds per PDF with maximum parallel processing** 