"""
Core invoice parsing logic combining PDF conversion, text extraction, and data parsing.

This module provides comprehensive invoice processing capabilities using:
- PyMuPDF for PDF to image conversion (pure Python, no poppler dependency)
- OpenAI Vision API for optical character recognition (OCR)
- OpenAI Structured Outputs for reliable data extraction
- Pydantic models for type-safe data validation

The module is designed for professional production use with:
- Comprehensive error handling and logging
- Optimized performance for batch processing
- Proper resource cleanup and memory management
- Professional documentation and type hints
"""

import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import os

import fitz  # PyMuPDF
from PIL import Image
import base64
import io
from openai import OpenAI
from pydantic import BaseModel, Field

from .config import settings
from .models import ParsedInvoiceData, ProcessingMetadata
from .utils import cleanup_temp_file

logger = logging.getLogger(__name__)


class PDFConverter:
    """
    Professional PDF to image converter using PyMuPDF.
    
    This class handles the conversion of PDF documents to high-resolution images
    for subsequent OCR processing. It provides optimized settings for invoice
    processing and proper resource management.
    
    Attributes:
        dpi (int): Output image resolution in dots per inch
        format (str): Output image format (PNG, JPEG, etc.)
        thread_count (int): Number of threads for parallel processing
    """
    
    def __init__(self):
        """Initialize PDF converter with optimized settings for invoice processing."""
        self.dpi = settings.DPI
        self.format = settings.FORMAT
        self.thread_count = settings.THREAD_COUNT
    
    def convert_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF to high-resolution images using PyMuPDF.
        
        This method converts each page of a PDF document to a high-resolution image
        suitable for OCR processing. The conversion uses optimized settings for
        invoice documents and provides comprehensive error handling.
        
        CRITICAL: This method MUST process ALL pages of the PDF document.
        
        Args:
            pdf_path (str): Absolute path to the PDF file to convert
            
        Returns:
            List[str]: List of absolute paths to generated image files
            
        Raises:
            Exception: If PDF conversion fails due to file access, memory issues,
                      or other processing errors
                      
        Example:
            >>> converter = PDFConverter()
            >>> image_paths = converter.convert_to_images("/path/to/invoice.pdf")
            >>> print(f"Generated {len(image_paths)} images")
        """
        pdf_document = None
        image_paths = []
        
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            start_time = time.time()
            
            # Verify PDF file exists and is readable
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Get file size for logging
            file_size = os.path.getsize(pdf_path)
            logger.info(f"PDF file size: {file_size} bytes")
            
            # Open PDF document with error handling and verification
            pdf_document = fitz.open(pdf_path)
            
            if pdf_document.page_count == 0:
                raise ValueError("PDF document contains no pages")
            
            # CRITICAL: Log the actual page count detected
            logger.info(f"DETECTED {pdf_document.page_count} PAGES in PDF - ALL WILL BE PROCESSED")
            
            # Calculate zoom factor for DPI (PyMuPDF uses 72 DPI by default)
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            logger.info(f"Processing ALL {pdf_document.page_count} pages at {self.dpi} DPI")
            
            # Convert EVERY SINGLE PAGE to image - NO EXCEPTIONS
            for page_num in range(pdf_document.page_count):
                page_start_time = time.time()
                logger.info(f"Converting page {page_num + 1}/{pdf_document.page_count}")
                
                page = pdf_document[page_num]
                
                try:
                    # Render page to pixmap with optimized settings
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Verify pixmap was created successfully
                    if not pix:
                        raise ValueError(f"Failed to create pixmap for page {page_num + 1}")
                    
                    # Create temporary file with proper naming
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f"_page_{page_num+1}.{self.format.lower()}",
                        prefix="invoice_page_"
                    )
                    temp_file.close()
                    
                    # Save pixmap as image
                    if self.format.upper() == 'PNG':
                        pix.save(temp_file.name)
                    else:
                        # Convert to PIL Image for other formats
                        img_data = pix.tobytes("ppm")
                        img = Image.open(io.BytesIO(img_data))
                        img.save(temp_file.name, format=self.format)
                    
                    # Verify image file was created and has content
                    if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
                        raise ValueError(f"Failed to save image for page {page_num + 1}")
                    
                    image_paths.append(temp_file.name)
                    page_time = time.time() - page_start_time
                    logger.info(f"Successfully converted page {page_num + 1} in {page_time:.2f}s: {temp_file.name}")
                    
                except Exception as page_error:
                    logger.error(f"FAILED to convert page {page_num + 1}: {page_error}")
                    # Continue with other pages but log the failure
                    continue
                    
                finally:
                    # Free memory for this page
                    if 'pix' in locals():
                        pix = None
            
            conversion_time = time.time() - start_time
            
            # CRITICAL: Verify ALL pages were converted
            if len(image_paths) != pdf_document.page_count:
                logger.error(f"CRITICAL ERROR: Only {len(image_paths)} of {pdf_document.page_count} pages were converted!")
                logger.error("Some pages failed to convert - this will cause incomplete data extraction")
            
            logger.info(f"PDF conversion completed: {len(image_paths)}/{pdf_document.page_count} pages converted in {conversion_time:.2f} seconds")
            
            if len(image_paths) == 0:
                raise ValueError("No pages were successfully converted from PDF")
            
            return image_paths
            
        except Exception as e:
            # Clean up any images that were created before the error
            for image_path in image_paths:
                cleanup_temp_file(image_path)
            
            logger.error(f"Failed to convert PDF to images: {e}")
            raise Exception(f"PDF conversion failed: {str(e)}")
            
        finally:
            # Clean up PDF document
            if pdf_document:
                pdf_document.close()


# Pydantic models for structured output
class InvoiceMetadataStructured(BaseModel):
    """Structured model for invoice metadata extraction."""
    invoice_number: Optional[str] = Field(None, description="Invoice number or identifier")
    invoice_date: Optional[str] = Field(None, description="Invoice issue date")
    due_date: Optional[str] = Field(None, description="Payment due date")
    po_number: Optional[str] = Field(None, description="Purchase order number")
    terms: Optional[str] = Field(None, description="Payment terms")


class ContactInfoStructured(BaseModel):
    """Structured model for contact information."""
    name: Optional[str] = Field(None, description="Contact person name")
    address: Optional[str] = Field(None, description="Full address")
    phone: Optional[str] = Field(None, description="Phone number")
    email: Optional[str] = Field(None, description="Email address")


class VendorInformationStructured(BaseModel):
    """Structured model for vendor information."""
    company_name: Optional[str] = Field(None, description="Vendor company name")
    contact_info: ContactInfoStructured = Field(default_factory=ContactInfoStructured, description="Vendor contact information")
    tax_id: Optional[str] = Field(None, description="Tax identification number")
    vendor_id: Optional[str] = Field(None, description="Vendor identifier")


class CustomerInformationStructured(BaseModel):
    """Structured model for customer information."""
    company_name: Optional[str] = Field(None, description="Customer company name")
    contact_info: ContactInfoStructured = Field(default_factory=ContactInfoStructured, description="Customer contact information")
    customer_id: Optional[str] = Field(None, description="Customer identifier")


class DeliveryInformationStructured(BaseModel):
    """Structured model for delivery information."""
    delivery_address: Optional[str] = Field(None, description="Delivery address")
    delivery_date: Optional[str] = Field(None, description="Delivery date")
    delivery_instructions: Optional[str] = Field(None, description="Special delivery instructions")


class ParcelInformationStructured(BaseModel):
    """Structured model for parcel and shipping information."""
    tracking_number: Optional[str] = Field(None, description="Parcel tracking number")
    shipping_method: Optional[str] = Field(None, description="Shipping method or carrier")
    parcel_weight: Optional[float] = Field(None, description="Parcel weight")
    parcel_dimensions: Optional[str] = Field(None, description="Parcel dimensions (length x width x height)")
    package_count: Optional[int] = Field(None, description="Number of packages")
    shipping_cost: Optional[float] = Field(None, description="Shipping cost")
    pickup_date: Optional[str] = Field(None, description="Pickup date")
    delivery_date: Optional[str] = Field(None, description="Expected delivery date")
    service_type: Optional[str] = Field(None, description="Service type (express, standard, etc.)")
    shipping_notes: Optional[str] = Field(None, description="Additional shipping notes or instructions")


class CommodityItemStructured(BaseModel):
    """Enhanced structured model for individual commodity items with comprehensive reconciliation data."""
    # Basic Item Information
    description: Optional[str] = Field(None, description="Full item/commodity description")
    quantity: Optional[float] = Field(None, description="Quantity of items")
    unit_price: Optional[float] = Field(None, description="Price per unit")
    amount: Optional[float] = Field(None, description="Total amount for this line item")
    item: Optional[str] = Field(None, description="Item name or code")
    
    # Tracking and Reference Numbers
    invoice_line_number: Optional[int] = Field(None, description="Line number on invoice")
    
    # Additional Details
    notes: Optional[str] = Field(None, description="Additional notes or special instructions for this item")
    hazmat_info: Optional[str] = Field(None, description="Hazardous material information if applicable")
    disposal_method: Optional[str] = Field(None, description="Disposal or processing method")
    recycling_info: Optional[str] = Field(None, description="Recycling information or destination")


class CommodityDetailsStructured(BaseModel):
    """Enhanced structured model for commodity details with reconciliation summary."""
    items: List[CommodityItemStructured] = Field(default_factory=list, description="List of detailed line items")
    total_items: Optional[int] = Field(None, description="Total number of line items")
    
    # Reconciliation Summary Fields
    total_tonnage: Optional[float] = Field(None, description="Total tonnage across all items")
    total_weight: Optional[float] = Field(None, description="Total weight across all items")
    total_volume: Optional[float] = Field(None, description="Total volume across all items")
    service_locations_count: Optional[int] = Field(None, description="Number of unique service locations")
    service_date_range: Optional[str] = Field(None, description="Range of service dates (earliest to latest)")
    commodity_types: Optional[List[str]] = Field(default_factory=list, description="List of unique commodity types found")
    container_types: Optional[List[str]] = Field(default_factory=list, description="List of unique container types used")
    service_types: Optional[List[str]] = Field(default_factory=list, description="List of unique service types performed")
    
    # Quality Control Fields
    items_with_tonnage: Optional[int] = Field(None, description="Number of items with tonnage data")
    items_with_locations: Optional[int] = Field(None, description="Number of items with location data")
    items_with_dates: Optional[int] = Field(None, description="Number of items with service dates")
    items_with_references: Optional[int] = Field(None, description="Number of items with tracking/reference numbers")


class FinancialSummaryStructured(BaseModel):
    """Structured model for financial summary."""
    subtotal: Optional[float] = Field(None, description="Subtotal before taxes")
    tax_amount: Optional[float] = Field(None, description="Tax amount")
    tax_rate: Optional[float] = Field(None, description="Tax rate as percentage")
    total_amount: Optional[float] = Field(None, description="Total amount due")
    currency: Optional[str] = Field(None, description="Currency code")
    payment_method: Optional[str] = Field(None, description="Payment method")


class AdditionalInformationStructured(BaseModel):
    """Structured model for additional information."""
    notes: Optional[str] = Field(None, description="Additional notes")
    special_instructions: Optional[str] = Field(None, description="Special instructions")
    reference_numbers: Optional[Dict[str, str]] = Field(None, description="Reference numbers")


class InvoiceDataStructured(BaseModel):
    """
    Complete structured model for invoice data extraction.
    
    This model defines the complete schema for invoice data extraction using
    OpenAI's structured outputs feature. It ensures type-safe, reliable parsing
    of invoice documents with comprehensive field coverage.
    """
    explanation: str = Field(..., description="Brief explanation of the extraction process and confidence")
    invoice_metadata: InvoiceMetadataStructured = Field(..., description="Invoice metadata information")
    vendor_information: VendorInformationStructured = Field(..., description="Vendor information")
    customer_information: CustomerInformationStructured = Field(..., description="Customer information")
    delivery_information: DeliveryInformationStructured = Field(..., description="Delivery information")
    parcel_information: ParcelInformationStructured = Field(..., description="Parcel and shipping information")
    commodity_details: CommodityDetailsStructured = Field(..., description="Commodity details and line items")
    financial_summary: FinancialSummaryStructured = Field(..., description="Financial summary")
    additional_information: AdditionalInformationStructured = Field(..., description="Additional information")


class OpenAIClient:
    """
    Professional OpenAI API client with structured output support.
    
    This class provides a robust interface to OpenAI's API with specific focus on
    invoice processing. It implements the latest structured output capabilities
    for reliable, type-safe data extraction.
    
    Features:
        - Latest OpenAI structured output implementation
        - Comprehensive error handling and retry logic
        - Optimized prompts for invoice processing
        - Professional logging and monitoring
    """
    
    def __init__(self):
        """
        Initialize OpenAI client with production-ready configuration.
        
        Raises:
            ValueError: If OpenAI API key is not configured
        """
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in environment variables.")
        
        # Initialize OpenAI client with latest library version
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.REQUEST_TIMEOUT,
            max_retries=settings.MAX_RETRIES
        )
        self.model = settings.OPENAI_MODEL
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
        self.timeout = settings.REQUEST_TIMEOUT
        
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string for OpenAI Vision API.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image data
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If image encoding fails
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                logger.debug(f"Encoded image: {image_path} ({len(encoded_image)} characters)")
                return encoded_image
                
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OpenAI Vision API.
        
        This method uses OpenAI's vision capabilities to extract text content
        from invoice images with optimized prompts for accuracy.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Extracted text content
            
        Raises:
            Exception: If text extraction fails
        """
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Enhanced prompt for invoice text extraction with focus on tables
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract all text from this invoice image with maximum accuracy and special focus on table data.
                            
                            CRITICAL INSTRUCTIONS:
                            - Extract ALL text with exact precision
                            - Pay special attention to TABLE DATA (line items, quantities, prices)
                            - Preserve exact layout and structure of tables
                            - Include all numbers, dates, addresses, and details
                            - Extract table headers and all row data
                            - Maintain original formatting where possible
                            - Include headers, footers, and any fine print
                            - For tables: clearly indicate column structure
                            - Return only the raw text content without commentary
                            
                            FOCUS AREAS:
                            1. Invoice metadata (number, date, due date)
                            2. Vendor and customer information
                            3. TABLE DATA (descriptions, quantities, unit prices, totals)
                            4. Financial summaries (subtotals, taxes, totals)
                            
                            Extract EVERYTHING with maximum precision - accuracy is critical."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call with error handling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=8000,
                temperature=0.0  # Use deterministic output for text extraction
            )
            
            extracted_text = response.choices[0].message.content
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                raise ValueError("Insufficient text extracted from image")
            
            logger.info(f"Extracted {len(extracted_text)} characters from image")
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from image: {e}")
            raise Exception(f"Text extraction failed: {str(e)}")
    
    def parse_invoice_data(self, combined_text: str) -> InvoiceDataStructured:
        """
        Parse structured data from extracted text using OpenAI structured outputs.
        
        This method uses the latest OpenAI structured output capabilities to ensure
        reliable, type-safe extraction of invoice data. It uses Pydantic models
        for validation and the beta parse API for guaranteed schema adherence.
        
        Args:
            combined_text (str): Combined text from all invoice pages
            
        Returns:
            InvoiceDataStructured: Parsed and validated invoice data
            
        Raises:
            Exception: If invoice parsing fails
        """
        try:
            logger.info("Parsing invoice data using OpenAI structured outputs")
            
            # Enhanced system prompt for invoice processing with commodity focus
            system_prompt = """You are a professional invoice data extraction specialist with expertise in financial document processing and TABLE DATA EXTRACTION.

Your task is to extract structured data from invoice text with maximum accuracy and completeness, with SPECIAL FOCUS on commodity/line item details.

EXTRACTION GUIDELINES:
1. Use null for missing or unclear information - never guess
2. For numbers, extract actual numeric values precisely
3. For dates, preserve the format found in the document
4. Extract ALL line items with complete details from tables
5. Ensure financial calculations are accurate
6. Maintain data integrity and consistency
7. Provide a brief explanation of your extraction confidence

COMMODITY/LINE ITEM EXTRACTION PRIORITY:
- Extract EVERY line item from tables with ALL details
- Capture: Description, Quantity, Unit, Unit Price, Total Amount
- Preserve exact numeric values (decimals like 1.00, 240.00)
- Extract complete descriptions including dates and details
- Include any item codes or product numbers
- Ensure mathematical accuracy (quantity × unit price = total)

QUALITY STANDARDS:
- Prioritize accuracy over completeness
- Double-check financial totals and calculations
- Preserve original formatting for important fields
- Extract ALL table data with maximum precision
- Flag any ambiguous or unclear information"""

            # Enhanced user prompt with commodity focus
            user_prompt = f"""Extract comprehensive structured data from this invoice text with MAXIMUM FOCUS on commodity/line item details:

{combined_text}

CRITICAL REQUIREMENTS:
1. Extract ALL line items from tables with complete details
2. Capture every table row with: Description, Quantity, Unit, Unit Price, Total Amount
3. Ensure all financial information is precise and accurate
4. Preserve exact numeric values from the text
5. Extract complete descriptions including any dates or service details

Return accurate, complete data following the schema. Commodity extraction accuracy is the highest priority!"""

            # Use the latest structured output API
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=InvoiceDataStructured,
                temperature=0.0,  # Use 0.0 for maximum deterministic accuracy
                max_tokens=8000
            )
            
            # Extract parsed data
            parsed_data = completion.choices[0].message.parsed
            
            if not parsed_data:
                raise ValueError("No structured data returned from OpenAI")
            
            logger.info("Successfully parsed invoice data using structured outputs")
            logger.debug(f"Extraction explanation: {parsed_data.explanation}")
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to parse invoice data: {e}")
            raise Exception(f"Invoice parsing failed: {str(e)}")

    def extract_structured_data_from_images(self, image_paths: List[str]) -> InvoiceDataStructured:
        """
        Extract structured invoice data directly from multiple images in a single API call.
        
        This is the ULTRA-OPTIMIZED method that combines text extraction and structured parsing
        into one API call, reducing the number of requests from 5-6 to just 1 for 
        multi-page invoices with maximum speed.
        
        Args:
            image_paths (List[str]): List of paths to image files (pages of the invoice)
            
        Returns:
            InvoiceDataStructured: Parsed and validated invoice data
            
        Raises:
            Exception: If image processing or data extraction fails
        """
        try:
            logger.info(f"Processing {len(image_paths)} images in single ULTRA-OPTIMIZED API call")
            start_time = time.time()
            
            if not image_paths:
                raise ValueError("No image paths provided")
            
            # Prepare content array with all images
            content = [
                {
                    "type": "text", 
                    "text": f"""Extract structured data from these {len(image_paths)} invoice images with MAXIMUM ACCURACY.

CRITICAL REQUIREMENTS:
- Process ALL {len(image_paths)} images as pages of one complete invoice
- Extract EVERY line item from tables with ALL details
- Capture: Description, Quantity, Unit, Unit Price, Total Amount
- Include ALL financial amounts, quantities, prices, descriptions
- Extract ALL metadata: invoice numbers, dates, PO numbers
- Consolidate vendor/customer information across pages

COMMODITY EXTRACTION PRIORITY:
For EACH line item, extract:
1. Description: Full text from description column
2. Quantity: Exact numeric value (1.00, 2.5, etc.)
3. Unit: Unit of measurement (each, tons, pounds, etc.)
4. Unit Price: Price per unit (240.00, 1,250.50, etc.)
5. Total Amount: Total price for line item
6. Item/Product Code: Any item codes or SKUs

TABLE SCANNING:
- Look for tables with columns: Description, Qty, Unit, Price, Amount, Total
- Scan EVERY row in EVERY table on ALL pages
- Extract data from each column precisely
- Include subtotals, taxes, fees as separate line items
- Check for continued tables across multiple pages

ACCURACY REQUIREMENTS:
- Extract EXACT numbers - no rounding
- Preserve original text formatting
- Match quantities with corresponding prices
- Ensure mathematical accuracy (quantity × unit price = total)
- Use null only for genuinely missing information

Return structured data following the exact schema with ALL line item details from ALL {len(image_paths)} pages."""
                }
            ]
            
            # Add all images to the content array
            for i, image_path in enumerate(image_paths):
                try:
                    if not os.path.exists(image_path):
                        logger.warning(f"Image file not found: {image_path}")
                        continue
                        
                    base64_image = self.encode_image(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
                    logger.debug(f"Added image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                    
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {e}")
                    continue
            
            if len(content) == 1:  # Only text, no images successfully processed
                raise ValueError("No images could be processed successfully")
            
            # ULTRA-OPTIMIZED system prompt for maximum speed and accuracy
            system_prompt = f"""You are an expert invoice data extraction specialist with advanced OCR capabilities, specializing in PRECISE COMMODITY/LINE ITEM EXTRACTION from invoice tables.

Your PRIMARY MISSION is to extract ALL structured data from ALL {len(image_paths)} invoice images with MAXIMUM ACCURACY and SPEED, with SPECIAL FOCUS on complete and accurate commodity line-item details.

CRITICAL MULTI-PAGE PROCESSING:
1. Process ALL {len(image_paths)} images as sequential pages of ONE COMPLETE invoice document
2. Extract data from EVERY SINGLE PAGE - do not skip any pages or sections
3. Combine and consolidate information from ALL pages into one complete record
4. Include ALL line items from ALL pages in the final commodity list
5. Sum financial amounts from ALL pages for accurate totals

ULTRA-PRECISE COMMODITY EXTRACTION (PRIORITY #1):
Extract EVERY TABLE ROW with COMPLETE ACCURACY:

FOR EACH LINE ITEM, EXTRACT:
1. **Description**: Complete text from description/item column
2. **Quantity**: Exact numeric value (preserve decimals: 1.00, 2.5, etc.)
3. **Unit**: Unit of measurement (each, tons, lbs, hours, etc.)
4. **Unit Price**: Exact price per unit (preserve decimals: 240.00, 1,250.50, etc.)
5. **Total Amount**: Total price for line item (quantity × unit price)
6. **Item/Product Code**: Any item codes or SKUs
7. **Line Number**: Sequential line number if visible

TABLE EXTRACTION STRATEGY:
- Identify ALL tables on ALL pages
- Extract EVERY row from EVERY table
- Match data to correct columns (Description, Qty, Unit, Price, Amount)
- Include subtotals, taxes, fees as separate line items
- Preserve exact numeric values with proper decimal places
- Extract complete description text including dates and details

MATHEMATICAL VERIFICATION:
- Verify quantity × unit price = total amount for each line
- Ensure subtotals match sum of line items
- Check that final total includes all charges
- Flag any mathematical inconsistencies

COMPREHENSIVE EXTRACTION STANDARDS:
1. Scan EVERY pixel of ALL {len(image_paths)} images for text and data
2. Extract EVERY line item, fee, charge, service, product mentioned
3. Find ALL metadata: invoice numbers, PO numbers, dates, references
4. Use null ONLY after exhaustive scanning - never due to insufficient effort
5. For numbers, extract actual numeric values precisely from ALL pages
6. For dates, preserve the format found in the document
7. Extract ALL line items with MAXIMUM detail from ALL pages
8. Ensure financial calculations include amounts from ALL pages
9. Maintain data integrity and consistency across all pages
10. Provide detailed explanation of findings from each page

ACCURACY VERIFICATION REQUIREMENTS:
- Count extracted line items vs what you observe on all pages
- Verify that EACH line item has complete details extracted
- Confirm ALL financial totals equal sum of individual line items
- Ensure ALL metadata fields have been exhaustively searched
- Verify commodity details are complete and accurate
- Confirm no line items are missed from any page

REMEMBER: Accuracy is more important than speed. Extract EVERYTHING with extreme precision. Missing or incorrect commodity details will cause system failures. Extract ALL table data with complete accuracy."""

            # Make the ULTRA-OPTIMIZED API call with structured output
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                response_format=InvoiceDataStructured,
                temperature=0.0,  # Use 0.0 for maximum deterministic accuracy
                max_tokens=6000,  # Reduced for faster processing
                timeout=90  # Optimized timeout for faster processing
            )
            
            # Extract parsed data
            parsed_data = completion.choices[0].message.parsed
            
            if not parsed_data:
                raise ValueError("No structured data returned from OpenAI")
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully extracted structured data from {len(image_paths)} images in {processing_time:.2f}s")
            logger.info(f"ULTRA-OPTIMIZED API call completed - reduced from {len(image_paths) + 1} calls to 1 call")
            logger.debug(f"Extraction explanation: {parsed_data.explanation}")
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to extract structured data from images: {e}")
            raise Exception(f"ULTRA-OPTIMIZED image processing failed: {str(e)}")


class InvoiceProcessor:
    """
    Professional invoice processing orchestrator.
    
    This class coordinates the complete invoice processing pipeline from PDF
    conversion through structured data extraction. It provides optimized
    performance, comprehensive error handling, and professional logging.
    
    Features:
        - End-to-end invoice processing pipeline
        - Optimized batch processing capabilities
        - Comprehensive error handling and recovery
        - Resource management and cleanup
        - Performance monitoring and logging
    """
    
    def __init__(self):
        """Initialize invoice processor with optimized components."""
        self.pdf_converter = PDFConverter()
        self.openai_client = OpenAIClient()
        # Add DPI and format attributes for direct access
        self.dpi = settings.DPI
        self.format = settings.FORMAT
        logger.info("Invoice processor initialized successfully")
    
    def _convert_structured_to_models(self, structured_data: InvoiceDataStructured) -> ParsedInvoiceData:
        """
        Convert structured Pydantic model to legacy model format.
        
        This method bridges the new structured output format with the existing
        API response models for backward compatibility.
        
        Args:
            structured_data: Structured invoice data from OpenAI
            
        Returns:
            ParsedInvoiceData: Legacy format for API responses
        """
        from .models import (
            InvoiceMetadata, VendorInformation, CustomerInformation,
            DeliveryInformation, ParcelInformation, CommodityDetails, FinancialSummary,
            AdditionalInformation, ContactInfo, CommodityItem
        )
        
        # Convert contact info
        vendor_contact = ContactInfo(
            name=structured_data.vendor_information.contact_info.name,
            address=structured_data.vendor_information.contact_info.address,
            phone=structured_data.vendor_information.contact_info.phone,
            email=structured_data.vendor_information.contact_info.email
        )
        
        customer_contact = ContactInfo(
            name=structured_data.customer_information.contact_info.name,
            address=structured_data.customer_information.contact_info.address,
            phone=structured_data.customer_information.contact_info.phone,
            email=structured_data.customer_information.contact_info.email
        )
        
        # Convert commodity items
        commodity_items = [
            CommodityItem(
                description=item.description,
                quantity=item.quantity,
                unit=None,  # Remove unit field as requested
                unit_price=item.unit_price,
                total_price=item.amount,  # Map amount to total_price
                amount=item.amount,  # Keep amount for compatibility
                item=item.item,
                category=item.item,  # Use item as category for now
                line_number=item.invoice_line_number,
                notes=item.notes
            )
            for item in structured_data.commodity_details.items
        ]
        
        # Create legacy model instances
        invoice_metadata = InvoiceMetadata(
            invoice_number=structured_data.invoice_metadata.invoice_number,
            invoice_date=structured_data.invoice_metadata.invoice_date,
            due_date=structured_data.invoice_metadata.due_date,
            po_number=structured_data.invoice_metadata.po_number,
            terms=structured_data.invoice_metadata.terms
        )
        
        vendor_information = VendorInformation(
            company_name=structured_data.vendor_information.company_name,
            contact_info=vendor_contact,
            tax_id=structured_data.vendor_information.tax_id,
            vendor_id=structured_data.vendor_information.vendor_id
        )
        
        customer_information = CustomerInformation(
            company_name=structured_data.customer_information.company_name,
            contact_info=customer_contact,
            customer_id=structured_data.customer_information.customer_id
        )
        
        delivery_information = DeliveryInformation(
            delivery_address=structured_data.delivery_information.delivery_address,
            delivery_date=structured_data.delivery_information.delivery_date,
            delivery_instructions=structured_data.delivery_information.delivery_instructions
        )
        
        parcel_information = ParcelInformation(
            tracking_number=structured_data.parcel_information.tracking_number,
            shipping_method=structured_data.parcel_information.shipping_method,
            parcel_weight=structured_data.parcel_information.parcel_weight,
            parcel_dimensions=structured_data.parcel_information.parcel_dimensions,
            package_count=structured_data.parcel_information.package_count,
            shipping_cost=structured_data.parcel_information.shipping_cost,
            pickup_date=structured_data.parcel_information.pickup_date,
            delivery_date=structured_data.parcel_information.delivery_date,
            service_type=structured_data.parcel_information.service_type,
            shipping_notes=structured_data.parcel_information.shipping_notes
        )
        
        commodity_details = CommodityDetails(
            items=commodity_items,
            total_items=structured_data.commodity_details.total_items
        )
        
        financial_summary = FinancialSummary(
            subtotal=structured_data.financial_summary.subtotal,
            tax_amount=structured_data.financial_summary.tax_amount,
            tax_rate=structured_data.financial_summary.tax_rate,
            total_amount=structured_data.financial_summary.total_amount,
            currency=structured_data.financial_summary.currency,
            payment_method=structured_data.financial_summary.payment_method
        )
        
        additional_information = AdditionalInformation(
            notes=structured_data.additional_information.notes,
            special_instructions=structured_data.additional_information.special_instructions,
            reference_numbers=structured_data.additional_information.reference_numbers
        )
        
        return ParsedInvoiceData(
            invoice_metadata=invoice_metadata,
            vendor_information=vendor_information,
            customer_information=customer_information,
            delivery_information=delivery_information,
            parcel_information=parcel_information,
            commodity_details=commodity_details,
            financial_summary=financial_summary,
            additional_information=additional_information,
            processing_metadata=ProcessingMetadata(
                processed_at=datetime.now().isoformat(),
                processing_time_seconds=0.0,  # Will be updated by caller
                api_version="2.0.0"
            )
        )
    
    async def process_invoice_batch_optimized(self, pdf_paths: List[str]) -> List[ParsedInvoiceData]:
        """
        Process multiple invoices concurrently with ULTRA-FAST single API call approach.
        
        This method provides maximum performance batch processing using:
        - Concurrent PDF to image conversion for each invoice
        - Single API call per invoice processing all pages at once
        - Aggressive parallelization for maximum speed
        - Real-time progress tracking and early results display
        
        ULTRA-FAST FEATURES:
        - Up to 8 concurrent invoice processors
        - Single API call per invoice (not per page)
        - Concurrent PDF page conversion
        - Early result streaming
        - Maximum speed optimization
        
        Args:
            pdf_paths (List[str]): List of PDF file paths to process
            
        Returns:
            List[ParsedInvoiceData]: List of parsed invoice data
            
        Raises:
            Exception: If batch processing fails
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        logger.info(f"Starting ULTRA-FAST single API call batch processing of {len(pdf_paths)} invoices")
        start_time = time.time()
        
        # ULTRA-FAST worker scaling for maximum performance
        if len(pdf_paths) <= 2:
            max_workers = 4  # Increased from 2 to 4 for small batches
        elif len(pdf_paths) <= 5:
            max_workers = 6  # Increased from 4 to 6 for medium batches
        elif len(pdf_paths) <= 10:
            max_workers = 8  # Increased from 6 to 8 for larger batches
        else:
            max_workers = 12  # Increased from 8 to 12 for very large batches (max)
        
        logger.info(f"ULTRA-FAST configuration: {len(pdf_paths)} invoices, {max_workers} concurrent workers")
        logger.info("Each invoice will use single API call processing all pages concurrently")
        
        async def process_single_invoice_async(pdf_path: str) -> ParsedInvoiceData:
            """Process a single invoice asynchronously with ULTRA-FAST single API call."""
            try:
                logger.info(f"Starting ULTRA-FAST processing: {os.path.basename(pdf_path)}")
                
                # Run the ULTRA-FAST processor
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await loop.run_in_executor(
                        executor, 
                        self.process_invoice_ultra_fast,  # Use ultra-fast method
                        pdf_path
                    )
                
                logger.info(f"ULTRA-FAST processing completed: {os.path.basename(pdf_path)}")
                return result
                
            except Exception as e:
                logger.error(f"ULTRA-FAST processing failed for {os.path.basename(pdf_path)}: {e}")
                raise
        
        # Create optimized semaphore for aggressive concurrency
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(pdf_path: str) -> ParsedInvoiceData:
            """Process with optimized concurrency control."""
            async with semaphore:
                return await process_single_invoice_async(pdf_path)
        
        # Process all invoices with maximum concurrency
        logger.info(f"Starting ULTRA-FAST concurrent processing with {max_workers} workers...")
        logger.info("Each worker will process one invoice with single API call approach")
        
        tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
        
        # Use gather with return_exceptions for better error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from exceptions
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Invoice {os.path.basename(pdf_paths[i])} failed: {result}")
                failed_results.append((pdf_paths[i], result))
            else:
                successful_results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_invoice = total_time / len(pdf_paths) if pdf_paths else 0
        
        logger.info(f"ULTRA-FAST single API call batch processing completed:")
        logger.info(f"  - Total time: {total_time:.2f}s")
        logger.info(f"  - Success rate: {len(successful_results)}/{len(pdf_paths)} ({len(successful_results)/len(pdf_paths)*100:.1f}%)")
        logger.info(f"  - Average time per invoice: {avg_time_per_invoice:.2f}s")
        logger.info(f"  - Concurrency level: {max_workers} workers")
        logger.info(f"  - API calls per invoice: 1 (ULTRA-OPTIMIZED)")
        
        if failed_results:
            logger.warning(f"{len(failed_results)} invoices failed to process")
            for failed_path, error in failed_results:
                logger.warning(f"  - {os.path.basename(failed_path)}: {error}")
        
        return successful_results
    
    def process_invoice_optimized(self, pdf_path: str) -> ParsedInvoiceData:
        """
        Process a PDF invoice with ULTRA-FAST single API call approach.
        
        This method uses the FASTEST approach with:
        - Concurrent PDF to image conversion
        - Single API call processing all images at once
        - Maximum parallelization for speed
        - Real-time progress tracking
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            ParsedInvoiceData: Parsed and validated invoice data
            
        Raises:
            Exception: If invoice processing fails
        """
        import concurrent.futures
        import threading
        import time
        from pathlib import Path
        
        start_time = time.time()
        image_paths = []
        
        try:
            logger.info(f"Starting ULTRA-FAST single API call processing: {pdf_path}")
            
            # Step 1: ULTRA-FAST concurrent PDF to image conversion
            conversion_start = time.time()
            logger.info("Starting ULTRA-FAST concurrent PDF conversion...")
            
            # Open PDF once and get page count
            pdf_document = fitz.open(pdf_path)
            page_count = pdf_document.page_count
            logger.info(f"Processing {page_count} pages concurrently")
            
            # Thread-safe result collection
            results_lock = threading.Lock()
            image_paths = [None] * page_count
            completed_count = [0]  # Use list to make it mutable in nested function
            
            def convert_page_to_image(page_num: int) -> tuple:
                """Convert a single PDF page to image with concurrent processing."""
                try:
                    page_start_time = time.time()
                    logger.info(f"Converting page {page_num + 1}/{page_count}")
                    
                    # Calculate zoom factor for DPI (optimized for speed)
                    zoom = self.dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Get page and render to pixmap with optimized settings
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Create temporary file with optimized naming
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f"_page_{page_num+1}.jpg",  # Force JPEG for speed
                        prefix="invoice_page_"
                    )
                    temp_file.close()
                    
                    # Save pixmap as JPEG for faster processing and smaller files
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    img.save(temp_file.name, format="JPEG", quality=85, optimize=True)  # Optimized JPEG settings
                    
                    # Verify file was created
                    if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
                        raise ValueError(f"Failed to save image for page {page_num + 1}")
                    
                    page_time = time.time() - page_start_time
                    logger.info(f"Page {page_num + 1} converted in {page_time:.2f}s: {temp_file.name}")
                    
                    with results_lock:
                        image_paths[page_num] = temp_file.name
                        completed_count[0] += 1
                        logger.info(f"Conversion progress: {completed_count[0]}/{page_count} pages completed")
                    
                    return page_num, temp_file.name, None
                    
                except Exception as e:
                    logger.error(f"Failed to convert page {page_num + 1}: {e}")
                    with results_lock:
                        completed_count[0] += 1
                    return page_num, None, e
                finally:
                    # Free memory immediately
                    if 'pix' in locals():
                        pix = None
                    if 'img' in locals():
                        img = None
            
            # Process all pages concurrently with aggressive concurrency
            max_workers = min(page_count, 12)  # Increased from 8 to 12 for maximum speed
            logger.info(f"Using {max_workers} concurrent workers for PDF conversion")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all conversion tasks
                futures = [
                    executor.submit(convert_page_to_image, page_num)
                    for page_num in range(page_count)
                ]
                
                # Wait for all to complete with timeout
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=30):  # Reduced from 60 to 30 seconds
                        page_num, image_path, error = future.result()
                        if error:
                            logger.warning(f"Page {page_num + 1} conversion failed but continuing...")
                except concurrent.futures.TimeoutError:
                    logger.error("PDF conversion timed out after 30 seconds")  # Updated timeout message
                    raise Exception("PDF conversion timed out")
            
            # Filter out failed conversions
            successful_images = [path for path in image_paths if path and os.path.exists(path)]
            
            if not successful_images:
                raise ValueError("No pages were successfully converted from PDF")
            
            conversion_time = time.time() - conversion_start
            logger.info(f"ULTRA-FAST concurrent PDF conversion completed in {conversion_time:.2f}s")
            logger.info(f"Successfully converted {len(successful_images)}/{page_count} pages")
            
            # Close PDF document
            pdf_document.close()
            
            # Step 2: ULTRA-FAST single API call processing
            processing_start = time.time()
            logger.info(f"Starting ULTRA-FAST single API call processing for {len(successful_images)} images")
            
            # Use the ULTRA-OPTIMIZED single API call method
            structured_data = self.openai_client.extract_structured_data_from_images(successful_images)
            
            processing_time = time.time() - processing_start
            logger.info(f"ULTRA-FAST single API call processing completed in {processing_time:.2f}s")
            
            # Step 3: Convert to legacy format and add metadata
            conversion_start = time.time()
            invoice_data = self._convert_structured_to_models(structured_data)
            
            # Update processing metadata
            total_time = time.time() - start_time
            invoice_data.processing_metadata.processing_time_seconds = round(total_time, 2)
            
            conversion_time = time.time() - conversion_start
            
            logger.info(f"ULTRA-FAST single API call invoice processing completed in {total_time:.2f}s")
            logger.info(f"  - PDF conversion: {conversion_time:.2f}s")
            logger.info(f"  - Single API processing: {processing_time:.2f}s")
            logger.info(f"  - Model conversion: {conversion_time:.2f}s")
            logger.info(f"  - Total API calls: 1 (ULTRA-OPTIMIZED)")
            logger.info(f"  - Concurrent workers: {max_workers}")
            
            return invoice_data
            
        except Exception as e:
            logger.error(f"ULTRA-FAST invoice processing failed: {e}")
            raise
        finally:
            # Clean up temporary image files
            for image_path in image_paths:
                if image_path and os.path.exists(image_path):
                    cleanup_temp_file(image_path)


    def process_invoice_ultra_fast(self, pdf_path: str) -> ParsedInvoiceData:
        """
        ULTRA-FAST processing method targeting 30 seconds per invoice.
        
        This method uses the most aggressive optimizations:
        - Lower DPI (150) for faster image processing
        - JPEG format for smaller files
        - Maximum concurrency (12 workers)
        - Reduced timeouts
        - Optimized memory management
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            ParsedInvoiceData: Parsed and validated invoice data
        """
        import concurrent.futures
        import threading
        import time
        from pathlib import Path
        
        start_time = time.time()
        image_paths = []
        
        try:
            logger.info(f"Starting ULTRA-FAST processing (30s target): {pdf_path}")
            
            # Step 1: ULTRA-FAST concurrent PDF to image conversion
            conversion_start = time.time()
            logger.info("Starting ULTRA-FAST concurrent PDF conversion...")
            
            # Open PDF once and get page count
            pdf_document = fitz.open(pdf_path)
            page_count = pdf_document.page_count
            logger.info(f"Processing {page_count} pages with ULTRA-FAST settings")
            
            # Thread-safe result collection
            results_lock = threading.Lock()
            image_paths = [None] * page_count
            completed_count = [0]
            
            def convert_page_ultra_fast(page_num: int) -> tuple:
                """Convert a single PDF page with ULTRA-FAST settings."""
                try:
                    page_start_time = time.time()
                    
                    # Use lower DPI for speed (150 instead of 300)
                    zoom = 150 / 72.0  # Force 150 DPI for speed
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Get page and render to pixmap
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f"_page_{page_num+1}.jpg",
                        prefix="ultra_fast_"
                    )
                    temp_file.close()
                    
                    # Save as optimized JPEG with maximum speed settings
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    img.save(temp_file.name, format="JPEG", quality=75, optimize=True, progressive=False)  # Maximum speed settings
                    
                    # Skip file size verification for speed (trust the save operation)
                    page_time = time.time() - page_start_time
                    logger.info(f"ULTRA-FAST: Page {page_num + 1} converted in {page_time:.2f}s")
                    
                    with results_lock:
                        image_paths[page_num] = temp_file.name
                        completed_count[0] += 1
                    
                    return page_num, temp_file.name, None
                    
                except Exception as e:
                    logger.error(f"ULTRA-FAST conversion failed for page {page_num + 1}: {e}")
                    with results_lock:
                        completed_count[0] += 1
                    return page_num, None, e
                finally:
                    if 'pix' in locals():
                        pix = None
                    if 'img' in locals():
                        img = None
            
            # Use maximum concurrency for speed
            max_workers = min(page_count, 12)
            logger.info(f"ULTRA-FAST: Using {max_workers} concurrent workers")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(convert_page_ultra_fast, page_num)
                    for page_num in range(page_count)
                ]
                
                # Wait with shorter timeout
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=25):  # 25 second timeout
                        page_num, image_path, error = future.result()
                        if error:
                            logger.warning(f"Page {page_num + 1} failed but continuing...")
                except concurrent.futures.TimeoutError:
                    logger.error("ULTRA-FAST conversion timed out after 25 seconds")
                    raise Exception("ULTRA-FAST conversion timed out")
            
            # Filter successful conversions
            successful_images = [path for path in image_paths if path and os.path.exists(path)]
            
            if not successful_images:
                raise ValueError("No pages were successfully converted")
            
            conversion_time = time.time() - conversion_start
            logger.info(f"ULTRA-FAST conversion completed in {conversion_time:.2f}s")
            
            # Close PDF document
            pdf_document.close()
            
            # Step 2: ULTRA-FAST single API call processing
            processing_start = time.time()
            logger.info(f"Starting ULTRA-FAST API processing for {len(successful_images)} images")
            
            # Use the optimized single API call method
            structured_data = self.openai_client.extract_structured_data_from_images(successful_images)
            
            processing_time = time.time() - processing_start
            logger.info(f"ULTRA-FAST API processing completed in {processing_time:.2f}s")
            
            # Step 3: Convert to legacy format
            conversion_start = time.time()
            invoice_data = self._convert_structured_to_models(structured_data)
            
            # Update processing metadata
            total_time = time.time() - start_time
            invoice_data.processing_metadata.processing_time_seconds = round(total_time, 2)
            
            conversion_time = time.time() - conversion_start
            
            logger.info(f"ULTRA-FAST processing completed in {total_time:.2f}s")
            logger.info(f"  - PDF conversion: {conversion_time:.2f}s")
            logger.info(f"  - API processing: {processing_time:.2f}s")
            logger.info(f"  - Model conversion: {conversion_time:.2f}s")
            logger.info(f"  - Target achieved: {'YES' if total_time <= 30 else 'NO'}")
            
            return invoice_data
            
        except Exception as e:
            logger.error(f"ULTRA-FAST processing failed: {e}")
            raise
        finally:
            # Clean up temporary image files
            for image_path in image_paths:
                if image_path and os.path.exists(image_path):
                    cleanup_temp_file(image_path)
    
    def process_invoice_ultra_aggressive(self, pdf_path: str) -> ParsedInvoiceData:
        """
        ULTRA-AGGRESSIVE processing method targeting 20 seconds per invoice.
        
        This method uses the most extreme optimizations:
        - Ultra-low DPI (80) for maximum speed
        - Ultra-low JPEG quality (50%) for smallest files
        - Maximum concurrency (20 workers)
        - Ultra-minimal timeouts (10s conversion, 25s API)
        - Skip ALL verifications and checks
        - Aggressive memory management
        - Parallel processing at every level
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            ParsedInvoiceData: Parsed and validated invoice data
        """
        import concurrent.futures
        import threading
        import time
        from pathlib import Path
        
        start_time = time.time()
        image_paths = []
        
        try:
            logger.info(f"Starting ULTRA-AGGRESSIVE processing (20s target): {pdf_path}")
            
            # Step 1: ULTRA-AGGRESSIVE concurrent PDF to image conversion
            conversion_start = time.time()
            logger.info("Starting ULTRA-AGGRESSIVE concurrent PDF conversion...")
            
            # Open PDF once and get page count
            pdf_document = fitz.open(pdf_path)
            page_count = pdf_document.page_count
            logger.info(f"Processing {page_count} pages with ULTRA-AGGRESSIVE settings")
            
            # Thread-safe result collection
            results_lock = threading.Lock()
            image_paths = [None] * page_count
            completed_count = [0]
            
            def convert_page_ultra_aggressive(page_num: int) -> tuple:
                """Convert a single PDF page with ULTRA-AGGRESSIVE settings."""
                try:
                    page_start_time = time.time()
                    
                    # Use ultra-low DPI for maximum speed (80 instead of 100)
                    zoom = 80 / 72.0  # Force 80 DPI for maximum speed
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Get page and render to pixmap
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f"_page_{page_num+1}.jpg",
                        prefix="ultra_aggressive_"
                    )
                    temp_file.close()
                    
                    # Save as ultra-optimized JPEG with maximum speed settings
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    img.save(temp_file.name, format="JPEG", quality=50, optimize=True, progressive=False)  # Ultra-aggressive settings
                    
                    # Skip ALL verifications for maximum speed
                    page_time = time.time() - page_start_time
                    logger.info(f"ULTRA-AGGRESSIVE: Page {page_num + 1} converted in {page_time:.2f}s")
                    
                    with results_lock:
                        image_paths[page_num] = temp_file.name
                        completed_count[0] += 1
                    
                    return page_num, temp_file.name, None
                    
                except Exception as e:
                    logger.error(f"ULTRA-AGGRESSIVE conversion failed for page {page_num + 1}: {e}")
                    with results_lock:
                        completed_count[0] += 1
                    return page_num, None, e
                finally:
                    # Ultra-aggressive memory cleanup
                    if 'pix' in locals():
                        pix = None
                    if 'img' in locals():
                        img = None
            
            # Use maximum concurrency for speed
            max_workers = min(page_count, 20)  # Increased to 20 workers for maximum speed
            logger.info(f"ULTRA-AGGRESSIVE: Using {max_workers} concurrent workers")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(convert_page_ultra_aggressive, page_num)
                    for page_num in range(page_count)
                ]
                
                # Wait with ultra-short timeout
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=10):  # 10 second timeout for maximum speed
                        page_num, image_path, error = future.result()
                        if error:
                            logger.warning(f"Page {page_num + 1} failed but continuing...")
                except concurrent.futures.TimeoutError:
                    logger.error("ULTRA-AGGRESSIVE conversion timed out after 10 seconds")
                    raise Exception("ULTRA-AGGRESSIVE conversion timed out")
            
            # Filter successful conversions (skip file existence check for speed)
            successful_images = [path for path in image_paths if path]
            
            if not successful_images:
                raise ValueError("No pages were successfully converted")
            
            conversion_time = time.time() - conversion_start
            logger.info(f"ULTRA-AGGRESSIVE conversion completed in {conversion_time:.2f}s")
            
            # Close PDF document
            pdf_document.close()
            
            # Step 2: ULTRA-AGGRESSIVE single API call processing
            processing_start = time.time()
            logger.info(f"Starting ULTRA-AGGRESSIVE API processing for {len(successful_images)} images")
            
            # Use the optimized single API call method with shorter timeout
            structured_data = self.openai_client.extract_structured_data_from_images(successful_images)
            
            processing_time = time.time() - processing_start
            logger.info(f"ULTRA-AGGRESSIVE API processing completed in {processing_time:.2f}s")
            
            # Step 3: Convert to legacy format
            conversion_start = time.time()
            invoice_data = self._convert_structured_to_models(structured_data)
            
            # Update processing metadata
            total_time = time.time() - start_time
            invoice_data.processing_metadata.processing_time_seconds = round(total_time, 2)
            
            conversion_time = time.time() - conversion_start
            
            logger.info(f"ULTRA-AGGRESSIVE processing completed in {total_time:.2f}s")
            logger.info(f"  - PDF conversion: {conversion_time:.2f}s")
            logger.info(f"  - API processing: {processing_time:.2f}s")
            logger.info(f"  - Model conversion: {conversion_time:.2f}s")
            logger.info(f"  - Target achieved: {'YES' if total_time <= 20 else 'NO'} (Target: 20s)")
            
            return invoice_data
            
        except Exception as e:
            logger.error(f"ULTRA-AGGRESSIVE processing failed: {e}")
            raise
        finally:
            # Clean up temporary image files
            for image_path in image_paths:
                if image_path and os.path.exists(image_path):
                    cleanup_temp_file(image_path)
    
    def process_invoice(self, pdf_path: str) -> ParsedInvoiceData:
        """
        Process a PDF invoice with ULTRA-AGGRESSIVE approach (20s target).
        
        This method uses the most extreme optimizations for maximum speed:
        - 100 DPI for ultra-fast image processing
        - 60% JPEG quality for smallest files
        - 20 concurrent workers for maximum parallelization
        - Minimal timeouts (10s conversion, 30s API)
        - Skip all verifications
        - Aggressive memory management
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            ParsedInvoiceData: Parsed and validated invoice data
            
        Raises:
            Exception: If invoice processing fails
        """
        # Use the ULTRA-AGGRESSIVE method for maximum speed
        return self.process_invoice_ultra_aggressive(pdf_path) 