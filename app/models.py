"""
Pydantic models for API request and response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class InvoiceMetadata(BaseModel):
    """Invoice metadata information."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    po_number: Optional[str] = None
    terms: Optional[str] = None


class ContactInfo(BaseModel):
    """Contact information structure."""
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class VendorInformation(BaseModel):
    """Vendor information structure."""
    company_name: Optional[str] = None
    contact_info: Optional[ContactInfo] = None
    tax_id: Optional[str] = None
    vendor_id: Optional[str] = None


class CustomerInformation(BaseModel):
    """Customer information structure."""
    company_name: Optional[str] = None
    contact_info: Optional[ContactInfo] = None
    customer_id: Optional[str] = None


class DeliveryInformation(BaseModel):
    """Delivery information structure."""
    delivery_address: Optional[str] = None
    delivery_date: Optional[str] = None
    delivery_instructions: Optional[str] = None


class ParcelInformation(BaseModel):
    """Parcel and shipping information structure."""
    tracking_number: Optional[str] = None
    shipping_method: Optional[str] = None
    parcel_weight: Optional[float] = None
    parcel_dimensions: Optional[str] = None
    package_count: Optional[int] = None
    shipping_cost: Optional[float] = None
    pickup_date: Optional[str] = None
    delivery_date: Optional[str] = None
    service_type: Optional[str] = None
    shipping_notes: Optional[str] = None


class CommodityItem(BaseModel):
    """Individual commodity item with comprehensive details."""
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None  # Added missing total_price field
    amount: Optional[float] = None  # Keep both amount and total_price for compatibility
    item: Optional[str] = None
    category: Optional[str] = None  # Added category field
    line_number: Optional[int] = None  # Added line number field
    notes: Optional[str] = None  # Added notes field


class CommodityDetails(BaseModel):
    """Commodity details structure."""
    items: List[CommodityItem] = []
    total_items: Optional[int] = None


class FinancialSummary(BaseModel):
    """Financial summary structure."""
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_rate: Optional[float] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    payment_method: Optional[str] = None


class AdditionalInformation(BaseModel):
    """Additional information structure."""
    notes: Optional[str] = None
    special_instructions: Optional[str] = None
    reference_numbers: Optional[Dict[str, str]] = None


class ProcessingMetadata(BaseModel):
    """Processing metadata."""
    processed_at: str
    processing_time_seconds: float
    api_version: str = "1.0.0"
    confidence_score: Optional[float] = None


class ParsedInvoiceData(BaseModel):
    """Complete parsed invoice data structure."""
    invoice_metadata: InvoiceMetadata
    vendor_information: VendorInformation
    customer_information: CustomerInformation
    delivery_information: DeliveryInformation
    parcel_information: ParcelInformation
    commodity_details: CommodityDetails
    financial_summary: FinancialSummary
    additional_information: AdditionalInformation
    processing_metadata: ProcessingMetadata


class InvoiceResult(BaseModel):
    """Individual invoice processing result."""
    filename: str
    success: bool
    data: Optional[ParsedInvoiceData] = None
    error_details: Optional[str] = None
    processing_time_seconds: float


class BatchProcessingMetadata(BaseModel):
    """Batch processing metadata."""
    total_files: int
    successful_files: int
    failed_files: int
    total_batches: int
    total_processing_time_seconds: float
    processed_at: str
    api_version: str = "1.0.0"


class ParseInvoiceResponse(BaseModel):
    """API response for invoice parsing."""
    success: bool
    message: str
    request_id: str
    data: Optional[ParsedInvoiceData] = None
    error_details: Optional[str] = None


class ParseMultipleInvoicesResponse(BaseModel):
    """API response for multiple invoice parsing."""
    success: bool
    message: str
    request_id: str
    results: List[InvoiceResult] = []
    batch_metadata: Optional[BatchProcessingMetadata] = None
    error_details: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response structure."""
    success: bool = False
    message: str
    request_id: str
    error_details: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


class FirebaseStorageRequest(BaseModel):
    """Request for Firebase Storage file processing."""
    storage_path: Optional[str] = Field(None, description="Firebase Storage path (e.g., 'invoices/file.pdf')")
    storage_url: Optional[str] = Field(None, description="Firebase Storage download URL")
    bucket_name: Optional[str] = Field(None, description="Firebase Storage bucket name (optional)")
    
    def get_effective_path(self) -> str:
        """Get the effective path for Firebase Storage access."""
        if self.storage_url:
            # Extract path from Firebase Storage URL
            if 'firebasestorage.googleapis.com' in self.storage_url:
                # Format: https://firebasestorage.googleapis.com/v0/b/bucket/o/path%2Fto%2Ffile.pdf
                parts = self.storage_url.split('/o/')
                if len(parts) > 1:
                    # Decode URL-encoded path
                    import urllib.parse
                    path = urllib.parse.unquote(parts[1].split('?')[0])
                    return path
            return self.storage_url
        return self.storage_path or ""


class FirebaseStorageMultipleRequest(BaseModel):
    """Request for multiple Firebase Storage files processing."""
    files: List[FirebaseStorageRequest] = Field(..., description="List of Firebase Storage file requests")
    
    def get_effective_paths(self) -> List[str]:
        """Get list of effective paths for Firebase Storage access."""
        return [file_req.get_effective_path() for file_req in self.files] 