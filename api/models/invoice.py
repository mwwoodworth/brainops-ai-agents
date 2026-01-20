from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class InvoiceStatus(str, Enum):
    draft = "draft"
    sent = "sent"
    partial = "partial"
    paid = "paid"
    overdue = "overdue"
    cancelled = "cancelled"
    void = "void"
    unpaid = "unpaid"


class InvoiceLineItem(BaseModel):
    id: Optional[str] = None
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    unit_of_measure: Optional[str] = None
    taxable: Optional[bool] = None

    class Config:
        extra = "allow"


class Invoice(BaseModel):
    id: str
    tenant_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    customer_id: str
    job_id: Optional[str] = None
    estimate_id: Optional[str] = None
    invoice_number: str
    status: Optional[InvoiceStatus] = None
    invoice_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    paid_date: Optional[datetime] = None
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None
    amount_paid: Optional[float] = None
    balance_due: Optional[float] = None
    total_cents: Optional[int] = None
    amount_paid_cents: Optional[int] = None
    balance_cents: Optional[int] = None
    line_items: Optional[List[InvoiceLineItem]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"
