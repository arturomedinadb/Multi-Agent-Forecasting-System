"""
Pydantic models for demand forecasting schema validation.
"""

from datetime import date
from typing import Optional, List
from decimal import Decimal
from pydantic import BaseModel

class DemandForecastingRecord(BaseModel):
    
    # --- Core Identifiers (Required for Merging & Analysis) ---
    date: date
    product_id: str
    store_id: str
    
    # --- Sales & Revenue Metrics ---
    units_sold: Optional[int] = None
    unit_price: Optional[Decimal] = None
    
    # --- Pricing & Promotion ---
    promotion_active: Optional[bool] = None
    promotional_price: Optional[Decimal] = None
    
    # --- Product Master Data ---
    category: Optional[str] = None
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    
    # --- Store Master Data ---
    store_city: Optional[str] = None
    store_state: Optional[str] = None
    store_cluster: Optional[str] = None
    
    # --- Inventory Metrics ---
    beginning_inventory: Optional[int] = None
    units_ordered: Optional[int] = None
    
    # --- External Factors ---
    is_holiday: Optional[bool] = None
    holiday_name: Optional[str] = None

class ColumnMapping(BaseModel):
    """Represents a mapping between a source and target column."""
    source_column: str
    target_column: str
    confidence: float
    reasoning: Optional[str] = None

class MappingResult(BaseModel):
    """Represents the final output of the schema mapping process."""
    mappings: List[ColumnMapping]
    final_dataframe: Optional[str] = None # Can hold a JSON representation of the df
    quality_report: Optional[dict] = None
