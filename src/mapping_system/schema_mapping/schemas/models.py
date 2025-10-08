"""
Pydantic models for demand forecasting schema validation.
"""

from datetime import date
from typing import Optional, List
from decimal import Decimal
from pydantic import BaseModel


class DemandForecastingRecord(BaseModel):

    # --- Primary Identifiers ---
    date: date
    product_id: str
    store_id: str

    # --- Holiday & Special Events ---
    is_holiday: Optional[bool] = None
    holiday_name: Optional[str] = None
    is_retail_holiday: Optional[bool] = None
    school_holiday_flag: Optional[bool] = None

    # --- Daily Sales Performance ---
    unit_net_price: Optional[Decimal] = None
    units_sold: Optional[int] = None

    # --- Pricing & Promotion Information ---
    promotional_price: Optional[Decimal] = None
    promotion_active: Optional[bool] = None
    promotion_start_date: Optional[date] = None
    promotion_end_date: Optional[date] = None

    # --- Product Master Information ---
    category: Optional[str] = None
    subcategory: Optional[str] = None
    product_launch_date: Optional[date] = None
    unit_orig_price: Optional[Decimal] = None
    is_seasonal_product: Optional[bool] = None
    seasonal_peak_months: Optional[str] = None

    # --- Store Master Information ---
    store_cluster: Optional[str] = None
    store_size_sqft: Optional[int] = None
    store_city: Optional[str] = None
    store_state: Optional[str] = None
    store_country: Optional[str] = None
    store_region: Optional[str] = None

    # --- Daily Inventory & Operations ---
    beginning_inventory: Optional[int] = None
    units_ordered: Optional[int] = None

    # --- Weather & Environmental Data ---
    avg_temperature_c: Optional[int] = None
    precipitation_mm: Optional[Decimal] = None

    # --- Economic & Market Context ---
    unemployment_rate_monthly: Optional[Decimal] = None
    population_monthly: Optional[int] = None
    cpi_monthly: Optional[int] = None
    gdp_monthly: Optional[int] = None

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
