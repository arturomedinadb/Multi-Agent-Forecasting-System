# Schema Mapping Evaluation (schema_mapping_20251021T002105Z)

## Deterministic Metrics
- **Field Coverage** – PASS (score=1.000, threshold=0.9)
  - Details: {
  "covered_fields": [
    "date",
    "product_id",
    "store_id"
  ],
  "missing_fields": [],
  "required_total": 3
}
- **Type Compatibility** – FAIL (score=0.424, threshold=1.0)
  - Details: {
  "inspected_fields": 33,
  "incompatible_fields": [
    {
      "field": "is_holiday",
      "observed": "float64",
      "expected": "boolean"
    },
    {
      "field": "holiday_name",
      "observed": "float64",
      "expected": "string"
    },
    {
      "field": "is_retail_holiday",
      "observed": "float64",
      "expected": "boolean"
    },
    {
      "field": "school_holiday_flag",
      "observed": "float64",
      "expected": "boolean"
    },
    {
      "field": "unit_net_price",
      "observed": "float64",
      "expected": "number"
    },
    {
      "field": "promotional_price",
      "observed": "int64",
      "expected": "number"
    },
    {
      "field": "promotion_active",
      "observed": "int64",
      "expected": "boolean"
    },
    {
      "field": "unit_orig_price",
      "observed": "float64",
      "expected": "number"
    },
    {
      "field": "is_seasonal_product",
      "observed": "float64",
      "expected": "boolean"
    },
    {
      "field": "store_cluster",
      "observed": "float64",
      "expected": "string"
    },
    {
      "field": "store_size_sqft",
      "observed": "float64",
      "expected": "integer"
    },
    {
      "field": "store_state",
      "observed": "float64",
      "expected": "string"
    },
    {
      "field": "store_country",
      "observed": "float64",
      "expected": "string"
    },
    {
      "field": "avg_temperature_c",
      "observed": "float64",
      "expected": "integer"
    },
    {
      "field": "precipitation_mm",
      "observed": "float64",
      "expected": "number"
    },
    {
      "field": "unemployment_rate_monthly",
      "observed": "float64",
      "expected": "number"
    },
    {
      "field": "population_monthly",
      "observed": "float64",
      "expected": "integer"
    },
    {
      "field": "cpi_monthly",
      "observed": "float64",
      "expected": "integer"
    },
    {
      "field": "gdp_monthly",
      "observed": "float64",
      "expected": "integer"
    }
  ]
}
- **Semantic Similarity** – FAIL (score=0.347, threshold=0.5)
  - Details: {
  "mappings": [
    {
      "source": "transaction_date",
      "target": "date",
      "score": 0.5,
      "reasoning": "Direct temporal field match; transaction_date clearly records the date of the transaction."
    },
    {
      "source": "prod_code",
      "target": "product_id",
      "score": 0.0,
      "reasoning": "prod_code is the unique product identifier in transactional retail datasets, matching product_id."
    },
    {
      "source": "store_label",
      "target": "store_id",
      "score": 0.333,
      "reasoning": "store_label is the unique identifier for the store, semantically synonymous with store_id."
    },
    {
      "source": "unit_price",
      "target": "unit_net_price",
      "score": 0.667,
      "reasoning": "unit_price is the price per unit; assumed to be net as discount is shown separately."
    },
    {
      "source": "qty_sold",
      "target": "units_sold",
      "score": 0.333,
      "reasoning": "qty_sold is a standard retail synonym for units_sold."
    },
    {
      "source": "qty_ordered",
      "target": "units_ordered",
      "score": 0.333,
      "reasoning": "qty_ordered (number ordered from supplier or DC) matches business logic for units_ordered."
    },
    {
      "source": "stock_level",
      "target": "beginning_inventory",
      "score": 0.0,
      "reasoning": "stock_level reflects inventory at a given point, most likely beginning inventory in a daily sales view."
    },
    {
      "source": "promo_flag",
      "target": "promotion_active",
      "score": 0.0,
      "reasoning": "promo_flag indicates whether a promotion is currently running (active)."
    },
    {
      "source": "discount_percent",
      "target": "promotional_price",
      "score": 0.0,
      "reasoning": "discount_percent could be used to derive promotional price, but is not a direct field; included at low confidence as it may signal promotion intensity if business rules apply."
    },
    {
      "source": "season",
      "target": "seasonal_peak_months",
      "score": 0.0,
      "reasoning": "season gives the current seasonal context, which can relate to seasonal peaks (though not by month)."
    },
    {
      "source": "prod_code",
      "target": "product_id",
      "score": 0.0,
      "reasoning": "prod_code is the product identifier and matches semantic meaning of product_id."
    },
    {
      "source": "item_category",
      "target": "category",
      "score": 0.5,
      "reasoning": "item_category is a direct match to the general category of a product."
    },
    {
      "source": "item_subcategory",
      "target": "subcategory",
      "score": 0.5,
      "reasoning": "item_subcategory gives further breakdown beneath the main category; direct semantic match."
    },
    {
      "source": "launch_date",
      "target": "product_launch_date",
      "score": 0.667,
      "reasoning": "launch_date reflects the product introduction date; matches product_launch_date."
    },
    {
      "source": "orig_price",
      "target": "unit_orig_price",
      "score": 0.667,
      "reasoning": "orig_price is the original unit price of the product, matching unit_orig_price."
    },
    {
      "source": "store_label",
      "target": "store_id",
      "score": 0.333,
      "reasoning": "store_label is the unique identifier for a store, matching store_id."
    },
    {
      "source": "location_region",
      "target": "store_region",
      "score": 0.333,
      "reasoning": "location_region is likely equivalent to store_region in target schema."
    },
    {
      "source": "city_name",
      "target": "store_city",
      "score": 0.333,
      "reasoning": "city_name directly maps to store_city."
    },
    {
      "source": "floor_area",
      "target": "store_size_sqft",
      "score": 0.0,
      "reasoning": "floor_area assumed it is in sqft, representing store size as in target field."
    },
    {
      "source": "date",
      "target": "date",
      "score": 1.0,
      "reasoning": "Direct temporal field match."
    },
    {
      "source": "is_public_holiday",
      "target": "is_holiday",
      "score": 0.667,
      "reasoning": "is_public_holiday (0/1 flag) semantically matches is_holiday."
    },
    {
      "source": "holiday_name",
      "target": "holiday_name",
      "score": 1.0,
      "reasoning": "holiday_name is same concept in both schemas."
    },
    {
      "source": "prod_code",
      "target": "product_id",
      "score": 0.0,
      "reasoning": "prod_code matches business role of product_id."
    },
    {
      "source": "promo_start_date",
      "target": "promotion_start_date",
      "score": 0.5,
      "reasoning": "promo_start_date is the start of the promo, direct mapping."
    },
    {
      "source": "promo_end_date",
      "target": "promotion_end_date",
      "score": 0.5,
      "reasoning": "promo_end_date is the end date, direct mapping."
    },
    {
      "source": "promo_price",
      "target": "promotional_price",
      "score": 0.333,
      "reasoning": "promo_price is the price during promotion, direct mapping."
    },
    {
      "source": "orig_price",
      "target": "unit_orig_price",
      "score": 0.667,
      "reasoning": "orig_price is pre-promotion price, matches unit_orig_price."
    },
    {
      "source": "Date",
      "target": "date",
      "score": 1.0,
      "reasoning": "Date is the temporal reference for the weather period."
    },
    {
      "source": "Region",
      "target": "store_region",
      "score": 0.5,
      "reasoning": "Region can be aligned with store_region for weather context."
    },
    {
      "source": "Mean Temperature",
      "target": "avg_temperature_c",
      "score": 0.25,
      "reasoning": "Mean Temperature is a direct match for average temperature."
    },
    {
      "source": "Total Precipitation (mm)",
      "target": "precipitation_mm",
      "score": 0.667,
      "reasoning": "Total Precipitation (mm) is a direct match for precipitation_mm."
    },
    {
      "source": "REF_DATE",
      "target": "date",
      "score": 0.5,
      "reasoning": "REF_DATE is the standard StatsCan/retail monthly date/effective temporal marker."
    },
    {
      "source": "GEO",
      "target": "store_region",
      "score": 0.0,
      "reasoning": "GEO refers to geography; in retail often corresponds to region for economic indicators."
    },
    {
      "source": "VALUE",
      "target": "cpi_monthly",
      "score": 0.0,
      "reasoning": "VALUE provides the Consumer Price Index for the month (CPI-monthly)."
    },
    {
      "source": "REF_DATE",
      "target": "date",
      "score": 0.5,
      "reasoning": "REF_DATE is the temporal anchor (month) in labour dataset."
    },
    {
      "source": "GEO",
      "target": "store_region",
      "score": 0.0,
      "reasoning": "GEO as region information for economic indicators matches store_region logic."
    },
    {
      "source": "REF_DATE",
      "target": "date",
      "score": 0.5,
      "reasoning": "REF_DATE is temporal anchor for GDP value."
    },
    {
      "source": "GEO",
      "target": "store_region",
      "score": 0.0,
      "reasoning": "GEO for region; might be used for region-level aggregations."
    },
    {
      "source": "VALUE",
      "target": "gdp_monthly",
      "score": 0.0,
      "reasoning": "VALUE is GDP value for the given date and region."
    },
    {
      "source": "REF_DATE",
      "target": "date",
      "score": 0.5,
      "reasoning": "REF_DATE is the temporal reference."
    },
    {
      "source": "GEO",
      "target": "store_region",
      "score": 0.0,
      "reasoning": "GEO is a geographic region, analogous to store_region for context."
    },
    {
      "source": "VALUE",
      "target": "population_monthly",
      "score": 0.0,
      "reasoning": "VALUE is total population for month/region, used for population_monthly target field."
    }
  ],
  "evaluated": 42
}

## LLM Metrics
- **LLM Metrics** – skipped: DEEPEVAL_API_KEY not set; deterministic metrics still executed.

## Recommended Next Prompt

Deterministic evaluation flagged the following issues:
- Fix dtype alignment for: is_holiday, holiday_name, is_retail_holiday, school_holiday_flag, unit_net_price, promotional_price, promotion_active, unit_orig_price, is_seasonal_product, store_cluster
- Revisit semantic alignment for: prod_code→product_id (0.0), store_label→store_id (0.333), qty_sold→units_sold (0.333), qty_ordered→units_ordered (0.333), stock_level→beginning_inventory (0.0)
Regenerate mappings focusing on these corrections before the next run.