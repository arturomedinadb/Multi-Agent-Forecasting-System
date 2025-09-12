# In-Store Demand Forecasting Schema

**Table Structure**: One row per date-product-store combination with all related information included.

### Primary Identifiers
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `date` | DATE | Business date (primary dimension) |
| `product_id` | VARCHAR(50) | Unique product identifier |
| `store_id` | VARCHAR(50) | Unique store identifier |

### Holiday & Special Events
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `is_holiday` | BOOLEAN | Federal/national holiday indicator |
| `holiday_name` | VARCHAR(50) | Name of specific holiday (if applicable) |
| `is_retail_holiday` | BOOLEAN | Major retail shopping day (Black Friday, etc.) |
| `school_holiday_flag` | BOOLEAN | School break/holiday indicator |

### Daily Sales Performance
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `unit_price` | DECIMAL(8,2) | Price for the day |
| `units_sold` | INTEGER | Total units sold during the day |
| `units_returned` | INTEGER | Units returned on this date |
| `return_value` | DECIMAL(10,2) | Dollar value of returned merchandise |

### Pricing & Promotion Information
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `regular_price` | DECIMAL(8,2) | Standard retail price for the day |
| `actual_selling_price` | DECIMAL(8,2) | Average price actually charged |
| `promotional_price` | DECIMAL(8,2) | Promotional price (if on promotion) |
| `promotion_active` | BOOLEAN | Indicates if product was on promotion |
| `promotion_type` | VARCHAR(30) | Type (clearance, seasonal, BOGO, percent-off, etc.) |
| `discount_percentage` | DECIMAL(5,2) | Percentage discount from regular price |
| `promotion_start_date` | DATE | When current promotion started |
| `promotion_end_date` | DATE | When current promotion ends |

### Product Master Information
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `brand` | VARCHAR(100) | Product brand name |
| `category` | VARCHAR(50) | Top-level product category |
| `subcategory` | VARCHAR(50) | Detailed subcategory |
| `product_type` | VARCHAR(30) | Classification (regular, seasonal, limited-edition) |
| `product_launch_date` | DATE | When product was first introduced |
| `product_lifecycle_stage` | VARCHAR(20) | Current stage (new, growth, mature, declining) |
| `is_seasonal_product` | BOOLEAN | Indicates seasonal demand patterns |
| `seasonal_peak_months` | VARCHAR(50) | Peak selling months (if seasonal) |
| `supplier_name` | VARCHAR(100) | Primary supplier name |

### Store Master Information
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `store_cluster` | VARCHAR(30) | Store type (flagship, outlet, mall, strip-center) |
| `store_size_sqft` | INTEGER | Store floor space in square feet |
| `store_city` | VARCHAR(100) | Store city |
| `store_state` | VARCHAR(50) | State or province |
| `store_country` | VARCHAR(50) | Country |
| `store_region` | VARCHAR(50) | Regional classification |
| `store_district` | VARCHAR(50) | District or area classification |
| `parking_availability` | VARCHAR(20) | Parking situation (ample, limited, street) |

### Daily Inventory & Operations
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `beginning_inventory` | INTEGER | Units in stock at start of day |
| `ending_inventory` | INTEGER | Units in stock at end of day |
| `units_received` | INTEGER | New inventory received during the day |
| `units_transferred_out` | INTEGER | Units transferred to other stores |
| `units_transferred_in` | INTEGER | Units received from other stores |
| `inventory_adjustments` | INTEGER | Adjustments (damage, theft, counting errors) |
| `stockout_occurred` | BOOLEAN | Indicates if product was out of stock |
| `stockout_duration_hours` | DECIMAL(4,1) | Hours the product was unavailable |
| `days_of_supply` | INTEGER | Days of inventory remaining at current sales rate |
| `inventory_turnover_velocity` | DECIMAL(6,3) | Daily inventory turn rate |

### Weather & Environmental Data
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `temperature_average_f` | INTEGER | Average daily temperature |
| `precipitation_inches` | DECIMAL(4,2) | Total daily precipitation |
| `severe_weather_flag` | BOOLEAN | Severe weather event occurred |

### Economic & Market Context
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `consumer_confidence_index` | DECIMAL(6,2) | Regional consumer confidence level |
| `unemployment_rate` | DECIMAL(4,2) | Local unemployment percentage |
| `gas_price_average` | DECIMAL(4,2) | Local average gas price per gallon |
| `economic_conditions` | VARCHAR(20) | General economic climate (strong, stable, weak) |
| `market_share_category` | DECIMAL(5,2) | Store's category market share percentage |
