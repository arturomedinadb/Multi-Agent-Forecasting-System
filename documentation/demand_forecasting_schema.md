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
| `unit_net_price` | DECIMAL(8,2) | Price per unit actually charged for the day |
| `units_sold` | INTEGER | Total units sold during the day |

### Pricing & Promotion Information
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `promotional_price` | DECIMAL(8,2) | Promotional price (if on promotion) |
| `promotion_active` | BOOLEAN | Indicates if product was on promotion |
| `promotion_start_date` | DATE | When current promotion started |
| `promotion_end_date` | DATE | When current promotion ends |

### Product Master Information
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `category` | VARCHAR(50) | Top-level product category |
| `subcategory` | VARCHAR(50) | Detailed subcategory |
| `product_launch_date` | DATE | When product was first introduced |
| `unit_orig_price` | DECIMAL(8,2) | Original pricing of the product |
| `is_seasonal_product` | BOOLEAN | Indicates seasonal demand patterns |
| `seasonal_peak_months` | VARCHAR(50) | Peak selling months (if seasonal) |


### Store Master Information
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `store_cluster` | VARCHAR(30) | Store type (flagship, outlet, mall, strip-center) |
| `store_size_sqft` | INTEGER | Store floor space in square feet |
| `store_city` | VARCHAR(100) | Store city |
| `store_state` | VARCHAR(50) | State or province |
| `store_country` | VARCHAR(50) | Country |
| `store_region` | VARCHAR(50) | Regional classification |

### Daily Inventory & Operations
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `beginning_inventory` | INTEGER | Units in stock at start of day |
| `units_ordered` | INTEGER | Units received from other stores |

### Weather & Environmental Data
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `avg_temperature_c` | INTEGER | Average monthly temperature |
| `precipitation_inches` | DECIMAL(4,2) | Total monthly precipitation |
| `severe_weather_flag` | BOOLEAN | Severe weather event occurred |

### Economic & Market Context
| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `unemployment_rate_month` | DECIMAL(4,2) | Local unemployment percentage of the transaction month |
| `population_month` | INTEGER | Total population amount of the transaction month |
| `cpi_month` | INTEGER | CPI of the transaction month |
