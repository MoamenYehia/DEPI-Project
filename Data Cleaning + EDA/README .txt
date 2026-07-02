
Files 
----------------------------------------------------------------

1. cleaned_master_df.parquet  
2. cleaned_master_df.csv    
3. order_df.parquet          
4. order_df.csv               
5. gx_validation_results.json 
6. Cleaning_EDA.ipynb         

----------------------------------------------------------------
Cleaning Steps Applied
----------------------------------------------------------------

- Removed duplicate rows
- Removed canceled orders
- Removed rows with null product_id
- Filled missing values: payment_value=0, freight_value=0, review_score=median
- Type casting: date columns -> datetime | payment_value -> float
- Added delivery_days column (delivery date - purchase date)
- Filtered out negative delivery_days (logically invalid records)

----------------------------------------------------------------
Validation Rules (Great Expectations)
----------------------------------------------------------------

- order_id      -> must not be null
- product_id    -> must not be null
- payment_value -> must be >= 0
- review_score  -> must be between 1 and 5
- order_status  -> must be in [delivered, shipped, processing, approved]
- delivery_days -> must be between 0 and 365

----------------------------------------------------------------
EDA Visualizations (inside the Notebook)
----------------------------------------------------------------

1.  Top 10 Product Categories (by count)
2.  Top 10 Categories by Revenue
3.  Review Score vs Average Payment Value
4.  Top 10 Customer Cities
5.  Top States by Number of Customers
6.  Review Score vs Sales (Scatter Plot)
7.  Payment Value Distribution (Boxplot)
8.  Monthly Revenue Trend
9.  Delivery Days Distribution

================================================================
