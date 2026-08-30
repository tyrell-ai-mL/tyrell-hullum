# Snowflake Stock Market Analytics Project

This project is an end-to-end example of a stock market analytics pipeline built on Snowflake.

## Components

- `sql/01_create_tables.sql` – Raw, clean, and analytics tables
- `sql/02_stage_and_file_format.sql` – Stage + file format for CSV loads
- `sql/03_copy_into_raw.sql` – COPY INTO commands for loading data
- `sql/04_task_refresh_layers.sql` – Task to refresh clean + metrics layers
- `ingestion/alpha_vantage_to_csv.py` – Example Python script to pull stock prices from Alpha Vantage and save as CSV
- `data/sample_prices.csv` – Example CSV you can upload to Snowflake stage

## High-Level Flow

1. Use `alpha_vantage_to_csv.py` to generate or update a CSV of stock prices (or replace with your own).
2. Upload the CSV to a Snowflake stage created in `02_stage_and_file_format.sql`.
3. Run `03_copy_into_raw.sql` to load data into `RAW_STOCK_PRICES`.
4. Run `04_task_refresh_layers.sql` to maintain `CLEAN_STOCK_PRICES` and `STOCK_DAILY_METRICS`.
5. Query `STOCK_DAILY_METRICS` from Snowsight or your BI tool for analytics and dashboards.
