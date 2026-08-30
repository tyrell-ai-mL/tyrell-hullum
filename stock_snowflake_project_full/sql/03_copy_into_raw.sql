-- COPY data from the STOCK_STAGE into RAW_STOCK_PRICES
-- Make sure you have uploaded a CSV file (like data/sample_prices.csv)
-- to @STOCK_STAGE before running this.

COPY INTO RAW_STOCK_PRICES
FROM @STOCK_STAGE
FILE_FORMAT = STOCK_CSV_FORMAT
ON_ERROR = 'CONTINUE';
