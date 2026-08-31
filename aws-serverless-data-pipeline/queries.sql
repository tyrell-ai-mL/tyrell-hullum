-- NorthStar Commerce
-- AWS Serverless Sales Data Pipeline
-- Amazon Athena Analytics Queries


-- 1. Preview cleaned transaction data
SELECT *
FROM processed_data
LIMIT 10;


-- 2. Sales performance by product category
SELECT
    category,
    SUM(quantity) AS total_units_sold,
    ROUND(SUM(quantity * unit_price), 2) AS total_sales
FROM processed_data
GROUP BY category
ORDER BY total_sales DESC;


-- 3. Sales performance by region
SELECT
    region,
    ROUND(SUM(quantity * unit_price), 2) AS total_sales,
    SUM(quantity) AS units_sold
FROM processed_data
GROUP BY region
ORDER BY total_sales DESC;