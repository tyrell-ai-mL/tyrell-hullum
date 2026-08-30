from pyspark import pipelines as dp
from pyspark.sql.functions import col


# ============================================================
# BRONZE LAYER
# Incrementally ingest raw JSON sales-order data with Auto Loader
# ============================================================

@dp.table(
    name="orders_bronze",
    comment="Bronze layer: raw sales orders ingested from JSON files."
)
@dp.expect_or_drop(
    "valid_order_datetime",
    "order_datetime IS NOT NULL AND length(order_datetime) > 0"
)
def orders_bronze():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .load("/databricks-datasets/retail-org/sales_orders")
    )


# ============================================================
# SILVER LAYER
# Clean the bronze data and create a usable order date
# ============================================================

@dp.materialized_view(
    name="orders_silver",
    comment="Silver layer: cleaned sales-order data with order date."
)
def orders_silver():
    return (
        spark.read.table("orders_bronze")
        .withColumn(
            "order_date",
            col("order_datetime")
            .cast("int")
            .cast("timestamp")
            .cast("date")
        )
    )


# ============================================================
# GOLD LAYER
# Aggregate cleaned orders into daily customer metrics
# ============================================================

@dp.materialized_view(
    name="daily_customer_orders_gold",
    comment="Gold layer: daily order totals by customer."
)
def daily_customer_orders_gold():
    return (
        spark.read.table("orders_silver")
        .groupBy(
            "customer_id",
            "order_date"
        )
        .count()
        .withColumnRenamed(
            "count",
            "daily_order_count"
        )
    )
