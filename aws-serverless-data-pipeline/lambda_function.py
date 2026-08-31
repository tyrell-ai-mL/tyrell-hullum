import boto3
import csv
import io
import urllib.parse

s3 = boto3.client("s3")


def lambda_handler(event, context):
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(
        event["Records"][0]["s3"]["object"]["key"]
    )

    # Only process CSV files uploaded to raw-data/
    if not key.startswith("raw-data/") or not key.endswith(".csv"):
        return {
            "statusCode": 200,
            "message": "File ignored"
        }

    response = s3.get_object(Bucket=bucket, Key=key)
    csv_text = response["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(csv_text))

    cleaned_rows = []
    seen_transactions = set()

    for row in reader:
        transaction_id = row["transaction_id"].strip()

        # Remove duplicate transactions
        if transaction_id in seen_transactions:
            continue

        seen_transactions.add(transaction_id)

        # Standardize text fields
        row["transaction_id"] = transaction_id
        row["product"] = row["product"].strip()
        row["category"] = row["category"].strip().title()
        row["customer_type"] = row["customer_type"].strip().title()
        row["region"] = row["region"].strip().title()
        row["payment_method"] = row["payment_method"].strip().title()

        # Handle missing values
        if not row["customer_type"]:
            row["customer_type"] = "Unknown"

        if not row["region"]:
            row["region"] = "Unknown"

        # Convert numeric fields
        quantity = int(row["quantity"])
        unit_price = float(row["unit_price"])

        row["quantity"] = quantity
        row["unit_price"] = round(unit_price, 2)

        # Calculate transaction revenue
        row["total_sale"] = round(quantity * unit_price, 2)

        cleaned_rows.append(row)

    output = io.StringIO()

    fieldnames = [
        "transaction_id",
        "transaction_date",
        "product",
        "category",
        "quantity",
        "unit_price",
        "customer_type",
        "region",
        "payment_method",
        "total_sale"
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cleaned_rows)

    output_key = "processed-data/northstar_sales_clean.csv"

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=output.getvalue(),
        ContentType="text/csv"
    )

    return {
        "statusCode": 200,
        "input_file": key,
        "output_file": output_key,
        "rows_processed": len(cleaned_rows)
    }