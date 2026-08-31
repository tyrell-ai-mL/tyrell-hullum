\# ☁️ NorthStar Commerce — AWS Serverless Sales Data Pipeline



A serverless AWS data engineering project that automatically ingests, cleans, transforms, catalogs, and analyzes e-commerce sales transaction data.



The project demonstrates an end-to-end cloud data pipeline using Amazon S3, AWS Lambda, AWS Glue, and Amazon Athena.



\## 🏗️ Architecture



Raw Sales CSV  

↓  

Amazon S3 — `raw-data/`  

↓  

S3 Event Trigger  

↓  

AWS Lambda — Python ETL  

↓  

Amazon S3 — `processed-data/`  

↓  

AWS Glue Crawler  

↓  

AWS Glue Data Catalog  

↓  

Amazon Athena  

↓  

SQL Analytics



\## 📊 Dataset



The project uses a synthetic dataset created for the fictional company \*\*NorthStar Commerce\*\*.



The raw dataset contains more than 1,000 e-commerce transactions with fields including:



\- Transaction ID

\- Transaction Date

\- Product

\- Product Category

\- Quantity

\- Unit Price

\- Customer Type

\- Region

\- Payment Method



The dataset intentionally contains data-quality issues including missing values, inconsistent capitalization, and a duplicate transaction so the ETL pipeline performs meaningful transformations.



\## ⚙️ AWS Services



\### Amazon S3



The S3 bucket is organized into four areas:



\- `raw-data/` — incoming transaction files

\- `processed-data/` — cleaned data produced by Lambda

\- `athena-results/` — Athena SQL query results

\- `archive/` — reserved for processed source-file archival



\### AWS Lambda



The `northstar-sales-etl` Lambda function automatically runs when a CSV file is uploaded to `raw-data/`.



The Python ETL process:



\- Reads the source CSV from S3

\- Removes duplicate transactions

\- Standardizes text fields

\- Handles missing customer and region values

\- Converts numeric fields

\- Calculates transaction revenue

\- Writes the cleaned dataset to `processed-data/`



The raw dataset contained \*\*1,001 rows\*\*. After duplicate removal, the processed dataset contained \*\*1,000 transactions\*\*.



\### AWS IAM



The Lambda execution role follows least-privilege principles.



The function is allowed to:



\- Read objects from `raw-data/`

\- Write objects to `processed-data/`



It is not granted unrestricted access to the entire S3 service.



\### AWS Glue



AWS Glue catalogs the processed dataset.



Components:



\- Database: `northstar\_sales\_db`

\- Crawler: `northstar-sales-crawler`

\- Catalog table: `processed\_data`



The crawler detected the CSV schema and created a 10-column table for analytics.



\### Amazon Athena



Amazon Athena provides serverless SQL analytics directly against the processed data stored in S3.



\## 📈 Example Analytics



\### Sales by Product Category



| Category | Units Sold | Total Sales |

|---|---:|---:|

| Electronics | 594 | $55,144.06 |

| Home | 615 | $40,648.85 |

| Office | 611 | $35,103.89 |

| Outdoors | 690 | $33,168.10 |

| Fitness | 601 | $31,253.99 |



Electronics generated the highest sales revenue at \*\*$55,144.06\*\*.



\### Sales by Region



| Region | Total Sales | Units Sold |

|---|---:|---:|

| Midwest | $44,161.99 | 701 |

| West | $43,132.57 | 643 |

| Southeast | $37,371.82 | 618 |

| Southwest | $36,472.11 | 589 |

| Northeast | $34,150.41 | 559 |

| Unknown | $29.99 | 1 |



The `Unknown` region demonstrates how the pipeline preserves and identifies incomplete source data for downstream data-quality analysis.



\## 💻 Technologies



\- AWS

\- Amazon S3

\- AWS Lambda

\- AWS Glue

\- Amazon Athena

\- AWS IAM

\- Python

\- SQL

\- Serverless Architecture

\- ETL

\- Data Engineering



\## 📁 Repository Files



`lambda\_function.py` — Python Lambda ETL function



`queries.sql` — Athena analytics queries



`README.md` — project documentation



\## 🎯 Skills Demonstrated



This project demonstrates hands-on experience with:



\- AWS serverless architecture

\- Event-driven data pipelines

\- Cloud storage

\- Python ETL development

\- Data cleaning and transformation

\- IAM permissions

\- AWS Glue Data Catalog

\- Serverless SQL analytics

\- Data-quality handling

\- End-to-end cloud data engineering



\## 🔐 Security



The S3 bucket is private with public access blocked.



The Lambda function uses a dedicated IAM execution role with permissions limited to the S3 paths required by the ETL workflow.



\## 📌 Project Status



\*\*Completed — August 2026\*\*



The end-to-end pipeline was successfully deployed and tested in AWS:



`S3 Upload → Lambda ETL → Processed S3 Data → Glue Catalog → Athena SQL Analytics`



\## 📸 AWS Pipeline Screenshots



\### Amazon S3 — Data Storage

The S3 bucket stores raw transaction data, processed output, Athena query results, and the reserved archive area.



!\[Amazon S3 Bucket Setup](screenshots/01-s3-bucket-setup.png)



\### AWS Lambda — Automated ETL Trigger

The `northstar-sales-etl` Lambda function is connected to Amazon S3 and automatically runs when a CSV file is uploaded to the `raw-data/` path.



!\[Lambda S3 Trigger](screenshots/02-lambda-s3-trigger.png)



\### Processed Data Output

After the Lambda ETL process runs, the cleaned dataset is written to the `processed-data/` path in Amazon S3.



!\[Processed S3 Output](screenshots/03-processed-s3-output.png)



\### AWS Glue Data Catalog

AWS Glue catalogs the processed dataset so it can be queried through Amazon Athena.



!\[AWS Glue Data Catalog](screenshots/04-glue-data-catalog.png)



\### Amazon Athena — Category Analysis

Athena SQL queries analyze sales performance across product categories.



!\[Athena Category Results](screenshots/05-athena-category-results.png)



\### Amazon Athena — Regional Analysis

Regional sales analysis demonstrates serverless SQL analytics directly against the processed S3 dataset.



!\[Athena Region Results](screenshots/06-athena-region-results.png)



\### End-to-End AWS Pipeline

Additional AWS console validation of the completed serverless data pipeline.



!\[AWS Pipeline Overview](screenshots/07-aws-pipeline-overview.png)

