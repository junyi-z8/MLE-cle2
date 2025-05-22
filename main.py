import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")
import os
import pyspark.sql.functions as F

raw_base    = "data/raw"               # raw
bronze_base = "datamart/bronze"        # Bronze


tables = {
    "feature_clickstream":         "feature_clickstream.csv",
    "features_attributes": "features_attributes.csv",
    "features_financials": "features_financials.csv",
    "lms_loan_daily":     "lms_loan_daily.csv",
}
for tbl_name, filename in tables.items():
    # 1) find the raw base path
    raw_path = os.path.join(raw_base, filename)
    # 2) autumatically read the format
    if filename.endswith(".csv"):
        df = spark.read \
                  .option("header", "true") \
                  .option("inferSchema", "true") \
                  .csv(raw_path)
    elif filename.endswith(".json"):
        df = spark.read.json(raw_path)
    elif filename.endswith(".parquet"):
        df = spark.read.parquet(raw_path)
    else:
        raise ValueError(f"not supported format：{filename}")

    # 3) add a current timestamp for each table
    df = df.withColumn("ingest_time", F.current_timestamp())

    # 4) write to bronze table
    out_path = os.path.join(bronze_base, tbl_name)
    df.write.mode("overwrite").parquet(out_path)

    print(f"[Bronze] is successfully writting {tbl_name} to {out_path}")
import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, to_date, regexp_replace, when, row_number
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.window import Window

# —— initialize SparkSession ——
spark = SparkSession.builder \
    .appName("data_pipeline") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# —— set the cata ——
bronze_base = "datamart/bronze"
silver_base = "datamart/silver"
gold_base   = "datamart/gold"

# —— 1. Silver cleaning & transforming ——

# 1.1 Clickstream feature
click_df = spark.read.parquet(os.path.join(bronze_base, "feature_clickstream"))
for c in click_df.columns:
    if c.startswith("fe_"):
        click_df = click_df.withColumn(c, col(c).cast(IntegerType()))
click_df = click_df.na.fill({c: 0 for c in click_df.columns if c.startswith("fe_")})
click_df.write.mode("overwrite").parquet(os.path.join(silver_base, "feature_clickstream"))

# 1.2 attribute
attr_df = spark.read.parquet(os.path.join(bronze_base, "features_attributes"))
attr_df = attr_df.withColumn("snapshot_date", to_date("snapshot_date", "yyyy/M/d"))
attr_df = attr_df.withColumn("Age", regexp_replace("Age", "[^0-9]", "").cast(IntegerType()))
attr_df = attr_df.withColumn(
    "SSN",
    when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), col("SSN")).otherwise(None)
)
attr_df.write.mode("overwrite").parquet(os.path.join(silver_base, "features_attributes"))

# 1.3 financial
fin_df = spark.read.parquet(os.path.join(bronze_base, "features_financials"))
fin_df = fin_df.withColumn("snapshot_date", to_date("snapshot_date", "yyyy/M/d"))
for c in [
    "Customer_Annual_Inc","Monthly_IR","Num_Bank_Accounts","Num_Credit_Ca",
    "Interest_Rate","Num_of_Loan","Delay_from_due_date","Num_of_Delayed_Payment",
    "Changed_Cred","Num_Credit_In","Outstanding_D","Credit_Utilizati",
    "Total_EMI_per_M","Amount_invest"
]:
    if c in fin_df.columns:
        fin_df = fin_df.withColumn(c, col(c).cast(FloatType()))
fin_df = fin_df.withColumn(
    "Credit_Mix",
    when(col("Credit_Mix").isin("", "_"), None).otherwise(col("Credit_Mix"))
)
fin_df.write.mode("overwrite").parquet(os.path.join(silver_base, "features_financials"))

# 1.4 loan
loan_df = spark.read.parquet(os.path.join(bronze_base, "lms_loan_daily"))
loan_df = loan_df \
    .withColumn("loan_start_date", to_date("loan_start_date", "yyyy/M/d")) \
    .withColumn("snapshot_date",   to_date("snapshot_date",   "yyyy/M/d"))
for c in ["loan_amt","due_amt","paid_amt","overdue_amt","balance"]:
    loan_df = loan_df.withColumn(c, col(c).cast(FloatType()))
loan_df = loan_df.withColumn("dpd_flag", when(col("overdue_amt") > 0, 1).otherwise(0))
loan_df.write.mode("overwrite").parquet(os.path.join(silver_base, "lms_loan_daily"))

# —— 1.5 drop ingest_time field ——  
attr_df  = attr_df.drop("ingest_time")
fin_df   = fin_df.drop("ingest_time")
click_df = click_df.drop("ingest_time")
loan_df  = loan_df.drop("ingest_time")


# —— 2. Gold summary & concat ——

w_attr = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
attr_latest = (
    attr_df
    .withColumn("rn", row_number().over(w_attr))
    .filter("rn = 1")
    .drop("rn")
    .withColumnRenamed("snapshot_date", "attr_snapshot_date")
)

w_fin = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
fin_latest = (
    fin_df
    .withColumn("rn", row_number().over(w_fin))
    .filter("rn = 1")
    .drop("rn")
    .withColumnRenamed("snapshot_date", "fin_snapshot_date")
)

# 2.2 Loan 
loan_summary = loan_df.groupBy("Customer_ID").agg(
    F.count("*").alias("loan_records"),
    F.sum("dpd_flag").alias("total_dpd_count"),
    F.sum("loan_amt").alias("total_loan_amount"),
    F.max("snapshot_date").alias("last_loan_snapshot")
)

# 2.3 form the user profile
gold_df = (
    attr_latest
      .join(fin_latest,   "Customer_ID", "left")
      .join(click_df,     "Customer_ID", "left")
      .join(loan_summary, "Customer_ID", "left")
)

# write Gold level
gold_df.write.mode("overwrite") \
    .parquet(os.path.join(gold_base, "customer_profile"))

print("✅ Silver & Gold successfully build！")
