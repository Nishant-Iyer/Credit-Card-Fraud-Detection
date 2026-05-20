import os
import sys
import logging
from typing import Iterator
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define schema for output
output_schema = StructType([
    StructField("fraud_probability", DoubleType(), True),
    StructField("is_fraud", IntegerType(), True)
])

def run_spark_batch_inference(
    input_csv: str, 
    output_parquet: str, 
    pipeline_path: str = "artifacts/full_pipeline.joblib",
    metadata_path: str = "artifacts/metadata.json"
):
    """
    Initializes a PySpark session and runs distributed batch inference using Pandas UDF.
    """
    logger.info("Initializing PySpark session...")
    spark = SparkSession.builder \
        .appName("CreditCardFraudBatchInference") \
        .master("local[*]") \
        .getOrCreate()
        
    logger.info(f"Loading input data into Spark DataFrame from {input_csv}...")
    # Load input data
    df = spark.read.csv(input_csv, header=True, inferSchema=True)
    
    # Broadcast the path to our pipeline so executors can access it
    # Note: In a real cluster, the model would be loaded from S3/HDFS.
    absolute_pipeline_path = os.path.abspath(pipeline_path)
    absolute_metadata_path = os.path.abspath(metadata_path)
    
    # Define Pandas UDF for vectorized inference
    @pandas_udf(output_schema)
    def predict_fraud_udf(batch_iter: Iterator[tuple]) -> Iterator[pd.DataFrame]:
        # Load predictor once per executor process
        from src.pipeline.inference_pipeline import FraudPredictor
        predictor = FraudPredictor(
            pipeline_path=absolute_pipeline_path, 
            metadata_path=absolute_metadata_path
        )
        
        for batch in batch_iter:
            # batch is a tuple of pandas Series. Reconstruct DataFrame with names.
            df_batch = pd.concat(batch, axis=1)
            df_batch.columns = feature_cols
            
            # Predict
            preds, probs = predictor.predict(df_batch)
            yield pd.DataFrame({
                "fraud_probability": probs,
                "is_fraud": preds
            })
            
    # Select feature columns in order
    feature_cols = [col for col in df.columns if col not in ["Class"]]
    
    logger.info("Running distributed inference via Pandas UDF...")
    # Call Pandas UDF on feature columns struct
    predictions_df = df.select(
        "*", 
        predict_fraud_udf(*[df[c] for c in feature_cols]).alias("predictions")
    )
    
    # Flatten the result columns
    final_df = predictions_df.select(
        "*", 
        "predictions.fraud_probability", 
        "predictions.is_fraud"
    ).drop("predictions")
    
    logger.info(f"Saving predictions to {output_parquet}...")
    final_df.write.mode("overwrite").parquet(output_parquet)
    logger.info("Spark batch inference finished successfully.")
    
    # Stop session
    spark.stop()

if __name__ == "__main__":
    # Test batch inference if arguments are passed
    if len(sys.argv) > 2:
        run_spark_batch_inference(sys.argv[1], sys.argv[2])
    else:
        # Default test locations
        run_spark_batch_inference("data/test.csv", "data/predictions_parquet")
