# Databricks notebook source
!pip install /Volumes/mlops_prod/house_prices/package/house_price-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import yaml
import logging
from pyspark.sql import SparkSession

from house_price.data_processor import DataProcessor
from house_price.config import ProjectConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path="/Volumes/mlops_prod/house_prices/data/project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    "/Volumes/mlops_prod/house_prices/data/data.csv",
    header=True,
    inferSchema=True).toPandas()

# COMMAND ----------

# Initialize DataProcessor
data_processor = DataProcessor(df, config)

# Preprocess the data
data_processor.preprocess()

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()


# COMMAND ----------


logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

data_processor.save_to_catalog(X_train, X_test, spark)
