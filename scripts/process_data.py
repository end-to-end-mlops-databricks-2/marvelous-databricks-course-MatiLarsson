# Databricks notebook source
import pandas as pd
import yaml
from databricks.connect import DatabricksSession

# from house_price.price_model import PriceModel
from house_price.config import ProjectConfig
from house_price.data_processor import DataProcessor

spark = DatabricksSession.builder.profile("<profile-name>").getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml")

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataProcessor
df = pd.read_csv("data/data.csv")
data_processor = DataProcessor(df, config)

# Preprocess the data
data_processor.preprocess()

# COMMAND ----------
# Split the data
X_train, X_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

data_processor.save_to_catalog(X_train, X_test, spark)
