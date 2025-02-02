# Databricks notebook source
import yaml
from house_price.data_processor import DataProcessor
from house_price.price_model import PriceModel
from house_price.config import ProjectConfig

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataProcessor
data_processor = DataProcessor("../data/data.csv", config)

# Preprocess the data
data_processor.preprocess_data()

# COMMAND ----------
# Split the data
X_train, X_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

data_processor.save_to_catalog(X_train, X_test)

