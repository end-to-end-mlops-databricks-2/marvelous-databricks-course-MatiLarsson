from loguru import logger

from house_price.config import ProjectConfig
from house_price.serving.fe_model_serving import FeatureLookupServing

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "house-prices-model-serving-fe"

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name="house_prices_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.house_features",
)

# Create the online table for house features
feature_model_server.create_online_table()
logger.info("Created online table")

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint")
