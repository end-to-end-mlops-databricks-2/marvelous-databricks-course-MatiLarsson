# Databricks notebook source

# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/house_prices/package/house_price-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import hashlib
import os
import time
from typing import Dict, List

import requests

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow.models import infer_signature
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from house_price.config import ProjectConfig, Tags
from house_price.models.basic_model import BasicModel

# COMMAND ----------
# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags = Tags(**{"git_sha": "abcd12345", "branch": "week3"})

os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------
# Train & register model A with the config path
basic_model_a = BasicModel(config=config, tags=tags, spark=spark)
basic_model_a.parameters = config.parameters_a
basic_model_a.model_name = f"{catalog_name}.{schema_name}.house_prices_model_basic_A"
basic_model_a.experiment_name = config.experiment_name_a
basic_model_a.load_data()
basic_model_a.prepare_features()
basic_model_a.train()
basic_model_a.log_model()
basic_model_a.register_model()
model_A = mlflow.sklearn.load_model(f"models:/{basic_model_a.model_name}@latest-model")

# COMMAND ----------
# Train & register model B with the config path
basic_model_b = BasicModel(config=config, tags=tags, spark=spark)
basic_model_b.parameters = config.parameters_b
basic_model_b.model_name = f"{catalog_name}.{schema_name}.house_prices_model_basic_B"
basic_model_a.experiment_name = config.experiment_name_b
basic_model_b.load_data()
basic_model_b.prepare_features()
basic_model_b.train()
basic_model_b.log_model()
basic_model_b.register_model()
model_B = mlflow.sklearn.load_model(f"models:/{basic_model_b.model_name}@latest-model")


# COMMAND ----------
class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context, model_input):
        house_id = str(model_input["Id"].values[0])
        hashed_id = hashlib.md5(house_id.encode(encoding="UTF-8")).hexdigest()
        # convert a hexadecimal (base-16) string into an integer
        if int(hashed_id, 16) % 2:
            predictions = self.model_a.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model A"}
        else:
            predictions = self.model_b.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model B"}


# COMMAND ----------
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
X_train = train_set[config.num_features + config.cat_features + ["Id"]]
X_test = test_set[config.num_features + config.cat_features + ["Id"]]


# COMMAND ----------
models = [model_A, model_B]
wrapped_model = HousePriceModelWrapper(models)  # we pass the loaded models to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/house-prices-ab-testing")
model_name = f"{catalog_name}.{schema_name}.house_prices_model_pyfunc_ab_test"

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 1234.5, "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model, artifact_path="pyfunc-house-price-model-ab", signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-house-price-model-ab", name=model_name, tags=tags.dict()
)

# COMMAND ----------
workspace = WorkspaceClient()
served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=model_version.version,
    )
]

endpoint_name = "house-price-model-ab"

workspace.serving_endpoints.create(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=served_entities,
    ),
)

# COMMAND ----------
# Call the endpoint with one sample record
def call_endpoint(record: List[Dict]):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

# COMMAND ----------

sampled_records = X_test.to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

status_code, response_text = call_endpoint(dataframe_records)
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# "load test"

for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
