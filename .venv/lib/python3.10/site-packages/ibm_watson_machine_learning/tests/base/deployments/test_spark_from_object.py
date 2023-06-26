#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

from ibm_watson_machine_learning.tests.base.abstract.abstract_online_deployment_test import AbstractOnlineDeploymentTest


class TestSparkDeployment(AbstractOnlineDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying Spark model
    using object.
    """
    deployment_type = "mllib_3.3"
    software_specification_name = "spark-mllib_3.3"
    model_name = deployment_name = "spark_model_from_object"
    IS_MODEL = True

    def get_model(self):
        spark = SparkSession.builder.getOrCreate()

        df_data = spark.read \
            .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat') \
            .option('header', 'true') \
            .option('inferSchema', 'true') \
            .load("Cloud/artifacts/GoSales_Tx_NaiveBayes.csv")

        splitted_data = df_data.randomSplit([0.8, 0.18, 0.02], 24)
        train_data = splitted_data[0]

        stringIndexer_label = StringIndexer(inputCol="PRODUCT_LINE", outputCol="label").fit(df_data)
        stringIndexer_prof = StringIndexer(inputCol="PROFESSION", outputCol="PROFESSION_IX")
        stringIndexer_gend = StringIndexer(inputCol="GENDER", outputCol="GENDER_IX")
        stringIndexer_mar = StringIndexer(inputCol="MARITAL_STATUS", outputCol="MARITAL_STATUS_IX")

        vectorAssembler_features = VectorAssembler(inputCols=["GENDER_IX", "AGE", "MARITAL_STATUS_IX", "PROFESSION_IX"],
                                                   outputCol="features")
        rf = RandomForestClassifier(labelCol="label", featuresCol="features")
        labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                       labels=stringIndexer_label.labels)
        pipeline_rf = Pipeline(stages=[stringIndexer_label, stringIndexer_prof, stringIndexer_gend, stringIndexer_mar,
                                       vectorAssembler_features, rf, labelConverter])
        model_rf = pipeline_rf.fit(train_data)

        TestSparkDeployment.pipeline = pipeline_rf
        TestSparkDeployment.training_data = train_data
        TestSparkDeployment.label = "PRODUCT_LINE"

        return model_rf

    def create_model_props(self):
        return {
            self.wml_client.repository.ModelMetaNames.NAME: self.model_name,
            self.wml_client.repository.ModelMetaNames.TYPE: self.deployment_type,
            self.wml_client.repository.ModelMetaNames.LABEL_FIELD: self.label,
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "fields": ["GENDER", "AGE", "MARITAL_STATUS", "PROFESSION"],
                "values": [["M", 23, "Single", "Student"], ["M", 55, "Single", "Executive"]]
            }]
        }

    def test_01_store_model(self):
        TestSparkDeployment.model = self.get_model()
        model_props = self.create_model_props()

        model_details = self.wml_client.repository.store_model(
            meta_props=model_props,
            model=self.model,
            pipeline=self.pipeline,
            training_data=self.training_data,
        )
        TestSparkDeployment.model_id = self.wml_client.repository.get_model_id(model_details)
        self.assertIsNotNone(self.model_id)


if __name__ == "__main__":
    unittest.main()
