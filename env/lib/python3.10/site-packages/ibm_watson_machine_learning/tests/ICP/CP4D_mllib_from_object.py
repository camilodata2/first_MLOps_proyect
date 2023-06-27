#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import os
import sys,time
from os.path import join as path_join

# SPARK_HOME_PATH = os.environ['SPARK_HOME']
# PYSPARK_PATH = str(SPARK_HOME_PATH) + "/python/"
# sys.path.insert(1, path_join(PYSPARK_PATH))

import logging
from preparation_and_cleaning import *
from models_preparation import *


class TestWMLClientWithSpark(unittest.TestCase):
    deployment_uid = None
    space_uid = None
    space_href = None
    model_uid = None
    scoring_url = None
    space_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSpark.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.model_name = "SparkMLlibFromObjectLocal Model"
        self.deployment_name = "Test deployment"
        self.space_id = None

    def test_01_set_space(self):
        self.space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "space_4_spark_model"})

        TestWMLClientWithSpark.space_id = self.client.spaces.get_uid(self.space)
        self.client.set.default_space(TestWMLClientWithSpark.space_id)
        self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithSpark.space_id))


    def test_02_publish_model(self):
        TestWMLClientWithSpark.logger.info("Creating spark model ...")

        model_data = create_spark_mllib_model_data()

        TestWMLClientWithSpark.logger.info("Publishing spark model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "Spark",
            self.client.repository.ModelMetaNames.TYPE: "mllib_2.4",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "spark-mllib_2.4",
                       }

        published_model = self.client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], pipeline=model_data['pipeline'])
        TestWMLClientWithSpark.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithSpark.logger.info("Published model ID:" + str(TestWMLClientWithSpark.model_uid))
        self.assertIsNotNone(TestWMLClientWithSpark.model_uid)

    def test_04_get_details(self):
        TestWMLClientWithSpark.logger.info("Get details")
        details = self.client.repository.get_details(self.model_uid)
        print(details)
        TestWMLClientWithSpark.logger.debug("Model details: " + str(details))
        self.assertTrue("Spark" in str(details))

    def test_05_create_deployment(self):
        TestWMLClientWithSpark.logger.info("Create deployment")
        deployment = self.client.deployments.create(self.model_uid, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE: {}})


        TestWMLClientWithSpark.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithSpark.logger.debug("Online deployment: " + str(deployment))
        TestWMLClientWithSpark.scoring_url = self.client.deployments.get_scoring_href(deployment)
        TestWMLClientWithSpark.logger.debug("Scoring url: {}".format(TestWMLClientWithSpark.scoring_url))
        TestWMLClientWithSpark.deployment_uid = self.client.deployments.get_uid(deployment)
        TestWMLClientWithSpark.logger.debug("Deployment uid: {}".format(TestWMLClientWithSpark.deployment_uid))
        self.assertTrue("online" in str(deployment))

    def test_06_get_deployment_details(self):
        TestWMLClientWithSpark.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details(TestWMLClientWithSpark.deployment_uid)
        print(deployment_details)
        TestWMLClientWithSpark.logger.debug("Deployment details: {}".format(deployment_details))
        self.assertTrue(self.deployment_name in str(deployment_details))

    def test_07_score(self):
        TestWMLClientWithSpark.logger.info("Score the model")
        scoring_data = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    "fields": ["GENDER","AGE","MARITAL_STATUS","PROFESSION"],
                    "values": [["M",23,"Single","Student"],["M",55,"Single","Executive"]]
                }
            ]
        }
        predictions = self.client.deployments.score(TestWMLClientWithSpark.deployment_uid, scoring_data)
        print("Predictions: {}".format(predictions))
        self.assertTrue("prediction" in str(predictions))

    def test_08_delete_deployment(self):
        TestWMLClientWithSpark.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithSpark.deployment_uid)

    def test_09_delete_model(self):
        TestWMLClientWithSpark.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithSpark.model_uid)

    def test_10_delete_space(self):
        TestWMLClientWithSpark.logger.info("Delete spaces")
        self.client.spaces.delete(TestWMLClientWithSpark.space_id)


if __name__ == '__main__':
    unittest.main()
