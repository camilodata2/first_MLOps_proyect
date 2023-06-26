#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import os

# SPARK_HOME_PATH = os.environ['SPARK_HOME']
# PYSPARK_PATH = str(SPARK_HOME_PATH) + "/python/"
# sys.path.insert(1, path_join(PYSPARK_PATH))

import logging
from preparation_and_cleaning import *
from models_preparation import *


class TestWMLClientWithSpark(unittest.TestCase):
    model_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSpark.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.model_name = "SparkMLlibFromObjectLocal Model"
        self.deployment_name = "Test deployment"
        self.project_id = os.environ['PROJECT_ID']
        self.client.set.default_project(self.project_id)

    def test_03_publish_model(self):
        TestWMLClientWithSpark.logger.info("Creating spark model ...")

        model_data = create_spark_mllib_model_data()

        TestWMLClientWithSpark.logger.info("Publishing spark model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "Spark",
            self.client.repository.ModelMetaNames.TYPE: "mllib_2.3",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "spark-mllib_2.3",
            self.client.repository.ModelMetaNames.SPACE_UID: TestWMLClientWithSpark.space_uid
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

    def test_09_delete_model(self):
        TestWMLClientWithSpark.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithSpark.model_uid)


if __name__ == '__main__':
    unittest.main()
