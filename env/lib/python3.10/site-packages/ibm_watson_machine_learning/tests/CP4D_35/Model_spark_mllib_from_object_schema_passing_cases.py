#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
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

from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.CP4D_35.models_preparation import *
from wml_client_error import WMLClientError

input_schema = {
            "id": "test1",
            "type": "list",
            "fields": [{
                "name": "id",
                "type": "double",
                "nullable": True,
            }]
        }

class TestWMLClientWithSpark(unittest.TestCase):
    deployment_uid = None
    space_uid = None
    space_href = None
    model_uid = None
    scoring_url = None
    space_id = None
    model_data = None
    sw_spec_id = None
    model_uids = []
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSpark.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.cos_credentials = get_cos_credentials()
        self.client = get_client()
        self.model_data = create_scikit_learn_model_data()


        self.space_name = str(uuid.uuid4())



        self.model_name = "SparkMLlibFromObjectLocal Model"
        self.deployment_name = "Test deployment"
        self.space_id = None

    def test_01_set_space(self):
        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
            self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        if not self.client.ICP:
            metadata[self.client.spaces.ConfigurationMetaNames.STORAGE] = {'type': 'bmcos_object_storage',
                                                                           'resource_crn': self.cos_credentials['resource_instance_id']}

        self.space = self.client.spaces.store(meta_props=metadata)

        print(self.space)

        self.client.spaces.list()

        TestWMLClientWithSpark.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientWithSpark.space_id)

        self.client.set.default_space(TestWMLClientWithSpark.space_id)
        # self.client.set.default_space('5fad8290-9c49-4403-b5f1-0cef4e061e00')
        self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithSpark.space_id))

    def test_01b_generate_model(self):
        TestWMLClientWithSpark.model_data = create_spark_mllib_model_data()
        TestWMLClientWithSpark.sw_spec_id = self.client.software_specifications.get_id_by_name('spark-mllib_3.2')

    def test_02_publish_model_input_schema_passed(self):
        model_props = {self.client.repository.ModelMetaNames.NAME: "Spark",
            self.client.repository.ModelMetaNames.TYPE: "mllib_3.2",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id,
            self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_schema
        }

        with self.assertRaises(Exception):
            self.client.repository.store_model(model=self.model_data['model'], meta_props=model_props, pipeline=self.model_data['pipeline'])

    def test_03_publish_model_training_data_passed(self):
        model_props = {self.client.repository.ModelMetaNames.NAME: "Spark",
            self.client.repository.ModelMetaNames.TYPE: "mllib_3.2",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id
        }

        published_model = self.client.repository.store_model(model=self.model_data['model'], meta_props=model_props, training_data=self.model_data['training_data'], pipeline=self.model_data['pipeline'])
        print("TEST", published_model)
        self.assertTrue(published_model['entity']['schemas']['input'] is not None)
        self.assertTrue(published_model['entity']['training_data_references'] is not None)
        self.model_uids.append(self.client.repository.get_model_id(published_model))

    def test_04_publish_model_input_schema_and_training_data_passed(self):
        model_props = {self.client.repository.ModelMetaNames.NAME: "Spark",
            self.client.repository.ModelMetaNames.TYPE: "mllib_3.2",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id,
            self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_schema
        }

        published_model = self.client.repository.store_model(model=self.model_data['model'], meta_props=model_props, training_data=self.model_data['training_data'], pipeline=self.model_data['pipeline'])
        print("TEST", published_model)
        self.assertTrue(published_model['entity']['schemas']['input'] is not None)
        self.assertTrue(published_model['entity']['training_data_references'] is not None)
        self.model_uids.append(self.client.repository.get_model_id(published_model))

    def test_05_publish_model_none_passed(self):
        model_props = {self.client.repository.ModelMetaNames.NAME: "Spark",
            self.client.repository.ModelMetaNames.TYPE: "mllib_3.2",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id
        }

        with self.assertRaises(Exception):
            self.client.repository.store_model(model=self.model_data['model'], meta_props=model_props, pipeline=self.model_data['pipeline'])

    def test_09_delete_model(self):
        TestWMLClientWithSpark.logger.info("Delete models")
        for id in self.model_uids:
            self.client.repository.delete(id)

    def test_10_delete_space(self):
        TestWMLClientWithSpark.logger.info("Delete spaces")
        self.client.spaces.delete(TestWMLClientWithSpark.space_id)


if __name__ == '__main__':
    unittest.main()
