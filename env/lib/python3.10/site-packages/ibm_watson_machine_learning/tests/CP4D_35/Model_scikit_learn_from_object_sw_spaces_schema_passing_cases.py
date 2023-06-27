#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import svm
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.CP4D_35.models_preparation import *

input_schema = {
            "id": "test1",
            "type": "list",
            "fields": [{
                "name": "id",
                "type": "double",
                "nullable": True,
            }]
        }

class TestWMLClientWithScikitLearn(unittest.TestCase):
    deployment_id = None
    model_id = None
    scoring_url = None
    space_id = None
    sw_spec_id = None
    model_uids = []
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithScikitLearn.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_data = create_scikit_learn_model_data()


        self.space_name = str(uuid.uuid4())



    def test_00_set_space(self):
        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
            self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        print(self.space)

        self.client.spaces.list()

        TestWMLClientWithScikitLearn.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientWithScikitLearn.space_id)

        self.client.set.default_space(TestWMLClientWithScikitLearn.space_id)
        # self.client.set.default_space('5fad8290-9c49-4403-b5f1-0cef4e061e00')
        self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithScikitLearn.space_id))

    def test_00b_prepared_sw_spec(self):
        TestWMLClientWithScikitLearn.sw_spec_id = self.client.software_specifications.get_uid_by_name("runtime-22.1-py3.9") # autoai-kb_rt22.1-py3.9

    def test_01_publish_model_with_input_schema_passed(self):
        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_1.0",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id,
            self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_schema
                       }

        published_model_details = self.client.repository.store_model(model=self.model_data['model'], meta_props=model_props, training_target=self.model_data['training_target'])
        print("TEST", published_model_details)
        TestWMLClientWithScikitLearn.model_uids.append(self.client.repository.get_model_id(published_model_details))

    def test_02_publish_model_with_training_data_refs_passed(self):
        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
                       self.client.repository.ModelMetaNames.TYPE: "scikit-learn_1.0",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id
                       }

        published_model_details = self.client.repository.store_model(model=self.model_data['model'],
                                                                     meta_props=model_props,
                                                                     training_data=self.model_data['training_data'],
                                                                     training_target=self.model_data['training_target'])
        print("TEST", published_model_details)
        self.assertTrue(published_model_details['entity']['schemas']['input'] is not None)
        TestWMLClientWithScikitLearn.model_uids.append(self.client.repository.get_model_id(published_model_details))

    def test_03_publish_model_with_input_schema_and_training_data_refs_passed(self):
        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_1.0",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id,
            self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_schema
                       }

        published_model_details = self.client.repository.store_model(model=self.model_data['model'], meta_props=model_props, training_data=self.model_data['training_data'], training_target=self.model_data['training_target'])
        print("TEST", published_model_details)
        self.assertTrue(published_model_details['entity']['schemas']['input'] is not None)
        self.assertTrue(published_model_details['entity']['training_data_references'] is not None)
        TestWMLClientWithScikitLearn.model_uids.append(self.client.repository.get_model_id(published_model_details))

    def test_04_publish_model_with_none_passed(self):
        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_1.0",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id
                       }

        #with self.assertRaises(Exception):
        published_model_details = self.client.repository.store_model(model=self.model_data['model'], meta_props=model_props, training_target=self.model_data['training_target'])
        print("TEST", published_model_details)

    def test_11_delete_models(self):
        TestWMLClientWithScikitLearn.logger.info("Delete model")
        for id in self.model_uids:
            self.client.repository.delete(id)

    def test_12_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithScikitLearn.space_id)


if __name__ == '__main__':
    unittest.main()