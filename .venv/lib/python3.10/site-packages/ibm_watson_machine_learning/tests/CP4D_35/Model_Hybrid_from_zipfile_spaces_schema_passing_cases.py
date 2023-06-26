#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import time
import pandas as pd
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *

output_data_schema = [{'id': 'stest',
                       'type': 'list',
                       'fields': [{'name': 'age', 'type': 'float'},
                                  {'name': 'sex', 'type': 'float'},
                                  {'name': 'cp', 'type': 'float'},
                                  {'name': 'restbp', 'type': 'float'},
                                  {'name': 'chol', 'type': 'float'},
                                  {'name': 'fbs', 'type': 'float'},
                                  {'name': 'restecg', 'type': 'float'},
                                  {'name': 'thalach', 'type': 'float'},
                                  {'name': 'exang', 'type': 'float'},
                                  {'name': 'oldpeak', 'type': 'float'},
                                  {'name': 'slope', 'type': 'float'},
                                  {'name': 'ca', 'type': 'float'},
                                  {'name': 'thal', 'type': 'float'}]
                       }, {'id': 'teste2',
                           'type': 'test',
                           'fields': [{'name': 'age', 'type': 'float'},
                                      {'name': 'sex', 'type': 'float'},
                                      {'name': 'cp', 'type': 'float'},
                                      {'name': 'restbp', 'type': 'float'},
                                      {'name': 'chol', 'type': 'float'},
                                      {'name': 'fbs', 'type': 'float'},
                                      {'name': 'restecg', 'type': 'float'},
                                      {'name': 'thalach', 'type': 'float'},
                                      {'name': 'exang', 'type': 'float'},
                                      {'name': 'oldpeak', 'type': 'float'},
                                      {'name': 'slope', 'type': 'float'},
                                      {'name': 'ca', 'type': 'float'},
                                      {'name': 'thal', 'type': 'float'}]}]
training_data_ref = [{'connection': {'endpoint_url': '',
                                     'access_key_id': '',
                                     'secret_access_key': ''},
                      'location': {'bucket': '', 'path': ''},
                      'type': 'fs',
                      'schema': {'id': '4cdb0a0a-1c69-43a0-a8c0-3918afc7d45f',
                                 'fields': [{'metadata': {'name': 'AGE', 'scale': 0},
                                             'name': 'AGE',
                                             'nullable': True,
                                             'type': 'integer'},
                                            {'metadata': {'name': 'SEX', 'scale': 0},
                                             'name': 'SEX',
                                             'nullable': True,
                                             'type': 'string'},
                                            {'metadata': {'name': 'BP', 'scale': 0},
                                             'name': 'BP',
                                             'nullable': True,
                                             'type': 'string'},
                                            {'metadata': {'name': 'CHOLESTEROL', 'scale': 0},
                                             'name': 'CHOLESTEROL',
                                             'nullable': True,
                                             'type': 'string'},
                                            {'metadata': {'name': 'NA', 'scale': 6},
                                             'name': 'NA',
                                             'nullable': True,
                                             'type': 'decimal(12,6)'},
                                            {'metadata': {'name': 'K', 'scale': 6},
                                             'name': 'K',
                                             'nullable': True,
                                             'type': 'decimal(13,6)'}],
                                 'type': 'struct'}}]
input_data_schema = [{'id': 'auto_ai_kb_input_schema',
                      'fields': [{'name': 'AGE', 'type': 'int64'},
                                 {'name': 'SEX', 'type': 'object'},
                                 {'name': 'BP', 'type': 'object'},
                                 {'name': 'CHOLESTEROL', 'type': 'object'},
                                 {'name': 'NA', 'type': 'float64'},
                                 {'name': 'K', 'type': 'float64'}]}]

class TestWMLClientWithHybrid(unittest.TestCase):
    deployment_id = None
    model_id = None
    scoring_url = None
    scoring_id = None
    sw_spec_id = None
    model_uids = []
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithHybrid.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()





        self.space_name = str(uuid.uuid4())

        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
            self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestWMLClientWithHybrid.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientWithHybrid.space_id)
        self.client.set.default_space(TestWMLClientWithHybrid.space_id)
       # self.model_path = os.path.join(os.getcwd(), 'artifacts', 'customer-satisfaction-prediction.str')

        # TestWMLClientWithDO.logger.info("Service Instance: setting up credentials")
        # self.wml_credentials = get_wml_credentials()
        # self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'DrugSelectionAutoAI_model_content.gzip')
        self.update_model_path = os.path.join(os.getcwd(), 'artifacts', 'pipeline-model.json')

    # def test_01_set_space(self):
    #     space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "DO_test_case"})
    #
    #     TestWMLClientWithDO.space_id = self.client.spaces.get_id(space)
    #     self.client.set.default_space(TestWMLClientWithDO.space_id)
    #     self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithDO.space_id))

    def test_01_prepare_test(self):
        TestWMLClientWithHybrid.sw_spec_id = self.client.software_specifications.get_id_by_name("hybrid_0.1")
        if self.sw_spec_id is None:
            TestWMLClientWithHybrid.sw_spec_id = "8c1a58c6-62b5-4dc4-987a-df751c2756b6"

    def test_02_publish_hybrid_model_in_repository_input_schema_passed(self):
        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "DrugSelection Model",
                       self.client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id,
                       self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_data_schema
                            }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        self.assertTrue(published_model['entity']['schemas']['input'] is not None)
        TestWMLClientWithHybrid.model_uids.append(self.client.repository.get_model_id(published_model))

    def test_03_publish_hybrid_model_in_repository_training_data_refs_passed(self):
        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "DrugSelection Model",
                       self.client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id,
                       self.client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES: training_data_ref
                            }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        self.assertTrue(published_model['entity']['schemas']['input'] is not None)
        self.assertTrue(published_model['entity']['training_data_references'] is not None)
        TestWMLClientWithHybrid.model_uids.append(self.client.repository.get_model_id(published_model))

    def test_04_publish_hybrid_model_in_repository_input_schema_passed_and_training_data_refs_passed(self):
        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "DrugSelection Model",
                       self.client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id,
                       self.client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES: training_data_ref,
                       self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_data_schema
                            }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        self.assertTrue(published_model['entity']['schemas']['input'] is not None)
        self.assertTrue(published_model['entity']['training_data_references'] is not None)
        TestWMLClientWithHybrid.model_uids.append(self.client.repository.get_model_id(published_model))

    def test_05_publish_hybrid_model_in_repository_none_passed(self):
        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "DrugSelection Model",
                       self.client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self.sw_spec_id
                            }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        print('TEST', published_model)
        TestWMLClientWithHybrid.model_uids.append(self.client.repository.get_model_id(published_model))

    def test_11_delete_models(self):
        TestWMLClientWithHybrid.logger.info("Delete models")
        for id in self.model_uids:
            self.client.repository.delete(id)

    def test_12_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithHybrid.space_id)


if __name__ == '__main__':
    unittest.main()
