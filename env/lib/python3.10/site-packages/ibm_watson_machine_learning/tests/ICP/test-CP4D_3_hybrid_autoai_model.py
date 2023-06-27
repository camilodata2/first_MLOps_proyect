#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *


class TestWMLClientWithHybrid(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    scoring_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithHybrid.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'DrugSelectionAutoAI_model_content.gzip')

    def test_01_set_space(self):
        space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_autoaimodel"})

        TestWMLClientWithHybrid.space_id = self.client.spaces.get_uid(space)
        self.client.set.default_space(TestWMLClientWithHybrid.space_id)
        self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithHybrid.space_id))


    def test_02_publish_local_model_in_repository(self):
        TestWMLClientWithHybrid.logger.info("Saving trained model in repo ...")
        TestWMLClientWithHybrid.logger.debug("Model path: {}".format(self.model_path))

        self.client.repository.ModelMetaNames.show()

        sw_spec_uid = self.client.software_specifications.get_uid_by_name("hybrid_0.1")
        input_data_schema = [{
            "id": "auto_ai_kb_input_schema",
            "fields": [
                {
                  "name": "AGE",
                  "type": "int64"
                },
                {
                  "name": "SEX",
                  "type": "object"
                },
                {
                  "name": "BP",
                  "type": "object"
                },
                {
                  "name": "CHOLESTEROL",
                  "type": "object"
                },
                {
                  "name": "NA",
                  "type": "float64"
                },
                {
                  "name": "K",
                  "type": "float64"
                }
            ]
        }]

        trainind_data_ref = [
                {
                    "connection": {
                        "endpoint_url": "",
                        "access_key_id": "",
                        "secret_access_key": ""
                    },
                    "location": {
                        "bucket": "",
                        "path": ""
                    },
                    "type": "fs",
                    "schema": {
                        "id": "4cdb0a0a-1c69-43a0-a8c0-3918afc7d45f",
                        "fields": [
                            {
                                "metadata": {
                                    "name": "AGE",
                                    "scale": 0
                                },
                                "name": "AGE",
                                "nullable": True,
                                "type": "integer"
                            },
                            {
                                "metadata": {
                                    "name": "SEX",
                                    "scale": 0
                                },
                                "name": "SEX",
                                "nullable": True,
                                "type": "string"
                            },
                            {
                                "metadata": {
                                    "name": "BP",
                                    "scale": 0
                                },
                                "name": "BP",
                                "nullable": True,
                                "type": "string"
                            },
                            {
                                "metadata": {
                                    "name": "CHOLESTEROL",
                                    "scale": 0
                                },
                                "name": "CHOLESTEROL",
                                "nullable": True,
                                "type": "string"
                            },
                            {
                                "metadata": {
                                    "name": "NA",
                                    "scale": 6
                                },
                                "name": "NA",
                                "nullable": True,
                                "type": "decimal(12,6)"
                            },
                            {
                                "metadata": {
                                    "name": "K",
                                    "scale": 6
                                },
                                "name": "K",
                                "nullable": True,
                                "type": "decimal(13,6)"
                            }
                        ],
                        "type": "struct"
                    }
                }
        ]

        model_meta_props = {
                       self.client.repository.ModelMetaNames.NAME: "LOCALLY created DrugSelectionAutoAIModel",
                       self.client.repository.ModelMetaNames.TYPE: 'wml-hybrid_0.1',
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid,
                       self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_data_schema,
                       self.client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES: trainind_data_ref
                       }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        TestWMLClientWithHybrid.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithHybrid.logger.info("Published model ID:" + str(TestWMLClientWithHybrid.model_uid))
        self.assertIsNotNone(TestWMLClientWithHybrid.model_uid)

    def test_03_load_model(self):
        TestWMLClientWithHybrid.logger.info("Load model from repository: {}".format(TestWMLClientWithHybrid.model_uid))
        self.tf_model = self.client.repository.load(TestWMLClientWithHybrid.model_uid)
        TestWMLClientWithHybrid.logger.debug("SPSS type: {}".format(type(self.tf_model)))
        self.assertTrue(self.tf_model)


    def test_05_get_model_details(self):
        TestWMLClientWithHybrid.logger.info("Get model details")
        details = self.client.repository.get_model_details(TestWMLClientWithHybrid.model_uid)
        self.assertIsNotNone(details)

    def test_11_delete_model(self):
        TestWMLClientWithHybrid.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithHybrid.model_uid)
        self.client.spaces.delete(TestWMLClientWithHybrid.space_id)


if __name__ == '__main__':
    unittest.main()
