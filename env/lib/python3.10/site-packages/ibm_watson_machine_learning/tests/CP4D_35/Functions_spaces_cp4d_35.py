#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *

class TestFunction(unittest.TestCase):
    runtime_uid = None
    deployment_uid = None
    function_id = None
    function_id2 = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestFunction.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()


        # self.space_id = get_project_id()
        # self.client.set.default_project(self.space_id)


        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestFunction.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestFunction.space_id)

        self.client.set.default_space(TestFunction.space_id)
        # self.client.set.default_space('db7b719a-c095-4bb9-94e8-29b93b51beb2')
        # self.client.set.default_space('b88a8678-c653-4a2e-88b7-870e9655f752')

        import time
        time.sleep(30)

    def test_01_create_function(self):

        self.client.repository.FunctionMetaNames.show()
        # self.client.software_specifications.list()
        sw_spec_id = '0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda'
        print('sw_spec_id: ', sw_spec_id)

        function_meta_props = {
            self.client.repository.FunctionMetaNames.NAME: 'sample_function_with_sw',
            self.client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: sw_spec_id #,
            # self.client.repository.FunctionMetaNames.SAMPLE_SCORING_INPUT: {
            #     "input_data": [
            #       {
            #         "fields": [
            #           "name",
            #           "age",
            #           "occupation"
            #         ],
            #         "values": [
            #           [
            #             "john",
            #             23,
            #             "student"
            #           ],
            #           [
            #             "paul",
            #             33,
            #             "engineer"
            #           ]
            #         ]
            #       }
            #     ]
            #   } #,
            # self.client.repository.FunctionMetaNames.INPUT_DATA_SCHEMAS: [{
            #     "id": "id",
            #     "name": "input_data_schema",
            #     "fields": [
            #         {
            #             "name": "x",
            #             "type": "double",
            #             "nullable": False
            #             # "metadata": {"measure":"default", "modeling_role":"none"}
            #         },
            #         {
            #             "name": "y",
            #             "type": "double",
            #             "nullable": False
            #             # "metadata": {"measure":"default", "modeling_role":"none"}
            #         }
            #     ]
            # }],
            # self.client.repository.FunctionMetaNames.OUTPUT_DATA_SCHEMAS: [{
            #     "id": "id",
            #     "name": "output_data_schema",
            #     "fields": [
            #         {
            #             "name": "multiplication",
            #             "type": "double",
            #             "nullable": False
            #             # "metadata": {"measure":"default", "modeling_role":"none"}
            #         }
            #     ]
            # }]
        }

        # def test_update():
        #     def score(payload):
        #         return {'predictions': [{'fields': [], 'values': [['working']]}]}
        #
        #     return score
        #
        def score(payload):
            payload = payload['input_data'][0]
            values = [[row[0] * row[1]] for row in payload['values']]
            return {'predictions': [{'fields': ['multiplication'], 'values': values}]}

        # def score_generator():
        #     import urllib3, requests, json, time, os
        #     import subprocess
        #     # Define scoring function
        #     def callModel(payload_scoring):
        #         ## user code
        #         import sys
        #         import os
        #         import sklearn
        #         import numpy as np
        #         skl_ver = sklearn.__version__
        #         np_ver = np.__version__
        #         v4_scoring_response = {
        #             'predictions': [{'fields': ['sklearn', 'np'],
        #                              'values': [[skl_ver, np_ver]]
        #                              }]
        #         }
        #         return v4_scoring_response
        #
        #     def score(input):
        #         """AI function example.
        #         Example:
        #             {
        #                 "input_data": [{
        #                         "id": "1",
        #                         "fields": [
        #                             "name",
        #                             "age",
        #                             "occupation"
        #                         ],
        #                         "values": [
        #                             [
        #                                 "john",
        #                                 23,
        #                                 "student"
        #                             ],
        #                             [
        #                                 "paul",
        #                                 33,
        #                                 "engineer"
        #                             ]
        #                         ]
        #                     }
        #                 ]
        #             }
        #         """
        #         # Score using the pre-defined model
        #         prediction = callModel(input);
        #         return prediction
        #
        #     return score

        # function_details = self.client.repository.store_function(score_generator, function_meta_props)
        function_details = self.client.repository.store_function(score, function_meta_props)
        # function_details = self.client.repository.store_function(test_update, function_meta_props)

        print(function_details)

        TestFunction.function_id = self.client.repository.get_function_id(function_details)
        function_url = self.client.repository.get_function_href(function_details)
        TestFunction.logger.info("function ID:" + str(TestFunction.function_id))
        TestFunction.logger.info("function URL:" + str(function_url))
        self.assertIsNotNone(TestFunction.function_id)
        self.assertIsNotNone(function_url)

    def test_02_create_function2(self):

        self.client.repository.FunctionMetaNames.show()
        sw_spec_id = self.client.software_specifications.get_id_by_name("ai-function_0.1-py3.6")

        print(sw_spec_id)

        function_meta_props = {
            self.client.repository.FunctionMetaNames.NAME: 'sample_function2_with_sw',
            self.client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
            self.client.repository.FunctionMetaNames.TYPE: 'python'
        }

        function_details2 = self.client.repository.store_function(function='artifacts/score.py.gz',
                                                                  meta_props=function_meta_props)
        # function_details2 = self.client.repository.store_function(function='/Users/mitun_vb/test_function.gz',
        #                                                           meta_props=function_meta_props)

        print(function_details2)

        TestFunction.function_id2 = self.client.repository.get_function_id(function_details2)
        function_url2 = self.client.repository.get_function_href(function_details2)
        TestFunction.logger.info("function ID:" + str(TestFunction.function_id2))
        TestFunction.logger.info("function URL:" + str(function_url2))
        self.assertIsNotNone(TestFunction.function_id2)
        self.assertIsNotNone(function_url2)

    def test_03_get_details(self):

        details = self.client.repository.get_function_details(TestFunction.function_id)
        print(details)
        self.assertTrue('sample_function' in str(details))

        details2 = self.client.repository.get_function_details(TestFunction.function_id2)
        print(details2)
        self.assertTrue('sample_function2' in str(details2))

    def test_04_list(self):
        # self.client.repository.list()
        self.client.repository.list_functions()

    def test_05_download_ai_function_content(self):
        try:
            os.remove('test_function.gz')
        except:
            pass
        self.client.repository.download(TestFunction.function_id, filename='test_function.gz')

        try:
            os.remove('test_function.gz')
        except:
            pass

        try:
            os.remove('test_function1.gz')
        except:
            pass
        self.client.repository.download(TestFunction.function_id2, filename='test_function1.gz')

        try:
            os.remove('test_function1.gz')
        except:
            pass

    def test_06_update_function(self):
        function_update_meta_props = {
            self.client.repository.FunctionMetaNames.NAME: 'updated_function_name'
        }

        # def test_update():
        #     def score(payload):
        #         return {'predictions': [{'fields': [], 'values': [['working']]}]}
        #
        #     return score

        details = self.client.repository.update_function(TestFunction.function_id,
                                                         function_update_meta_props,
                                                         # update_function=test_update)
                                                         update_function='artifacts/score_rev1.py.gz')
        self.assertFalse('sample_function' in json.dumps(details))
        self.assertTrue('updated_function_name' in json.dumps(details))

        try:
            os.remove('test_function_update.gz')
        except:
            pass
        self.client.repository.download(TestFunction.function_id, filename='test_function_update.gz')

        try:
            os.remove('test_function_update.gz')
        except:
            pass

    def test_07_revisions(self):
        revision = self.client.repository.create_function_revision(TestFunction.function_id)
        print('revision: ', revision)
        self.assertTrue(revision[u'metadata'][u'rev'] == '1')

        revision_details = self.client.repository.get_function_revision_details(TestFunction.function_id, 1)
        print('revision details: ', revision_details)
        self.assertTrue(revision[u'metadata'][u'rev'] == '1')

        self.client.repository.list_functions_revisions(TestFunction.function_id)

        try:
            os.remove('test_function_revision.gz')
        except:
            pass

        # Download revision 1 attachment
        self.client.repository.download(TestFunction.function_id,
                                        filename='test_function_revision.gz',
                                        rev_uid=1)
        try:
            os.remove('test_function_revision.gz')
        except:
            pass


    def test_08_create_deployment(self):
        deploy_meta = {
                self.client.deployments.ConfigurationMetaNames.NAME: "deployment_Function",
                self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment_description",
                self.client.deployments.ConfigurationMetaNames.ONLINE: {}
            }

        TestFunction.logger.info("Create deployment")
        deployment = self.client.deployments.create(artifact_uid=TestFunction.function_id, meta_props=deploy_meta)
        TestFunction.logger.debug("deployment: " + str(deployment))
        TestFunction.scoring_url = self.client.deployments.get_scoring_href(deployment)
        TestFunction.logger.debug("Scoring href: {}".format(TestFunction.scoring_url))
        TestFunction.deployment_id = self.client.deployments.get_id(deployment)
        TestFunction.logger.debug("Deployment uid: {}".format(TestFunction.deployment_id))
        self.client.deployments.list()
        self.assertTrue("deployment_Function" in str(deployment))

    def test_09_update_deployment(self):
        patch_meta = {
            self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment_Updated_Function_Description",
        }
        self.client.deployments.update(TestFunction.deployment_id, patch_meta)

    def test_10_get_deployment_details(self):
        TestFunction.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        TestFunction.logger.debug("Deployment details: {}".format(deployment_details))
        self.assertTrue('deployment_Function' in str(deployment_details))

    def test_11_score(self):
        scoring_payload = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "fields": ["multiplication"],
                "values": [[2.0, 2.0], [99.0, 99.0]]
            }
            ]
        }
        predictions = self.client.deployments.score(TestFunction.deployment_id, scoring_payload)
        print("Predictions: {}".format(predictions))
        self.assertTrue("values" in str(predictions))

    def test_12_delete_deployment(self):
        TestFunction.logger.info("Delete deployment")
        self.client.deployments.delete(TestFunction.deployment_id)

    def test_13_delete_function(self):
        TestFunction.logger.info("Delete function")
        self.client.repository.delete(TestFunction.function_id)
        self.client.repository.delete(TestFunction.function_id2)

    def test_14_delete_space(self):
        self.client.spaces.delete(TestFunction.space_id)

if __name__ == '__main__':
    unittest.main()
