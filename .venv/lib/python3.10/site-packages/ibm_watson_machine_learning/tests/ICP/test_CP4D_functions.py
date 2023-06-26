#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import logging
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *


class TestAIFunction(unittest.TestCase):
    runtime_uid = None
    deployment_uid = None
    function_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestAIFunction.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()



        self.function_name = 'simplest AI function'
        self.deployment_name = "Test deployment"
        self.space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_TestAIFunction"})
        self.space_id = self.client.spaces.get_uid(self.space)
        self.client.set.default_space(self.space_id)

    # def test_01_service_instance_details(self):
    #     TestAIFunction.logger.info("Check client ...")
    #     self.assertTrue(self.client.__class__.__name__ == 'APIClient')
    #
    #     TestAIFunction.logger.info("Getting instance details ...")
    #     details = self.client.service_instance.get_details()
    #     TestAIFunction.logger.debug(details)
    #
    #     self.assertTrue("published_models" in str(details))
    #     self.assertEqual(type(details), dict)

    def test_01_create_ai_function(self):

        self.client.repository.FunctionMetaNames.show()

        function_props = {
            self.client.repository.FunctionMetaNames.NAME: 'sample_function',
            self.client.repository.FunctionMetaNames.RUNTIME_UID: 'ai-function_0.1-py3.6'

        }

        def score(payload):
            payload = payload['input_data'][0]
            values = [[row[0] * row[1]] for row in payload['values']]
            return {'predictions': [{'fields': ['multiplication'], 'values': values}]}

        ai_function_details = self.client.repository.store_function(score, function_props)

        TestAIFunction.function_uid = self.client.repository.get_function_uid(ai_function_details)
        function_url = self.client.repository.get_function_href(ai_function_details)
        TestAIFunction.logger.info("AI function ID:" + str(TestAIFunction.function_uid))
        TestAIFunction.logger.info("AI function URL:" + str(function_url))
        self.assertIsNotNone(TestAIFunction.function_uid)
        self.assertIsNotNone(function_url)
        TestAIFunction.runtime_uid = self.client.runtimes.get_uid(ai_function_details)

    def test_02_update_function(self):
        function_props = {
            self.client.repository.FunctionMetaNames.NAME: 'simplest AI function',

        }

        details = self.client.repository.update_function(TestAIFunction.function_uid, function_props)
        self.assertFalse('sample_function' in json.dumps(details))

    def test_03_download_ai_function_content(self):
        try:
            os.remove('test_ai_function.tar.gz')
        except:
            pass
        self.client.repository.download(TestAIFunction.function_uid, filename='test_ai_function_v41.gz')
        try:
            os.remove('test_ai_function.tar.gz')
        except:
            pass

    def test_04_get_details(self):

        details = self.client.repository.get_function_details(self.function_uid)
        self.assertTrue(self.function_name in str(details))

    def test_05_list(self):
        self.client.repository.list()

        self.client.repository.list_functions()
    def test_06_create_deployment(self):
        deploy_meta = {
                self.client.deployments.ConfigurationMetaNames.NAME: "deployment_Function",
                self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment_description",
                self.client.deployments.ConfigurationMetaNames.ONLINE: {}
            }

        TestAIFunction.logger.info("Create deployment")
        deployment = self.client.deployments.create(artifact_uid=TestAIFunction.function_uid, meta_props=deploy_meta)
        TestAIFunction.logger.debug("deployment: " + str(deployment))
        TestAIFunction.scoring_url = self.client.deployments.get_scoring_href(deployment)
        TestAIFunction.logger.debug("Scoring href: {}".format(TestAIFunction.scoring_url))
        TestAIFunction.deployment_uid = self.client.deployments.get_uid(deployment)
        TestAIFunction.logger.debug("Deployment uid: {}".format(TestAIFunction.deployment_uid))
        self.client.deployments.list()
        self.assertTrue("deployment_Function" in str(deployment))

    def test_07_update_deployment(self):
        patch_meta = {
            self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment_Updated_Function_Description",
        }
        self.client.deployments.update(TestAIFunction.deployment_uid, patch_meta)

    def test_08_get_deployment_details(self):
        TestAIFunction.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        TestAIFunction.logger.debug("Deployment details: {}".format(deployment_details))
        self.assertTrue('deployment_Function' in str(deployment_details))

    def test_09_score(self):
        scoring_payload = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "fields": ["multiplication"],
                "values": [[2.0, 2.0], [99.0, 99.0]]
            }
            ]
        }
        predictions = self.client.deployments.score(TestAIFunction.deployment_uid, scoring_payload)
        print("Predictions: {}".format(predictions))
        self.assertTrue("values" in str(predictions))

    def test_10_delete_deployment(self):
        TestAIFunction.logger.info("Delete deployment")
        self.client.deployments.delete(TestAIFunction.deployment_uid)

    # def test_11_delete_function(self):
    #     TestAIFunction.logger.info("Delete function")
    #     self.client.repository.delete(TestAIFunction.function_uid)

    def test_11_delete_space(self):
        self.client.spaces.delete(TestAIFunction.space_id)


if __name__ == '__main__':
    unittest.main()
