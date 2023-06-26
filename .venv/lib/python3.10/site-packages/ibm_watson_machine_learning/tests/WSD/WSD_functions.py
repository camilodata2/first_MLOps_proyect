#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import logging
from preparation_and_cleaning import *
from models_preparation import *


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

        self.function_name = 'sample_function'
        self.deployment_name = "Test deployment"
        self.project_id = os.environ['PROJECT_ID']
        self.client.set.default_project(self.project_id)

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
            self.client.repository.FunctionMetaNames.NAME: 'sample_function-sample_2',
            self.client.repository.FunctionMetaNames.RUNTIME_UID: 'ai-function_0.1-py3.6',
            self.client.repository.FunctionMetaNames.INPUT_DATA_SCHEMAS: [{
                "id": "iris function",
                "fields": [
                    {"metadata": {}, "type": "double", "name": "sepal length (cm)", "nullable": False},
                    {"metadata": {}, "type": "double", "name": "sepal width (cm)", "nullable": False},
                    {"metadata": {}, "type": "double", "name": "petal length (cm)", "nullable": False},
                    {"metadata": {}, "type": "double", "name": "petal width (cm)", "nullable": False}
                ]
            }],
            self.client.repository.FunctionMetaNames.OUTPUT_DATA_SCHEMAS: [{
                "id": "iris function",
                "fields": [
                    {"metadata": {'modeling_role': 'prediction'}, "type": "string", "name": "iris_type",
                     "nullable": False}
                ]
            }]

        }

        def score(payload):
            payload = payload['input_data'][0]
            values = [[row[0] * row[1]] for row in payload['values']]
            return {'predictions': [{'fields': ['multiplication'], 'values': values}]}

        ai_function_details = self.client.repository.store_function(score, function_props)

        TestAIFunction.function_uid = self.client.repository.get_function_uid(ai_function_details)
       # function_url = self.client.repository.get_function_href(ai_function_details)
        TestAIFunction.logger.info("AI function ID:" + str(TestAIFunction.function_uid))
       # TestAIFunction.logger.info("AI function URL:" + str(function_url))
        self.assertIsNotNone(TestAIFunction.function_uid)


    def test_03_download_ai_function_content(self):
        try:
            os.remove('test_ai_function.tar.gz')
        except:
            pass
        self.client.repository.download(TestAIFunction.function_uid, filename='test_ai_function_v42.gz')
        try:
            os.remove('test_ai_function.tar.gz')
        except:
            pass

    def test_04_get_details(self):

        details = self.client.repository.get_function_details(self.function_uid)
        print(details)
        self.assertTrue(self.function_name in str(details))


    def test_11_delete_function(self):
        TestAIFunction.logger.info("Delete function")
        self.client.repository.delete(TestAIFunction.function_uid)


if __name__ == '__main__':
    unittest.main()
