#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import svm
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.ICP.models_preparation import *


class TestWMLClientWithCrossDeployment(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    function_uid = None
    space_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithCrossDeployment.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.function_name = 'simplest AI function'
        self.deployment_name = "Test deployment"


    def test_00_set_space(self):
        space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_TestWMLClientWithScikitLearnSW"})

        TestWMLClientWithCrossDeployment.space_id = self.client.spaces.get_uid(space)
        self.client.set.default_space(TestWMLClientWithCrossDeployment.space_id)
        self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithCrossDeployment.space_id))

    def test_01_publish_model(self):
        # space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case"})
        # space_id = self.client.spaces.get_uid(space)
        # self.client.set.default_space(space_id)
        # TestWMLClientWithScikitLearn.space_id = space_id
        #print("The space is" + space_id)
        TestWMLClientWithCrossDeployment.logger.info("Creating scikit-learn model ...")

        model_data = create_scikit_learn_model_data()
        predicted = model_data['prediction']

        TestWMLClientWithCrossDeployment.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        sw_spec_uid = self.client.software_specifications.get_uid_by_name("scikit-learn_0.20-py3.6")

        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid
                       }
        published_model_details = self.client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], training_target=model_data['training_target'])
        TestWMLClientWithCrossDeployment.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithCrossDeployment.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithCrossDeployment.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithCrossDeployment.model_url))
        self.assertIsNotNone(TestWMLClientWithCrossDeployment.model_uid)
        self.assertIsNotNone(TestWMLClientWithCrossDeployment.model_url)


    def test_02_create_deployment(self):
        TestWMLClientWithCrossDeployment.logger.info("Create deployments")
        deployment = self.client.deployments.create(self.model_uid, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE:{}})
        TestWMLClientWithCrossDeployment.deployment_uid = self.client.deployments.get_uid(deployment)
        TestWMLClientWithCrossDeployment.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithCrossDeployment.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithCrossDeployment.scoring_url = self.client.deployments.get_scoring_href(deployment)
        self.assertTrue("online" in str(deployment))

    def test_03_get_deployment_details(self):
        TestWMLClientWithCrossDeployment.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_04_get_deployment_details_using_uid(self):
        TestWMLClientWithCrossDeployment.logger.info("Get deployment details using uid")
        deployment_details = self.client.deployments.get_details(TestWMLClientWithCrossDeployment.deployment_uid)
        self.assertIsNotNone(deployment_details)

    def test_05_score(self):
        TestWMLClientWithCrossDeployment.logger.info("Score model")
        scoring_data = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                   'values': [[0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0, 16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 15.0, 1.0, 0.0, 0.0], [0.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 13.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 16.0, 9.0, 4.0, 1.0, 0.0, 0.0, 3.0, 16.0, 16.0, 16.0, 16.0, 10.0, 0.0, 0.0, 5.0, 16.0, 11.0, 9.0, 6.0, 2.0]]
                }
            ]
        }
        predictions = self.client.deployments.score(TestWMLClientWithCrossDeployment.deployment_uid, scoring_data)
        self.assertTrue("prediction" in str(predictions))

    def test_06_create_function(self):
        self.client.repository.FunctionMetaNames.show()
        sw_spec_uid = self.client.software_specifications.get_uid_by_name("ai-function_0.1-py3.6")

        function_props = {
            self.client.repository.FunctionMetaNames.NAME: 'sample_function_with_sw',
            self.client.repository.FunctionMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid

        }

        def score(payload):
            payload = payload['input_data'][0]
            values = [[row[0] * row[1]] for row in payload['values']]
            return {'predictions': [{'fields': ['multiplication'], 'values': values}]}

        ai_function_details = self.client.repository.store_function(score, function_props)

        TestWMLClientWithCrossDeployment.function_uid = self.client.repository.get_function_uid(ai_function_details)
        function_url = self.client.repository.get_function_href(ai_function_details)
        TestWMLClientWithCrossDeployment.logger.info("AI function ID:" + str(TestWMLClientWithCrossDeployment.function_uid))
        TestWMLClientWithCrossDeployment.logger.info("AI function URL:" + str(function_url))
        self.assertIsNotNone(TestWMLClientWithCrossDeployment.function_uid)
        self.assertIsNotNone(function_url)

    def test_07_udpate_deployment(self):

        TestWMLClientWithCrossDeployment.logger.info("Update deployments")
        updated_deployment_details = self.client.deployments.update(TestWMLClientWithCrossDeployment.deployment_uid,
                                                    changes={
            self.client.deployments.ConfigurationMetaNames.ASSET: { "id":TestWMLClientWithCrossDeployment.function_uid } })
        details = self.client.deployments.get_details(TestWMLClientWithCrossDeployment.deployment_uid)
        #self.assertTrue(TestWMLClientWithCrossDeployment.function_uid in details)

    def test_08_delete_deployment(self):
        TestWMLClientWithCrossDeployment.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithCrossDeployment.deployment_uid)

    def test_09_delete_model(self):
         TestWMLClientWithCrossDeployment.logger.info("Delete model")
         self.client.repository.delete(TestWMLClientWithCrossDeployment.model_uid)

    def test_10_delete_function(self):
        TestWMLClientWithCrossDeployment.logger.info("Delete function")
        self.client.repository.delete(TestWMLClientWithCrossDeployment.function_uid)

    def test_11_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithCrossDeployment.space_id)


if __name__ == '__main__':
    unittest.main()