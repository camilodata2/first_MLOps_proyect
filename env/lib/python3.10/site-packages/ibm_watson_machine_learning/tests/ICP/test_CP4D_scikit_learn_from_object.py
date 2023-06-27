#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
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


class TestWMLClientWithScikitLearn(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    space_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithScikitLearn.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()


    def test_00_set_space(self):
        space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_TestWMLClientWithScikitLearn"})

        TestWMLClientWithScikitLearn.space_id = self.client.spaces.get_uid(space)
        self.client.set.default_space(TestWMLClientWithScikitLearn.space_id)
        self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithScikitLearn.space_id))

    # def test_01_service_instance_details(self):
    #     TestWMLClientWithScikitLearn.logger.info("Check client ...")
    #     self.assertTrue(self.client.__class__.__name__ == 'APIClient')
    #
    #     self.logger.info("Getting instance details ...")
    #     details = self.client.service_instance.get_details()
    #     TestWMLClientWithScikitLearn.logger.debug(details)
    #
    #     self.assertTrue("published_models" in str(details))
    #     self.assertEqual(type(details), dict)

    def test_01_publish_model(self):
        # space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case"})
        # space_id = self.client.spaces.get_uid(space)
        # self.client.set.default_space(space_id)
        # TestWMLClientWithScikitLearn.space_id = space_id
        # print("The space is" + space_id)
        TestWMLClientWithScikitLearn.logger.info("Creating scikit-learn model ...")

        model_data = create_scikit_learn_model_data()
        predicted = model_data['prediction']

        TestWMLClientWithScikitLearn.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "scikit-learn_0.20-py3.6"
                       }
        published_model_details = self.client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], training_target=model_data['training_target'])
        TestWMLClientWithScikitLearn.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithScikitLearn.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithScikitLearn.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithScikitLearn.model_url))
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_uid)
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_url)

    # def test_03_download_model(self):
    #     TestWMLClientWithScikitLearn.logger.info("Download model")
    #     try:
    #         os.remove('download_test_url')
    #     except OSError:
    #         pass
    #
    #     try:
    #         file = open('download_test_uid', 'r')
    #     except IOError:
    #         file = open('download_test_uid', 'w')
    #         file.close()
    #
    #     self.client.repository.download(TestWMLClientWithScikitLearn.model_uid, filename='download_test_url')
    #     self.assertRaises(WMLClientError, self.client.repository.download, TestWMLClientWithScikitLearn.model_uid, filename='download_test_uid')
    #
    # def test_04_get_details(self):
    #     TestWMLClientWithScikitLearn.logger.info("Get model details")
    #     details = self.client.repository.get_details(self.model_uid)
    #     TestWMLClientWithScikitLearn.logger.debug("Model details: " + str(details))
    #     self.assertTrue("ScikitModel" in str(details))
    #
    #     details_all = self.client.repository.get_details()
    #     TestWMLClientWithScikitLearn.logger.debug("All artifacts details: " + str(details_all))
    #     self.assertTrue("ScikitModel" in str(details_all))
    #
    #     details_models = self.client.repository.get_model_details()
    #     TestWMLClientWithScikitLearn.logger.debug("All models details: " + str(details_models))
    #     self.assertTrue("ScikitModel" in str(details_models))

    def test_02_create_deployment(self):
        TestWMLClientWithScikitLearn.logger.info("Create deployments")
        deployment = self.client.deployments.create(self.model_uid, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE:{}})
        TestWMLClientWithScikitLearn.deployment_uid = self.client.deployments.get_uid(deployment)
        TestWMLClientWithScikitLearn.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithScikitLearn.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithScikitLearn.scoring_url = self.client.deployments.get_scoring_href(deployment)
        self.assertTrue("online" in str(deployment))
        #self.client.deployments.get_status(TestWMLClientWithScikitLearn.deployment_uid)

    def test_03_get_deployment_details(self):
        TestWMLClientWithScikitLearn.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_04_get_deployment_details_using_uid(self):
        TestWMLClientWithScikitLearn.logger.info("Get deployment details using uid")
        deployment_details = self.client.deployments.get_details(TestWMLClientWithScikitLearn.deployment_uid)
        self.assertIsNotNone(deployment_details)

    def test_05_score(self):
        TestWMLClientWithScikitLearn.logger.info("Score model")
        scoring_data = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                   'values': [[0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0, 16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 15.0, 1.0, 0.0, 0.0], [0.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 13.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 16.0, 9.0, 4.0, 1.0, 0.0, 0.0, 3.0, 16.0, 16.0, 16.0, 16.0, 10.0, 0.0, 0.0, 5.0, 16.0, 11.0, 9.0, 6.0, 2.0]]
                }
            ]
        }
        predictions = self.client.deployments.score(TestWMLClientWithScikitLearn.deployment_uid, scoring_data)
        self.assertTrue("prediction" in str(predictions))

    def test_06_delete_deployment(self):
        TestWMLClientWithScikitLearn.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithScikitLearn.deployment_uid)

    def test_07_delete_model(self):
        TestWMLClientWithScikitLearn.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithScikitLearn.model_uid)
        self.client.spaces.delete(TestWMLClientWithScikitLearn.space_id)


if __name__ == '__main__':
    unittest.main()