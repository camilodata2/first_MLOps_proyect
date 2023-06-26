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


class TestWMLClientWithPatchModelDeploymentArtifact(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    model_uid_spss = None
    space_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.function_name = 'simplest AI function'
        self.deployment_name = "Test deployment"
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'customer-satisfaction-prediction.str')

    def test_00_set_space(self):
        space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_TestWMLClientWithScikitLearnSW"})

        TestWMLClientWithPatchModelDeploymentArtifact.space_id = self.client.spaces.get_uid(space)
        self.client.set.default_space(TestWMLClientWithPatchModelDeploymentArtifact.space_id)
        self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithPatchModelDeploymentArtifact.space_id))

    def test_01_publish_model(self):
        # space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case"})
        # space_id = self.client.spaces.get_uid(space)
        # self.client.set.default_space(space_id)
        # TestWMLClientWithScikitLearn.space_id = space_id
        #print("The space is" + space_id)
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Creating scikit-learn model ...")

        model_data = create_scikit_learn_model_data()
        predicted = model_data['prediction']

        TestWMLClientWithPatchModelDeploymentArtifact.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        sw_spec_uid = self.client.software_specifications.get_uid_by_name("scikit-learn_0.20-py3.6")

        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid
                       }
        published_model_details = self.client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], training_target=model_data['training_target'])
        TestWMLClientWithPatchModelDeploymentArtifact.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithPatchModelDeploymentArtifact.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithPatchModelDeploymentArtifact.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithPatchModelDeploymentArtifact.model_url))
        self.assertIsNotNone(TestWMLClientWithPatchModelDeploymentArtifact.model_uid)
        self.assertIsNotNone(TestWMLClientWithPatchModelDeploymentArtifact.model_url)


    def test_02_create_deployment(self):
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Create deployments")
        deployment = self.client.deployments.create(self.model_uid, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE:{}})
        TestWMLClientWithPatchModelDeploymentArtifact.deployment_uid = self.client.deployments.get_uid(deployment)
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithPatchModelDeploymentArtifact.scoring_url = self.client.deployments.get_scoring_href(deployment)
        self.assertTrue("online" in str(deployment))

    def test_03_get_deployment_details(self):
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_04_get_deployment_details_using_uid(self):
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Get deployment details using uid")
        deployment_details = self.client.deployments.get_details(TestWMLClientWithPatchModelDeploymentArtifact.deployment_uid)
        self.assertIsNotNone(deployment_details)

    def test_05_score(self):
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Score model")
        scoring_data = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                   'values': [[0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0, 16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 15.0, 1.0, 0.0, 0.0], [0.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 13.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 16.0, 9.0, 4.0, 1.0, 0.0, 0.0, 3.0, 16.0, 16.0, 16.0, 16.0, 10.0, 0.0, 0.0, 5.0, 16.0, 11.0, 9.0, 6.0, 2.0]]
                }
            ]
        }
        predictions = self.client.deployments.score(TestWMLClientWithPatchModelDeploymentArtifact.deployment_uid, scoring_data)
        self.assertTrue("prediction" in str(predictions))

    def test_06_create_spssmode(self):
        sw_spec_uid = self.client.software_specifications.get_uid_by_name("spss-modeler_18.1")

        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "LOCALLY created agaricus prediction model",
                            self.client.repository.ModelMetaNames.TYPE: "spss-modeler_18.1",
                            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid
                            }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        TestWMLClientWithPatchModelDeploymentArtifact.model_uid_spss = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Published model ID:" + str(TestWMLClientWithPatchModelDeploymentArtifact.model_uid_spss))
        self.assertIsNotNone(TestWMLClientWithPatchModelDeploymentArtifact.model_uid_spss)


    def test_07_udpate_deployment(self):

        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Update deployments")
        updated_deployment_details = self.client.deployments.update(TestWMLClientWithPatchModelDeploymentArtifact.deployment_uid,
                                                                    changes={
            self.client.deployments.ConfigurationMetaNames.ASSET: { "id":TestWMLClientWithPatchModelDeploymentArtifact.model_uid_spss} })
        #self.assertTrue(TestWMLClientWithPatchModelDeploymentArtifact.model_uid_spss in updated_deployment_details)

    def test_08_delete_deployment(self):
        TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithPatchModelDeploymentArtifact.deployment_uid)

    def test_09_delete_model(self):
         TestWMLClientWithPatchModelDeploymentArtifact.logger.info("Delete model")
         self.client.repository.delete(TestWMLClientWithPatchModelDeploymentArtifact.model_uid)
         self.client.repository.delete(TestWMLClientWithPatchModelDeploymentArtifact.model_uid_spss)

    def test_11_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithPatchModelDeploymentArtifact.space_id)


if __name__ == '__main__':
    unittest.main()