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
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.CP4D_35.models_preparation import *


class TestWMLClientWithScikitLearn(unittest.TestCase):
    deployment_id = None
    model_id = None
    scoring_url = None
    space_id = None
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

    def test_01_publish_model(self):
        TestWMLClientWithScikitLearn.logger.info("Creating scikit-learn model ...")

        predicted = self.model_data['prediction']

        TestWMLClientWithScikitLearn.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        sw_spec_id = self.client.software_specifications.get_uid_by_name("scikit-learn_0.20-py3.6")
        #sw_spec_uid = '09c5a1d0-9c1e-4473-a344-eb7b665ff687'
        print('sw_space_uid: ', sw_spec_id)
        # input_schema = input_data_schema = {
        #                                 "id": "test1",
        #                                 "type": "list",
        #                                 "fields": [{
        #                                     "name": "id",
        #                                     "type": "double",
        #                                     "nullable": True,
        #                                   }]
        #                                 }
        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id #,
            # self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_schema
                       }
        published_model_details = self.client.repository.store_model(model=self.model_data['model'], meta_props=model_props, training_data=self.model_data['training_data'], training_target=self.model_data['training_target'])
        print(published_model_details)
        TestWMLClientWithScikitLearn.model_id = self.client.repository.get_model_id(published_model_details)
        TestWMLClientWithScikitLearn.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithScikitLearn.model_id))
        self.logger.info("Published model URL:" + str(TestWMLClientWithScikitLearn.model_url))
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_id)
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_url)

    def test_02_download_model(self):
        TestWMLClientWithScikitLearn.logger.info("Download model")
        try:
            os.remove('download_test_url')
        except OSError:
            pass

        try:
            file = open('download_test_id', 'r')
        except IOError:
            file = open('download_test_id', 'w')
            file.close()

        self.client.repository.download(TestWMLClientWithScikitLearn.model_id, filename='download_test_url')
        self.assertRaises(WMLClientError, self.client.repository.download, TestWMLClientWithScikitLearn.model_id, filename='download_test_id')

    def test_03_get_details(self):
        TestWMLClientWithScikitLearn.logger.info("Get details")
        self.assertIsNotNone(self.client.repository.get_details())
        det = self.client.repository.get_details()
        print(det)

    def test_04_get_model_details(self):
        TestWMLClientWithScikitLearn.logger.info("Get model details")
        details = self.client.repository.get_details(TestWMLClientWithScikitLearn.model_id)
        import json
        print(details)
        self.assertIsNotNone(self.client.repository.get_model_details(TestWMLClientWithScikitLearn.model_id))

    # def test_05_create_revision(self):
    #     TestWMLClientWithScikitLearn.logger.info("Create Revision")
    #     self.client.repository.create_model_revision(TestWMLClientWithScikitLearn.model_id)
    #     self.client.repository.list_models_revisions(TestWMLClientWithScikitLearn.model_id)
    #     model_meta_props = {self.client.repository.ModelMetaNames.NAME: "updated scikit model",
    #                         }
    #     published_model_updated = self.client.repository.update_model(TestWMLClientWithScikitLearn.model_id,
    #                                                                   model_meta_props, self.model_data['model'])
    #     self.assertIsNotNone(TestWMLClientWithScikitLearn.model_id)
    #     self.assertTrue("updated scikit model" in str(published_model_updated))
    #     self.client.repository.create_model_revision(TestWMLClientWithScikitLearn.model_id)
    #     rev_details = self.client.repository.get_model_revision_details(TestWMLClientWithScikitLearn.model_id, 2)
    #     self.assertTrue("updated scikit model" in str(rev_details))

    # def test_06_create_deployment(self):
    #     TestWMLClientWithScikitLearn.logger.info("Create deployments")
    #
    #     # # Temporary workaround
    #     # metadata = {
    #     #     self.client.spaces.ConfigurationMetaNames.NAME: "updated_space",
    #     #     self.client.spaces.ConfigurationMetaNames.COMPUTE: {"name": "test_instance",
    #     #                                                         "crn": self.instance_crn
    #     #                                                        }
    #     # }
    #     #
    #     # space_update_details = self.client.spaces.update(TestWMLClientWithScikitLearn.space_id, metadata)
    #     #
    #     # print("Updated space: ", space_update_details)
    #
    #     deployment = self.client.deployments.create(self.model_id, meta_props={
    #         self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",
    #         self.client.deployments.ConfigurationMetaNames.ONLINE: {}})
    #
    #     print('Deployment: ', deployment)
    #
    #     TestWMLClientWithScikitLearn.deployment_id = self.client.deployments.get_id(deployment)
    #     TestWMLClientWithScikitLearn.logger.info("model_id: " + self.model_id)
    #     TestWMLClientWithScikitLearn.logger.info("Online deployment: " + str(deployment))
    #     self.assertTrue(deployment is not None)
    #     TestWMLClientWithScikitLearn.scoring_url = self.client.deployments.get_scoring_href(deployment)
    #     self.assertTrue("online" in str(deployment))
    #     #self.client.deployments.get_status(TestWMLClientWithScikitLearn.deployment_id)
    #
    # def test_07_get_deployment_details(self):
    #     TestWMLClientWithScikitLearn.logger.info("Get deployment details")
    #     deployment_details = self.client.deployments.get_details()
    #     print(deployment_details)
    #     self.assertTrue("Test deployment" in str(deployment_details))
    #
    # def test_08_get_deployment_details_using_id(self):
    #     TestWMLClientWithScikitLearn.logger.info("Get deployment details using uid")
    #     deployment_details = self.client.deployments.get_details(TestWMLClientWithScikitLearn.deployment_id)
    #     print(deployment_details)
    #     self.assertIsNotNone(deployment_details)
    #
    # def test_09_score(self):
    #     TestWMLClientWithScikitLearn.logger.info("Score model")
    #     scoring_data = {
    #         self.client.deployments.ScoringMetaNames.INPUT_DATA: [
    #             {
    #                'values': [[0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0, 16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 15.0, 1.0, 0.0, 0.0], [0.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 13.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 16.0, 9.0, 4.0, 1.0, 0.0, 0.0, 3.0, 16.0, 16.0, 16.0, 16.0, 10.0, 0.0, 0.0, 5.0, 16.0, 11.0, 9.0, 6.0, 2.0]]
    #             }
    #         ]
    #     }
    #     predictions = self.client.deployments.score(TestWMLClientWithScikitLearn.deployment_id, scoring_data)
    #     print("predictions: ", predictions)
    #     self.assertTrue("prediction" in str(predictions))

    # def test_05_score(self):
    #     TestWMLClientWithScikitLearn.logger.info("Score model")
    #     scoring_data = {
    #         self.client.deployments.ScoringMetaNames.INPUT_DATA: [
    #             {
    #                'values': [[0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 15.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0, 16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 15.0, 1.0, 0.0, 0.0], [0.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 16.0, 13.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 16.0, 9.0, 4.0, 1.0, 0.0, 0.0, 3.0, 16.0, 16.0, 16.0, 16.0, 10.0, 0.0, 0.0, 5.0, 16.0, 11.0, 9.0, 6.0, 2.0]]
    #             }
    #         ]
    #     }
    #     predictions = self.client.deployments.score(TestWMLClientWithScikitLearn.deployment_id, scoring_data)
    #     self.assertTrue("prediction" in str(predictions))


    # def test_10_delete_deployment(self):
    #     TestWMLClientWithScikitLearn.logger.info("Delete deployment")
    #     self.client.deployments.delete(TestWMLClientWithScikitLearn.deployment_id)

    def test_11_delete_model(self):
        TestWMLClientWithScikitLearn.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithScikitLearn.model_id)

    def test_12_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithScikitLearn.space_id)


if __name__ == '__main__':
    unittest.main()