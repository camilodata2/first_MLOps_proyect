#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *


class TestWMLClientWithPMML(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    scoring_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithPMML.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()





        self.space_name = str(uuid.uuid4())

        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
            self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestWMLClientWithPMML.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientWithPMML.space_id)
        self.client.set.default_space(TestWMLClientWithPMML.space_id)

    def test_01_publish_local_model_in_repository(self):

        self.client.repository.ModelMetaNames.show()

        model_content_path = 'artifacts/pmml-model.xml'

        sw_spec_uid = self.client.software_specifications.get_uid_by_name("spark-mllib_2.4")

        meta_props = {self.client.repository.ModelMetaNames.NAME: "test_pmml",
                      self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid,
                      self.client.repository.ModelMetaNames.TYPE: 'pmml_1.1'}

        model_details = self.client.repository.store_model(model=model_content_path, meta_props=meta_props)

        TestWMLClientWithPMML.model_id = self.client.repository.get_model_uid(model_details)

        print(TestWMLClientWithPMML.model_id)

        self.assertIsNotNone(TestWMLClientWithPMML.model_id)

    def test_02_get_details(self):
        TestWMLClientWithPMML.logger.info("Get details")
        self.assertIsNotNone(self.client.repository.get_details(TestWMLClientWithPMML.model_id))

    def test_03_create_deployment(self):
        TestWMLClientWithPMML.logger.info("Create deployment")
        deployment_details = self.client.deployments.create(TestWMLClientWithPMML.model_id,
                                                            meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test pmml deployment",
                                                                        self.client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithPMML.logger.debug("Deployment details: {}".format(deployment_details))

        TestWMLClientWithPMML.deployment_id = self.client.deployments.get_id(deployment_details)

        TestWMLClientWithPMML.scoring_url = self.client.deployments.get_scoring_href(deployment_details)
        TestWMLClientWithPMML.logger.debug("Scoring url: {}".format(TestWMLClientWithPMML.scoring_url))

        self.assertTrue('online' in str(deployment_details))

    def test_04_get_deployment_details(self):
        TestWMLClientWithPMML.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details(deployment_uid=TestWMLClientWithPMML.deployment_id)
        self.assertIsNotNone(deployment_details)

    def test_05_delete_model(self):
        TestWMLClientWithPMML.logger.info("Delete model")
        self.client.deployments.delete(TestWMLClientWithPMML.deployment_id)
        self.client.repository.delete(TestWMLClientWithPMML.model_id)

    def test_06_delete_space(self):
        TestWMLClientWithPMML.logger.info("Delete space")
        self.client.spaces.delete(TestWMLClientWithPMML.space_id)

if __name__ == '__main__':
    unittest.main()
