#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.tests.Cloud.preparation_and_cleaning import *


class TestWMLClientWithKeras(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    scoring_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithKeras.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.cos_credentials = get_cos_credentials()
        self.cos_resource_crn = self.cos_credentials['resource_instance_id']
        self.instance_crn = get_instance_crn()

        self.space_name = str(uuid.uuid4())

        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: 'space' + self.space_name,
            self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description',
            self.client.spaces.ConfigurationMetaNames.STORAGE: {
                "type": "bmcos_object_storage",
                "resource_crn": self.cos_resource_crn
            },
            self.client.spaces.ConfigurationMetaNames.COMPUTE: {
                         "name": "test_tf",
                         "crn": self.instance_crn
            }
        }

        self.space = self.client.spaces.store(meta_props=metadata, background_mode=False)

        TestWMLClientWithKeras.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientWithKeras.space_id)
        self.client.set.default_space(TestWMLClientWithKeras.space_id)

    def test_01_publish_local_model_in_repository(self):

        self.client.repository.ModelMetaNames.show()

        # model_content_path = 'artifacts/mnistCNN.h5.tgz'
        model_content_path = 'artifacts/tf_model_fvt_test.tar.gz'
        sw_spec_id = self.client.software_specifications.get_uid_by_name('default_py3.7')

        model_details = self.client.repository.store_model(model=model_content_path,
                                                           meta_props={self.client.repository.ModelMetaNames.NAME: "test_keras",
                                                                  self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: '2b73a275-7cbf-420b-a912-eae7f436e0bc',
                                                                       self.client.repository.ModelMetaNames.TYPE: 'tensorflow_2.1'})
                                                                  # self.client.repository.ModelMetaNames.TYPE: 'keras_2.2.5'})
        TestWMLClientWithKeras.model_id = self.client.repository.get_model_uid(model_details)

        print(TestWMLClientWithKeras.model_id)

        self.assertIsNotNone(TestWMLClientWithKeras.model_id)

    def test_02_get_details(self):
        TestWMLClientWithKeras.logger.info("Get details")
        self.assertIsNotNone(self.client.repository.get_details(TestWMLClientWithKeras.model_id))

    def test_03_create_deployment(self):
        TestWMLClientWithKeras.logger.info("Create deployment")
        deployment_details = self.client.deployments.create(TestWMLClientWithKeras.model_id,
                                                            meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test keras deployment",
                                                                        self.client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithKeras.logger.debug("Deployment details: {}".format(deployment_details))

        TestWMLClientWithKeras.deployment_id = self.client.deployments.get_id(deployment_details)

        TestWMLClientWithKeras.scoring_url = self.client.deployments.get_scoring_href(deployment_details)
        TestWMLClientWithKeras.logger.debug("Scoring url: {}".format(TestWMLClientWithKeras.scoring_url))

        self.assertTrue('online' in str(deployment_details))

    def test_04_get_deployment_details(self):
        TestWMLClientWithKeras.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details(deployment_uid=TestWMLClientWithKeras.deployment_id)
        self.assertIsNotNone(deployment_details)

    def test_05_delete_model(self):
        TestWMLClientWithKeras.logger.info("Delete model")
        self.client.deployments.delete(TestWMLClientWithKeras.deployment_id)
        self.client.repository.delete(TestWMLClientWithKeras.model_id)

    def test_06_delete_space(self):
        TestWMLClientWithKeras.logger.info("Delete space")
        self.client.spaces.delete(TestWMLClientWithKeras.space_id)

if __name__ == '__main__':
    unittest.main()
