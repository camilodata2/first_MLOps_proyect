#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import time
import pandas as pd
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
#from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.CP4D_35.models_preparation import *


class TestWMLClientWithPMML(unittest.TestCase):
    deployment_id = None
    model_id = None
    scoring_url = None
    scoring_id = None
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
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'pmml-model.xml')

        # TestWMLClientWithDO.logger.info("Service Instance: setting up credentials")
        # self.wml_credentials = get_wml_credentials()
        # self.client = get_client()
        # self.model_path = os.path.join(os.getcwd(), 'artifacts', 'DrugSelectionAutoAI_model_content.gzip')
        # self.update_model_path = os.path.join(os.getcwd(), 'artifacts', 'pipeline-model.json')

    # def test_01_set_space(self):
    #     space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "DO_test_case"})
    #
    #     TestWMLClientWithDO.space_id = self.client.spaces.get_id(space)
    #     self.client.set.default_space(TestWMLClientWithDO.space_id)
    #     self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithDO.space_id))


    def test_02_publish_hybrid_model_in_repository(self):
        TestWMLClientWithPMML.logger.info("Saving trained model in repo ...")
        TestWMLClientWithPMML.logger.debug("Model path: {}".format(self.model_path))

        self.client.repository.ModelMetaNames.show()

        sw_spec_id = self.client.software_specifications.get_id_by_name("spark-mllib_2.4")
        print(sw_spec_id)
        meta_props = {
            self.client.repository.ModelMetaNames.NAME: "pmmlmodel",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
            self.client.repository.ModelMetaNames.TYPE: 'pmml_4.2.1'}
        published_model = self.client.repository.store_model(self.model_path, meta_props=meta_props)
        TestWMLClientWithPMML.model_id = self.client.repository.get_model_id(published_model)
        TestWMLClientWithPMML.logger.info("Published model ID:" + str(TestWMLClientWithPMML.model_id))
        self.assertIsNotNone(TestWMLClientWithPMML.model_id)

    def test_04_get_details(self):
        TestWMLClientWithPMML.logger.info("Get details")
        self.assertIsNotNone(self.client.repository.get_details())
        det = self.client.repository.get_details()
        print(det)

    def test_05_get_model_details(self):
        TestWMLClientWithPMML.logger.info("Get model details")
        details = self.client.repository.get_details(TestWMLClientWithPMML.model_id)
        import json
        print(details)
        self.assertIsNotNone(self.client.repository.get_model_details(TestWMLClientWithPMML.model_id))

    def test_10_download_model(self):
        TestWMLClientWithPMML.logger.info("Download model")
        try:
            os.remove('download_test_url-pmml')
        except OSError:
            pass

        try:
            os.remove('download_test_url-pmml')
        except IOError:
            pass

        self.client.repository.download(TestWMLClientWithPMML.model_id, filename='download_test_url-pmml')
        try:
            os.remove('download_test_url-pmml')
        except OSError:
            pass

    def test_11_delete_model(self):
        TestWMLClientWithPMML.logger.info("Delete function")
        self.client.script.delete(TestWMLClientWithPMML.model_id)

    def test_12_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithPMML.space_id)


if __name__ == '__main__':
    unittest.main()
