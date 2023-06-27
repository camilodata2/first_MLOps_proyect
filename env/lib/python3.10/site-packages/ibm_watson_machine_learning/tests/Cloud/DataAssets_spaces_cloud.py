#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.Cloud.preparation_and_cleaning import *

class TestDataAssets(unittest.TestCase):

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestDataAssets.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.cos_credentials = get_cos_credentials()
        self.cos_resource_crn = self.cos_credentials['resource_instance_id']
        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'space' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description',
                     self.client.spaces.ConfigurationMetaNames.STORAGE: {
                                                                           "type": "bmcos_object_storage",
                                                                           "resource_crn": self.cos_resource_crn
                                                                        }
        }

        self.space = self.client.spaces.store(meta_props=metadata, background_mode=False)

        TestDataAssets.space_id = self.client.spaces.get_id(self.space)
        # TestRshinyApp.space_id = '9cf73498-72d6-42eb-a6f4-424616de9f45'
        print('space_id: ', TestDataAssets.space_id)
        self.client.set.default_space(TestDataAssets.space_id)

    def test_01_create_data_asset(self):
        asset_meta_props = {
            self.client.data_assets.ConfigurationMetaNames.NAME: "test data asset",
            self.client.data_assets.ConfigurationMetaNames.DESCRIPTION: "test data asset",
            self.client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "artifacts/GoSales_Tx_NaiveBayes.csv"
        }
        asset_details = self.client.data_assets.store(asset_meta_props)
        print(asset_details)

        TestDataAssets.asset_id = self.client.data_assets.get_id(asset_details)
        self.assertIsNotNone(TestDataAssets.asset_id)

    def test_02_get_details(self):

        details = self.client.data_assets.get_details(TestDataAssets.asset_id)
        print(details)
        self.assertTrue(TestDataAssets.asset_id in str(details))

    def test_03_list(self):
        self.client.data_assets.list()

    def test_04_download(self):
        try:
            os.remove('test.csv')
        except:
            pass
        self.client.data_assets.download(TestDataAssets.asset_id, filename='test.csv')
        try:
            os.remove('test.csv')
        except:
            pass

    def test_05_delete_space(self):
        self.client.data_assets.delete(TestDataAssets.asset_id)
        self.client.spaces.delete(TestDataAssets.space_id)

if __name__ == '__main__':
    unittest.main()

