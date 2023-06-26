#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
# from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.tests.Cloud.preparation_and_cleaning import *
import os
import logging

logging.basicConfig(level=logging.DEBUG)


class TestFactsheets(unittest.TestCase):
    model_id = None
    model_content_path = f'{os.getcwd()}/Cloud/artifacts/tf_model_fvt_test.tar.gz'
    space_id = "c93565d9-41c2-4dbc-8586-2b32da8ad018"
    catalog_id = '12a02c34-0e5f-443a-9ea5-694ed6c6e56f'
    model_entry_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls):
        TestFactsheets.logger.info("Service Instance: setting up credentials")

        cls.wml_credentials = get_wml_credentials()
        cls.client = get_client()
        cls.client.set.default_space(TestFactsheets.space_id)

    def test_01_publish_local_model_in_repository(self):
        self.client.repository.ModelMetaNames.show()
        swspec_id = self.client.software_specifications.get_uid_by_name('default_py3.7')

        model_details = self.client.repository.store_model(
            model=self.model_content_path,
            meta_props={self.client.repository.ModelMetaNames.NAME: "test_tf",
                        self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: swspec_id,
                        self.client.repository.ModelMetaNames.TYPE: 'tensorflow_2.1'})
        TestFactsheets.model_id = self.client.repository.get_model_uid(model_details)

        print(TestFactsheets.model_id)

        self.assertIsNotNone(TestFactsheets.model_id)

    def test_02a__register_WKC_model_entry_new(self):
        register_details = self.client.factsheets.register_model_entry(
            model_id=self.model_id,
            meta_props={
                self.client.factsheets.ConfigurationMetaNames.NAME: 'Tensorflow model 2',
                self.client.factsheets.ConfigurationMetaNames.DESCRIPTION: 'Tensorflow model',
                self.client.factsheets.ConfigurationMetaNames.MODEL_ENTRY_CATALOG_ID: self.catalog_id
            })
        self.assertIsInstance(register_details, dict,
                              msg="register_details is not of dictionary type! It could be empty.")
        print(register_details)

        TestFactsheets.model_entry_id = register_details['model_entry_id']

    def test_02b__register_WKC_model_entry_existing(self):
        register_details = self.client.factsheets.register_model_entry(
            model_id=self.model_id,
            meta_props={
                self.client.factsheets.ConfigurationMetaNames.ASSET_ID: self.model_entry_id,
                self.client.factsheets.ConfigurationMetaNames.MODEL_ENTRY_CATALOG_ID: self.catalog_id
            })
        self.assertIsInstance(register_details, dict,
                              msg="register_details is not of dictionary type! It could be empty.")
        print(register_details)

    def test_03a_list_all_wkc_model_entries_for_specified_catalog(self):
        response = self.client.factsheets.list_model_entries(catalog_id=self.catalog_id)
        print(response)
        self.assertIsInstance(response, dict, msg="Response is not of dictionary type! It could be empty.")

    def test_03b_list_all_wkc_model_entries_for_all_catalogs(self):
        response = self.client.factsheets.list_model_entries()
        print(response)
        self.assertIsInstance(response, dict, msg="Response is not of dictionary type! It could be empty.")

    def test_04_unregister_WKC_model_entry(self):
        self.client.factsheets.unregister_model_entry(asset_id=self.model_id)


if __name__ == '__main__':
    unittest.main()
