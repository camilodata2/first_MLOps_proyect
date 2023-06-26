#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest


from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.wml_client_error import WMLClientError


class TestDBDrivers(unittest.TestCase):
    driver_location = './autoai/db_driver_jars/exajdbc-7.1.4.jar'
    driver_file_name = driver_location.split('/')[-1]

    @classmethod
    def setUp(cls) -> None:
        cls.wml_credentials_cp4d = get_wml_credentials('CPD_4_0')
        cls.wml_client_cpd = APIClient(wml_credentials=cls.wml_credentials_cp4d)
        cls.wml_client_cpd.set.default_project(cls.wml_credentials_cp4d['project_id'])

        cls.wml_credentials = get_wml_credentials('YPQA_V2')
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)
        cls.wml_client.set.default_project(cls.wml_credentials['project_id'])

        cls.cos_credentials = get_cos_credentials('YPQA_V2')

    def test_01_upload_driver_cpd(self):
        self.driver_file_name = self.driver_location.split('/')[-1]
        self.wml_client_cpd.connections.upload_db_driver(self.driver_location)
        self.wml_client_cpd.connections.list_uploaded_db_drivers()

    def test_02_sign_db_driver_url_cpd(self):
        jar_uris = self.wml_client_cpd.connections.sign_db_driver_url(self.driver_file_name)
        print(jar_uris)
        self.assertIsNotNone(jar_uris)

    def test_03_upload_driver_cpd(self):
        self.driver_file_name = self.driver_location.split('/')[-1]
        with self.assertRaises(WMLClientError):
            self.wml_client.connections.upload_db_driver(self.driver_location)

    def test_04_sign_db_driver_url_cpd(self):
        with self.assertRaises(WMLClientError):
            jar_uris = self.wml_client.connections.sign_db_driver_url(self.driver_file_name)
            print(jar_uris)


if __name__ == '__main__':
    unittest.main()
