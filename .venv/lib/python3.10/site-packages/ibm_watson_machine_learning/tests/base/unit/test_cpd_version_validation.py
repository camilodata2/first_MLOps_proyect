#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.tests.utils import get_wml_credentials


class TestVersionValidation(unittest.TestCase):
    wml_client: 'APIClient' = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

    def test_version_incorrect_in_wml_credentials(self):
        incorrect_wml_credentials = {
            'url': self.wml_credentials['url'],
            'username': self.wml_credentials['username'],
            'password': self.wml_credentials['password'],
            "instance_id": self.wml_credentials['instance_id'],
            'version': '4.0'
        }

        wml_client = APIClient(wml_credentials=incorrect_wml_credentials)
        wml_client.spaces.list()

        self.assertTrue(wml_client.ICP_47)
        self.assertFalse(wml_client.ICP_46)
        self.assertFalse(wml_client.ICP_40)

        self.assertEqual(incorrect_wml_credentials.get('version'), '4.0',
                         msg="original dictionary was modified by client.")

    def test_version_wml_credentials_no_version_passed(self):
        incorrect_wml_credentials = {
            'url': self.wml_credentials['url'],
            'username': self.wml_credentials['username'],
            'apikey': self.wml_credentials['apikey'],
            "instance_id": self.wml_credentials['instance_id'],
        }

        wml_client = APIClient(wml_credentials=incorrect_wml_credentials)
        wml_client.spaces.list()

        self.assertTrue(wml_client.ICP_47)
        self.assertFalse(wml_client.ICP_46)
        self.assertFalse(wml_client.ICP_40)

        self.assertIsNone(incorrect_wml_credentials.get('version', None),
                          msg="original dictionary was modified by client.")

    def test_version_wml_credentials_random_version(self):
        incorrect_wml_credentials = {
            'url': self.wml_credentials['url'],
            'username': self.wml_credentials['username'],
            'password': self.wml_credentials['password'],
            "instance_id": self.wml_credentials['instance_id'],
            'version': "random"
        }

        wml_client = APIClient(wml_credentials=incorrect_wml_credentials)
        wml_client.spaces.list()
        self.assertTrue(wml_client.ICP_47)
        self.assertFalse(wml_client.ICP_46)
        self.assertFalse(wml_client.ICP_40)

        self.assertEqual(incorrect_wml_credentials.get('version'), 'random',
                         msg="original dictionary was modified by client.")
