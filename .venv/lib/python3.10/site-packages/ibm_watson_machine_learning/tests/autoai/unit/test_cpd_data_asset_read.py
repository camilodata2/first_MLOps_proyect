#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import os
from pprint import pprint

import pandas as pd

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers import DataConnection, AssetLocation
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, create_connection_to_cos


class TestDataAsset(unittest.TestCase):
    wml_credentials = None

    flight_service_location = None
    flight_service_port = None

    # space_id = None
    project_id = None
    data_asset_id = None
    connected_data_asset_id = None
    data_connection: DataConnection = None
    data_connection_cda: DataConnection = None

    connection_id = 'c678420e-0537-4667-8edc-8ce0730534f9'
    bucket_name = 'wml-autoai-tests-v4ga-2022-03-03-dyjiw8dn'
    data_cos_path = "connected_data_asset.csv"

    data_location = './autoai/data/Wireless_Plans.csv'  # the smallest file in autoai/data

    @classmethod
    def setUp(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())

        cls.cos_credentials = get_cos_credentials()
        cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
        cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.project_id = cls.wml_credentials.get('project_id')
        cls.wml_client.set.default_project(cls.project_id)

        cls.flight_service_location = cls.wml_credentials.get('flight_service_location')
        cls.flight_service_port = cls.wml_credentials.get('flight_service_port', 443)

    def test_00__prepare_normal_data_asset(self):
        asset_details = self.wml_client.data_assets.create(
            name=self.data_location.split('/')[-1],
            file_path=self.data_location)

        TestDataAsset.data_asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.data_asset_id, str)

    def test_01__create_DataConnection_to_data_asset(self):
        TestDataAsset.data_connection = DataConnection(data_asset_id=self.data_asset_id)
        self.assertIsNotNone(TestDataAsset.data_connection)

    def test_02__connect_wml_client_to_data_connection(self):
        try:
            self.data_connection.set_client(self.wml_client)
        except AttributeError:
            self.data_connection._wml_client = self.wml_client

        self.assertIsNotNone(self.data_connection._wml_client)
        self.assertIsInstance(self.data_connection._wml_client, APIClient)

    def test_03_read_data_asset_with_data_asset_get_content_client_core(self):
        content = self.wml_client.data_assets.get_content(self.data_asset_id)
        pprint(content)

        self.assertIsNotNone(content)
        self.assertIsNotNone(content.decode('ascii'), str)

    def test_04_read_data_asset_without_flight_env_variables(self):
        if os.environ.get('FLIGHT_SERVICE_LOCATION'):
            del os.environ['FLIGHT_SERVICE_LOCATION']
        if os.environ.get('FLIGHT_SERVICE_PORT'):
            del os.environ['FLIGHT_SERVICE_PORT']

        self.assertNotIn('FLIGHT_SERVICE_LOCATION', os.environ)
        self.assertNotIn('FLIGHT_SERVICE_PORT', os.environ)

        data = self.data_connection.read()
        print(data.head())

        self.assertGreater(len(data), 0)

    def test_05_read_data_asset_with_flight_env_variables(self):
        os.environ['FLIGHT_SERVICE_LOCATION'] = self.flight_service_location
        os.environ['FLIGHT_SERVICE_PORT'] = str(self.flight_service_port)

        self.assertIsNotNone(os.environ.get('FLIGHT_SERVICE_LOCATION'))
        self.assertIsNotNone(os.environ.get('FLIGHT_SERVICE_PORT'))

        data = self.data_connection.read()
        print(data.head())

        self.assertGreater(len(data), 0)

    def test_10a_prepare_COS_instance_and_connection(self):
        TestDataAsset.connection_id, TestDataAsset.bucket_name = create_connection_to_cos(
            wml_client=self.wml_client,
            cos_credentials=self.cos_credentials,
            cos_endpoint=self.cos_endpoint,
            bucket_name=self.bucket_name,
            save_data=True,
            data_path=self.data_location,
            data_cos_path=self.data_cos_path)

        self.assertIsInstance(self.connection_id, str)

    def test_10b__prepare_connected_data_asset(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Connected data asset unit test",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: os.path.join(self.bucket_name,
                                                                                               self.data_cos_path)
        })

        TestDataAsset.connected_data_asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.connected_data_asset_id, str)

    def test_11__create_DataConnection_to_data_asset(self):
        TestDataAsset.data_connection_cda = DataConnection(data_asset_id=self.connected_data_asset_id)
        self.assertIsNotNone(TestDataAsset.data_connection_cda)

    def test_12__connect_wml_client_to_data_connection_cda(self):
        try:
            self.data_connection_cda.set_client(self.wml_client)
        except AttributeError:
            self.data_connection_cda._wml_client = self.wml_client

        self.assertIsNotNone(self.data_connection_cda._wml_client)
        self.assertIsInstance(self.data_connection_cda._wml_client, APIClient)

    def test_14_read_data_asset_without_flight_env_variables(self):
        with self.assertRaises(ConnectionError) as error:
            if os.environ.get('FLIGHT_SERVICE_LOCATION'):
                del os.environ['FLIGHT_SERVICE_LOCATION']
            if os.environ.get('FLIGHT_SERVICE_PORT'):
                del os.environ['FLIGHT_SERVICE_PORT']

            self.assertNotIn('FLIGHT_SERVICE_LOCATION', os.environ)
            self.assertNotIn('FLIGHT_SERVICE_PORT', os.environ)

            data = self.data_connection_cda.read()
            print(data.head())

            self.assertGreater(len(data), 0)

        print(error)

    def test_15_read_data_asset_with_flight_env_variables(self):
        os.environ['FLIGHT_SERVICE_LOCATION'] = self.flight_service_location
        os.environ['FLIGHT_SERVICE_PORT'] = str(self.flight_service_port)

        self.assertIsNotNone(os.environ.get('FLIGHT_SERVICE_LOCATION'))
        self.assertIsNotNone(os.environ.get('FLIGHT_SERVICE_PORT'))

        data = self.data_connection_cda.read()
        print(data.head())

        self.assertGreater(len(data), 0)

    def test_99_delete_data_asset_and_connection(self):
        self.wml_client.data_assets.delete(self.data_asset_id)
        self.wml_client.data_assets.delete(self.connected_data_asset_id)
        self.wml_client.connections.delete(self.connection_id)

if __name__ == '__main__':
    unittest.main()
