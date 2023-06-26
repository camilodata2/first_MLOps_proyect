#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import traceback
import os
import json
from os import environ
import ibm_boto3

import pandas as pd

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location, DatabaseLocation

from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id,
                                                     is_cp4d, bucket_exists, create_bucket)



class TestDataConnection(unittest.TestCase):
    """
    This tests covers:
    - ...
    """
    wml_client: 'APIClient' = None
    wml_credentials = None

    non_default_xlsx_data_location = './autoai/data/xlsx/iris_dataset.xlsx'
    non_default_sheet_name = 'iris'
    non_default_xlsx_prediction_column = 'species'
    non_default_UI_data_asset_id = '7ee49524-1210-402f-b5f1-41b1f9f53d4f'

    default_xlsx_data_location = './autoai/data/xlsx/bank.xlsx'
    default_sheet_name = 'Default'
    default_xlsx_prediction_column = 'y'
    default_UI_data_asset_id = 'b7462ec8-0db3-4e85-bc62-be1398fce485'

    data_connection = None

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    cos_resource_instance_id = None

    project_id = None
    space_id = None

    default_sheet_data_asset_id = None
    non_default_sheet_data_asset_id = None

    default_sheet_connected_data_asset_cos_id = None
    non_default_sheet_connected_data_asset_cos_id = None

    cos_connection_id = None

    FLIGHT_SERVICE_LOCATION = None
    FLIGHT_SERVICE_PORT = None

    cos_resource = None

    @classmethod
    def setUpClass(cls) -> None:
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

    def test_00b_create_normal_data_asset_xlsx_default_sheet(self):
        asset_details = self.wml_client.data_assets.create(
            name=self.default_xlsx_data_location.split('/')[-1],
            file_path=self.default_xlsx_data_location)

        TestDataConnection.default_sheet_data_asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.default_sheet_data_asset_id, str)

    def test_00c_create_normal_data_asset_xlsx_non_default_sheet(self):
        asset_details = self.wml_client.data_assets.create(
            name=self.non_default_xlsx_data_location.split('/')[-1],
            file_path=self.non_default_xlsx_data_location)

        TestDataConnection.non_default_sheet_data_asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.non_default_sheet_data_asset_id, str)

    def test_00d_prepare_cos_instance(self):
        import ibm_boto3

        TestDataConnection.cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # Prepare bucket
        if not bucket_exists(TestDataConnection.cos_resource, TestDataConnection.bucket_name):
            TestDataConnection.bucket_name = create_bucket(TestDataConnection.cos_resource,
                                                           TestDataConnection.bucket_name)
            self.assertIsNotNone(TestDataConnection.bucket_name)
            self.assertTrue(bucket_exists(TestDataConnection.cos_resource, TestDataConnection.bucket_name))

    def test_00e_prepare_cos_csv_connection_and_upload_data(self):
        print(f"Writing {self.default_xlsx_data_location} file with COS client to bucket {self.bucket_name}")
        TestDataConnection.cos_resource.Bucket(self.bucket_name).upload_file(
            self.default_xlsx_data_location,
            self.default_xlsx_data_location.split('/')[-1]
        )
        print("Write successful!\n")
        print(f"Writing {self.non_default_xlsx_data_location} file with COS client to bucket {self.bucket_name}")
        TestDataConnection.cos_resource.Bucket(self.bucket_name).upload_file(
            self.non_default_xlsx_data_location,
            self.non_default_xlsx_data_location.split('/')[-1]
        )
        print("Write successful!")

    def test_00f_create_connection_connection_id_created(self):
        print("Creating connection")
        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name('bluemixcloudobjectstorage'),
            'name': 'Connection to COS for tests',
            'properties': {
                'bucket': self.bucket_name,
                'access_key': self.cos_credentials['cos_hmac_keys']['access_key_id'],
                'secret_key': self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                'iam_url': self.wml_client.service_instance._href_definitions.get_iam_token_url(),
                'url': self.cos_endpoint
            }
        })

        TestDataConnection.cos_connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.cos_connection_id, str)

    def test_00g_prepare_connected_data_asset_COS_default_sheet_file(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.cos_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: f"{self.default_xlsx_data_location.split('/')[-1]} - connected data asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: '/' + self.bucket_name + '/' +
                                                                                  self.default_xlsx_data_location.split(
                                                                                      '/')[-1]
        })

        TestDataConnection.default_sheet_connected_data_asset_cos_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.default_sheet_connected_data_asset_cos_id, str)

    def test_00h_prepare_connected_data_asset_COS_default_sheet_file(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.cos_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: f"{self.non_default_xlsx_data_location.split('/')[-1]} - connected data asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: '/' + self.bucket_name + '/' +
                                                                                  self.non_default_xlsx_data_location.split(
                                                                                      '/')[-1]
        })

        TestDataConnection.non_default_sheet_connected_data_asset_cos_id = self.wml_client.data_assets.get_id(
            asset_details)
        self.assertIsInstance(self.non_default_sheet_connected_data_asset_cos_id, str)

    def test_01_download_normal_data_asset_default_sheet(self):
        data_connection = DataConnection(data_asset_id=self.default_sheet_data_asset_id)
        data_connection.set_client(self.wml_client)

        # using flight service implementation
        data = data_connection.read(use_flight=True, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.default_xlsx_prediction_column, data.columns)

        # using old implementation
        data = data_connection.read(use_flight=False,raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.default_xlsx_prediction_column, data.columns)

    def test_01a_download_normal_UI_data_asset_default_sheet(self):
        data_connection = DataConnection(data_asset_id=self.default_UI_data_asset_id)
        data_connection.set_client(self.wml_client)

        # using flight service implementation
        data = data_connection.read(use_flight=True, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.default_xlsx_prediction_column, data.columns)

        # using old implementation
        data = data_connection.read(use_flight=False, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.default_xlsx_prediction_column, data.columns)

    def test_02_download_normal_data_asset_non_default_sheet(self):
        data_connection = DataConnection(data_asset_id=self.non_default_sheet_data_asset_id)
        data_connection.set_client(self.wml_client)

        # using flight service implementation
        data = data_connection.read(use_flight=True, raw=True, excel_sheet=self.non_default_sheet_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.non_default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.non_default_xlsx_prediction_column} column in columns: {data.columns}")

        # using old implementation
        data = data_connection.read(use_flight=False, raw=True, excel_sheet=self.non_default_sheet_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.non_default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.non_default_xlsx_prediction_column} column in columns: {data.columns}")

    def test_02A_download_normal_UI_data_asset_non_default_sheet(self):
        data_connection = DataConnection(data_asset_id=self.non_default_UI_data_asset_id)
        data_connection.set_client(self.wml_client)

        # using flight service implementation
        data = data_connection.read(use_flight=True, raw=True, excel_sheet=self.non_default_sheet_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.non_default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.non_default_xlsx_prediction_column} column in columns: {data.columns}")

        # using old implementation
        data = data_connection.read(use_flight=False, raw=True, excel_sheet=self.non_default_sheet_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.non_default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.non_default_xlsx_prediction_column} column in columns: {data.columns}")

    def test_03_download_connection_asset_csv_from_COS_default_sheet(self):
        data_connection = DataConnection(connection_asset_id=self.cos_connection_id,
                                         location=S3Location(path=self.default_xlsx_data_location.split('/')[-1],
                                                             bucket=self.bucket_name))
        data_connection.set_client(self.wml_client)

        # using flight service implementation
        data = data_connection.read(use_flight=True, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.default_xlsx_prediction_column} column in columns: {data.columns}")

        # using old implementation
        data = data_connection.read(use_flight=False, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.default_xlsx_prediction_column} column in columns: {data.columns}")

    def test_04_download_connection_asset_csv_from_COS_non_default_sheet(self):
        data_connection = DataConnection(connection_asset_id=self.cos_connection_id,
                                         location=S3Location(path=self.non_default_xlsx_data_location.split('/')[-1],
                                                             bucket=self.bucket_name))
        data_connection.set_client(self.wml_client)

        # using flight service implementation
        data = data_connection.read(use_flight=True, raw=True, excel_sheet=self.non_default_sheet_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.non_default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.non_default_xlsx_prediction_column} column in columns: {data.columns}")

        # using old implementation
        data = data_connection.read(use_flight=False, raw=True, excel_sheet=self.non_default_sheet_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.non_default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.non_default_xlsx_prediction_column} column in columns: {data.columns}")

    def test_05_download_connected_data_asset_from_COS_default_sheet(self):
        #
        # flight_parameters = {
        #     "batch_size": 10000,
        #     "num_partitions": 1
        # }

        connection = DataConnection(data_asset_id=self.default_sheet_connected_data_asset_cos_id)
        connection.set_client(self.wml_client)

        # using flight service implementation
        data = connection.read(use_flight=True, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.default_xlsx_prediction_column} column in columns: {data.columns}")

        # using old implementation
        data = connection.read(use_flight=False, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.default_xlsx_prediction_column} column in columns: {data.columns}")

    def test_06_download_connected_data_asset_from_COS_non_default_sheet(self):
        data_connection = DataConnection(data_asset_id=self.non_default_sheet_connected_data_asset_cos_id)
        data_connection.set_client(self.wml_client)

        # using flight service implementation
        data = data_connection.read(use_flight=True, raw=True, excel_sheet=self.non_default_sheet_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.non_default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.non_default_xlsx_prediction_column} column in columns: {data.columns}")

        # using old implementation
        data = data_connection.read(use_flight=False, raw=True, excel_sheet=self.non_default_sheet_name)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)
        print(data.head())
        self.assertIn(self.non_default_xlsx_prediction_column, data.columns,
                      msg=f"Missing {self.non_default_xlsx_prediction_column} column in columns: {data.columns}")


    @classmethod
    def tearDownClass(cls) -> None:
        response = cls.wml_client.data_assets.delete(cls.default_sheet_data_asset_id)
        print(response)
        response = cls.wml_client.data_assets.delete(cls.non_default_sheet_data_asset_id)
        print(response)

        response = cls.wml_client.connections.delete(cls.cos_csv_connection_id)
        print(response)

        response = cls.wml_client.data_assets.delete(cls.default_sheet_connected_data_asset_cos_id)
        print(response)

        response = cls.wml_client.data_assets.delete(cls.non_default_sheet_connected_data_asset_cos_id)
        print(response)
