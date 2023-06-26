#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import traceback
import os
import json
from os import environ
from zipfile import ZipFile
import ibm_boto3

import pandas as pd

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location, DatabaseLocation

from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id,
                                                     is_cp4d, bucket_exists, create_bucket)

flight_enabled = environ.get('FLIGHT', False)

class TestDataConnection(unittest.TestCase):
    """
    This tests covers:
    - ...
    """
    wml_client: 'APIClient' = None
    wml_credentials = None

    csv_data_location = './autoai/data/breast_cancer.csv'
    write_csv_data_location = './autoai/data/Call_log.csv'
    json_data_location = './autoai/data/pipeline.json'
    xlsx_data_location = ''
    zip_data_location = './autoai/data/test_file.zip'

    zip_download_location = './autoai/data/downloaded_test_file.zip'

    sheet_name = ''

    prediction_column = 'species'

    data_connection = None

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    cos_resource_instance_id = None

    project_id = None
    space_id = None

    data_asset_id = None
    fs_csv_data_asset_id = None
    cos_csv_connection_id = None
    cos_json_connection_id = None
    cos_zip_connection_id = None
    cos_json_connected_data_asset_cos_id = None
    cos_zip_connected_data_asset_cos_id = None
    connected_data_asset_cos_id = None
    write_connected_data_asset_cos_id = None
    write_connected_data_asset_cos_id_json = None
    connection_id = None

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

    def test_00b_create_normal_data_asset(self):
        asset_details = self.wml_client.data_assets.create(
            name=self.csv_data_location.split('/')[-1],
            file_path=self.csv_data_location)

        TestDataConnection.fs_csv_data_asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.fs_csv_data_asset_id, str)

    def test_00c_prepare_cos_instance(self):
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

    def test_00d_prepare_cos_csv_connection_and_upload_data(self):
        TestDataConnection.cos_resource.Bucket(self.bucket_name).upload_file(
            self.csv_data_location,
            self.csv_data_location.split('/')[-1]
        )
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

        TestDataConnection.cos_csv_connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.cos_csv_connection_id, str)

    def test_00e_prepare_connected_data_asset_COS(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.cos_csv_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Iris - training asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: '/' + self.bucket_name + '/' +
                                                                                  self.csv_data_location.split('/')[-1]
        })

        TestDataConnection.connected_data_asset_cos_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.connected_data_asset_cos_id, str)

    def test_00f_prepare_cos_json_connection_and_upload_data(self):
        TestDataConnection.cos_resource.Bucket(self.bucket_name).upload_file(
            self.json_data_location,
            self.json_data_location.split('/')[-1]
        )
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

        TestDataConnection.cos_json_connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.cos_json_connection_id, str)

    def test_00g_prepare_connected_data_asset_json(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.cos_json_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "pipeline json",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: '/' + self.bucket_name + '/' +
                                                                                  self.json_data_location.split('/')[-1]
        })

        TestDataConnection.cos_json_connected_data_asset_cos_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.cos_json_connected_data_asset_cos_id, str)

    def test_00h_prepare_cos_zip_connection_and_upload_data(self):
        TestDataConnection.cos_resource.Bucket(self.bucket_name).upload_file(
            self.zip_data_location,
            self.zip_data_location.split('/')[-1]
        )
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

        TestDataConnection.cos_zip_connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.cos_zip_connection_id, str)

    def test_00i_prepare_connected_data_asset_zip(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.cos_zip_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "zip asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: '/' + self.bucket_name + '/' +
                                                                                  self.zip_data_location.split('/')[-1]
        })

        TestDataConnection.cos_zip_connected_data_asset_cos_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.cos_zip_connected_data_asset_cos_id, str)

    # def test_01_flight_env_vars_not_set_raise_error(self):
    #     connection = DataConnection(data_asset_id="dummy_id")
    #     connection.set_client(self.wml_client)
    #
    #     try:
    #         data = connection.read(use_flight=True, raw=True)
    #
    #     except ValueError as e:
    #         self.assertIn("Please make sure you have set 'FLIGHT_SERVICE_LOCATION' and 'FLIGHT_SERVICE_PORT'", str(e))

    def test_02_download_normal_data_asset_csv(self):

        connection = DataConnection(data_asset_id=self.fs_csv_data_asset_id)
        connection.set_client(self.wml_client)

        if flight_enabled:
            # using flight service implementation
            data = connection.read(use_flight=True, raw=True)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreaterEqual(len(data), 1)

        # using old implementation
        data = connection.read(use_flight=False, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)

    def test_03_download_connection_asset_csv_from_COS(self):

        connection = DataConnection(connection_asset_id=self.cos_csv_connection_id,
                                    location=S3Location(path=self.csv_data_location.split('/')[-1],
                                                        bucket=self.bucket_name))
        connection.set_client(self.wml_client)

        if flight_enabled:
            # using flight service implementation
            data = connection.read(use_flight=True, raw=True)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreaterEqual(len(data), 1)

        # using old implementation
        data = connection.read(use_flight=False, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)

    def test_04_download_data_asset_csv_from_COS_with_flight_parameters(self):

        flight_parameters = {
            "batch_size": 10000,
            "num_partitions": 1
        }

        connection = DataConnection(data_asset_id=self.connected_data_asset_cos_id)
        connection.set_client(self.wml_client)

        if flight_enabled:
            # using flight service implementation
            data = connection.read(use_flight=True, raw=True, flight_parameters=flight_parameters)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreaterEqual(len(data), 1)

        # using old implementation
        data = connection.read(use_flight=False, raw=True)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)

    @unittest.skipIf(not flight_enabled, "Flight not yet supported on Cloud")
    def test_05a_download_connection_asset_json_from_COS__binary_mode(self):

        connection = DataConnection(connection_asset_id=self.cos_json_connection_id,
                                    location=S3Location(path=self.json_data_location.split('/')[-1],
                                                        bucket=self.bucket_name))
        connection.set_client(self.wml_client)

        # using flight service implementation
        binary_data = connection.read(use_flight=True, raw=True, binary=True)
        data = json.loads(binary_data)
        print(data)
        self.assertIsInstance(data, dict)
        self.assertGreaterEqual(len(data), 1)

    @unittest.skipIf(not flight_enabled, "Flight not yet supported on Cloud")
    def test_05b_download_connection_asset_json_from_COS__normal_mode_error(self):

        connection = DataConnection(connection_asset_id=self.cos_json_connection_id,
                                    location=S3Location(path=self.json_data_location.split('/')[-1],
                                                        bucket=self.bucket_name))
        connection.set_client(self.wml_client)

        # using flight service implementation
        try:
            binary_data = connection.read(use_flight=True, raw=True)

        except TypeError as e:
            self.assertIn("Data is not of CSV/parquet type. Try to use a binary read mode.", str(e))

    @unittest.skipIf(not flight_enabled, "Flight not yet supported on Cloud")
    def test_06_download_data_asset_json_from_COS_with_flight_parameters(self):

        flight_parameters = {
            "batch_size": 10000,
            "num_partitions": 1
        }

        connection = DataConnection(data_asset_id=self.cos_json_connected_data_asset_cos_id)
        connection.set_client(self.wml_client)

        # using flight service implementation
        binary_data = connection.read(use_flight=True, raw=True, flight_parameters=flight_parameters, binary=True)
        data = json.loads(binary_data)
        print(data)
        self.assertIsInstance(data, dict)
        self.assertGreaterEqual(len(data), 1)

    @unittest.skipIf(not flight_enabled, "Flight not yet supported on Cloud")
    def test_07_download_data_asset_zip_from_COS_read_to_file(self):

        flight_parameters = {
            "batch_size": 10000,
            "num_partitions": 1
        }

        connection = DataConnection(data_asset_id=self.cos_zip_connected_data_asset_cos_id)
        connection.set_client(self.wml_client)

        # using flight service implementation
        zip_file_path = self.zip_download_location
        connection.read(use_flight=True, raw=True, flight_parameters=flight_parameters, binary=True, read_to_file=zip_file_path)

        with ZipFile(zip_file_path) as downloaded_zip:
            zip_content = downloaded_zip.namelist()
            self.assertIn('json_test_file.json', zip_content)
            self.assertIn('csv_test_file.csv', zip_content)
            with downloaded_zip.open('json_test_file.json') as f:
                json_data = json.loads(f.read())
        self.assertIsInstance(json_data, dict)
        self.assertGreaterEqual(len(json_data), 1)

    @unittest.skipIf(not flight_enabled, "Flight not yet supported on Cloud")
    def test_08a_upload_data_asset_csv_to_COS_from_file(self):

        # prepare connected data asset for COS
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.cos_csv_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Iris - training asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: '/' + self.bucket_name + '/' +
                                                                                  self.write_csv_data_location.split(
                                                                                      '/')[-1]
        })

        TestDataConnection.write_connected_data_asset_cos_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.write_connected_data_asset_cos_id, str)

        connection = DataConnection(data_asset_id=self.write_connected_data_asset_cos_id)
        connection.set_client(self.wml_client)
        connection.write(data=self.write_csv_data_location, use_flight=True)

        # using flight service implementation
        data = connection.read(use_flight=True, raw=True)
        print(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)

    @unittest.skipIf(not flight_enabled, "Flight not yet supported on Cloud")
    def test_08b_upload_data_asset_csv_to_COS_from_pandas(self):

        connection = DataConnection(data_asset_id=self.write_connected_data_asset_cos_id)
        connection.set_client(self.wml_client)

        data_1 = pd.read_csv(self.write_csv_data_location)

        try:
            connection.write(data=data_1, use_flight=True)

        except ValueError as e:
            if "Exceeds maximum data size. Please provide data file path instead of " \
               "the pandas DataFrame to upload data in binary mode." not in str(e):
                raise e

    @unittest.skipIf(not flight_enabled, "Flight not yet supported on Cloud")
    def test_09_upload_connection_asset_csv_to_COS_from_file(self):
        connection = DataConnection(connection_asset_id=self.cos_csv_connection_id,
                                    location=S3Location(path=self.write_csv_data_location.split('/')[-1],
                                                        bucket=self.bucket_name))
        connection.set_client(self.wml_client)

        connection.write(data=self.write_csv_data_location)

        # using flight service implementation
        data = connection.read(use_flight=True, raw=True)
        print(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreaterEqual(len(data), 1)

    @unittest.skipIf(not flight_enabled, "Flight not yet supported on Cloud")
    def test_10_upload_data_asset_json_to_COS_and_download_it(self):

        # prepare connected data asset for COS
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.cos_json_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "json test",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: '/' + self.bucket_name + '/' +
                                                                                  self.json_data_location.split(
                                                                                      '/')[-1]
        })

        TestDataConnection.write_connected_data_asset_cos_id_json = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.write_connected_data_asset_cos_id_json, str)

        connection = DataConnection(data_asset_id=self.write_connected_data_asset_cos_id_json)
        connection.set_client(self.wml_client)

        connection.write(data=self.json_data_location, use_flight=True)

        # using flight service implementation
        data = connection.read(use_flight=True, raw=True, binary=True)
        data = json.loads(data)
        print(data)
        self.assertIsInstance(data, dict)
        self.assertGreaterEqual(len(data), 1)

    @classmethod
    def tearDownClass(cls) -> None:

        for asset_id in [cls.fs_csv_data_asset_id, cls.cos_csv_connection_id, cls.connected_data_asset_cos_id,
                         cls.write_connected_data_asset_cos_id, cls.write_connected_data_asset_cos_id_json]:
            if asset_id:
                response = cls.wml_client.data_assets.delete(asset_id)
                print(response)
        try:
            os.remove(cls.zip_download_location)
        except:
            pass
