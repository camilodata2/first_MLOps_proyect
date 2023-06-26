#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import os
from os import environ
import ibm_boto3
import pandas as pd
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location, DatabaseLocation

from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id,
                                                     is_cp4d, bucket_exists, create_bucket)

class TestDataConnSampling(unittest.TestCase):
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
    zip_data_location = ''

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
    cos_json_connected_data_asset_cos_id = None
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

        cls.FLIGHT_SERVICE_LOCATION = cls.wml_credentials.get('FLIGHT_SERVICE_LOCATION', os.environ.get('FLIGHT_SERVICE_LOCATION'))
        cls.FLIGHT_SERVICE_PORT = cls.wml_credentials.get('FLIGHT_SERVICE_PORT', os.environ.get('FLIGHT_SERVICE_PORT'))

    def test_00b_create_normal_data_asset_FS(self):
        asset_details = self.wml_client.data_assets.create(
            name=self.csv_data_location.split('/')[-1],
            file_path=self.csv_data_location)

        TestDataConnSampling.fs_csv_data_asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.fs_csv_data_asset_id, str)

    def test_00c_prepare_cos_instance(self):
        TestDataConnSampling.cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # Prepare bucket
        if not bucket_exists(TestDataConnSampling.cos_resource, TestDataConnSampling.bucket_name):
            TestDataConnSampling.bucket_name = create_bucket(TestDataConnSampling.cos_resource, TestDataConnSampling.bucket_name)
            self.assertIsNotNone(TestDataConnSampling.bucket_name)
            self.assertTrue(bucket_exists(TestDataConnSampling.cos_resource, TestDataConnSampling.bucket_name))

    def test_00d_prepare_cos_csv_connection_and_upload_data(self):
        TestDataConnSampling.cos_resource.Bucket(self.bucket_name).upload_file(
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

        TestDataConnSampling.cos_csv_connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.cos_csv_connection_id, str)

    def test_00e_prepare_connected_data_asset_COS(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.cos_csv_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Iris - training asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: '/' + self.bucket_name + '/' + self.csv_data_location.split('/')[-1]
        })

        TestDataConnSampling.connected_data_asset_cos_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.connected_data_asset_cos_id, str)

    def test_01_download_data_asset_csv_from_FS_with_sampling(self):
        # Test how API is used by AutoAI backend

        os.environ['FLIGHT_SERVICE_LOCATION'] = self.FLIGHT_SERVICE_LOCATION
        os.environ['FLIGHT_SERVICE_PORT'] = self.FLIGHT_SERVICE_PORT

        connection = DataConnection(data_asset_id=self.fs_csv_data_asset_id)
        connection.set_client(self.wml_client)

        experiment_metadata = {
            "prediction_type": 'classification',
            "prediction_column": 'diagnosis'
        }

        full_data = connection.read(
            raw=True,
            experiment_metadata=experiment_metadata
        )
        self.assertIsInstance(full_data, pd.DataFrame)
        full_data_len = len(full_data)
        print(f"Full data rows: {full_data_len}")
        self.assertGreaterEqual(full_data_len, 569)

        for sampling_type in ['first_n_records', 'random', 'stratified']:

            for no_of_rows in [284, 376, 568]:
                data = connection.read(
                    raw=True,
                    experiment_metadata=experiment_metadata,
                    sampling_type=sampling_type,
                    sample_rows_limit=no_of_rows
                )
                self.assertIsInstance(data, pd.DataFrame)
                print(f"For: \n"
                      f"- sampling type: {sampling_type}, sample_rows_limit={no_of_rows} \n"
                      f"- data rows: {len(data)} vs expected data rows: {no_of_rows}")
                self.assertEqual(len(data), no_of_rows)

            for perc_of_rows in [0.25, 0.5, 0.7, 0.85]:
                data = connection.read(
                    raw=True,
                    experiment_metadata=experiment_metadata,
                    sampling_type=sampling_type,
                    sample_percentage_limit=perc_of_rows
                )
                self.assertIsInstance(data, pd.DataFrame)
                expected_rows = perc_of_rows*full_data_len if sampling_type != 'first_n_records' else full_data_len
                print(f"For: \n"
                      f"- sampling type: {sampling_type}, sample_percentage_limit={perc_of_rows} \n"
                      f"- data rows: {len(data)} vs expected data rows: {expected_rows}")
                # TO DO: Check percentage sampling for project csv data_assets
                #self.assertAlmostEqual(len(data), expected_rows, None, "differs", 5)


    def test_02_download_connection_asset_csv_from_COS_with_sampling(self):
        os.environ['FLIGHT_SERVICE_LOCATION'] = self.FLIGHT_SERVICE_LOCATION
        os.environ['FLIGHT_SERVICE_PORT'] = self.FLIGHT_SERVICE_PORT

        connection = DataConnection(connection_asset_id=self.cos_csv_connection_id,
                                    location=S3Location(path=self.csv_data_location.split('/')[-1],
                                                        bucket=self.bucket_name))
        connection.set_client(self.wml_client)

        experiment_metadata = {
            "prediction_type": 'classification',
            "prediction_column": 'diagnosis'
        }

        full_data = connection.read(
            raw=True,
            experiment_metadata=experiment_metadata
        )
        self.assertIsInstance(full_data, pd.DataFrame)
        full_data_len = len(full_data)
        print(f"Full data rows: {full_data_len}")
        self.assertGreaterEqual(full_data_len, 569)

        for sampling_type in ['first_n_records', 'random', 'stratified']:

            for no_of_rows in [284, 376, 568]:
                data = connection.read(
                    raw=True,
                    experiment_metadata=experiment_metadata,
                    sampling_type=sampling_type,
                    sample_rows_limit=no_of_rows
                )
                self.assertIsInstance(data, pd.DataFrame)
                print(f"For: \n"
                      f"- sampling type: {sampling_type}, sample_rows_limit={no_of_rows} \n"
                      f"- data rows: {len(data)} vs expected data rows: {no_of_rows}")
                self.assertEqual(len(data), no_of_rows)

            for perc_of_rows in [0.25, 0.5, 0.7, 0.85]:
                data = connection.read(
                    raw=True,
                    experiment_metadata=experiment_metadata,
                    sampling_type=sampling_type,
                    sample_percentage_limit=perc_of_rows
                )
                self.assertIsInstance(data, pd.DataFrame)
                expected_rows = perc_of_rows * full_data_len if sampling_type != 'first_n_records' else full_data_len
                print(f"For: \n"
                      f"- sampling type: {sampling_type}, sample_percentage_limit={perc_of_rows} \n"
                      f"- data rows: {len(data)} vs expected data rows: {expected_rows}")
                self.assertAlmostEqual(len(data), expected_rows, None, "differs", 1)


    @classmethod
    def tearDownClass(cls) -> None:
        response = cls.wml_client.data_assets.delete(cls.fs_csv_data_asset_id)
        print(response)

        response = cls.wml_client.connections.delete(cls.cos_csv_connection_id)
        print(response)

        response = cls.wml_client.data_assets.delete(cls.connected_data_asset_cos_id)
        print(response)
