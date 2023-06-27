#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import ibm_boto3

from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.utils.autoai.errors import CannotReadSavedRemoteDataBeforeFit, WMLClientError
from ibm_watson_machine_learning.tests.utils import bucket_exists, create_bucket
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import (
    AbstractTestAutoAIRemote)


class TestAutoAIRemote(AbstractTestAutoAIRemote, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    The test covers:
    - COS connection set-up
    - Saving data `iris.xlsx` to connected data assets
    - downloading training data from data assets
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline
    - deployment deletion
    Connection used in test:
     - input: DataAsset pointing to COS.
     - output: [None] - default connection depending on environment.
    """

    data_location = './autoai/data/xlsx/iris_dataset.xlsx'
    data_cos_path = 'iris_dataset.xlsx'
    # custom_separator = ','
    test_data_location = './autoai/data/xlsx/iris_dataset.xlsx'
    test_data_cos_path = 'iris_dataset.xlsx'

    test_asset_id=None
    sheet_name = "iris"
    test_sheet_name = 'iris'

    def test_00b_prepare_COS_instance(self):
        TestAutoAIRemote.cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # Prepare bucket
        if not bucket_exists(self.cos_resource, self.bucket_name):
            TestAutoAIRemote.bucket_name = create_bucket(self.cos_resource, self.bucket_name)

            self.assertIsNotNone(TestAutoAIRemote.bucket_name)
            self.assertTrue(bucket_exists(self.cos_resource, TestAutoAIRemote.bucket_name))

        self.cos_resource.Bucket(self.bucket_name).upload_file(
            self.data_location,
            self.data_cos_path
        )
        self.cos_resource.Bucket(self.bucket_name).upload_file(
            self.test_data_location,
            self.test_data_cos_path
        )

    def test_00c_prepare_connection_to_COS(self):
        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name('bluemixcloudobjectstorage'), #'bluemixcloudobjectstorage',
            'name': 'Connection to COS for tests',
            'properties': {
                'bucket': self.bucket_name,
                'access_key': self.cos_credentials['cos_hmac_keys']['access_key_id'],
                'secret_key': self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                'iam_url': self.wml_client.service_instance._href_definitions.get_iam_token_url(),
                'url': self.cos_endpoint
            }
        })

        TestAutoAIRemote.connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.connection_id, str)

    def test_00d_prepare_connected_data_asset_train_data(self):
        # asset_details = self.wml_client.data_assets.store({
        #     self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.connection_id,
        #     self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Iris - training asset",
        #     self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: self.data_cos_path
        # })
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Iris - training asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: f"/{self.bucket_name}/{self.data_cos_path}/{self.sheet_name}"
        })

        TestAutoAIRemote.asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.asset_id, str)

    def test_00e_prepare_connected_data_asset_holdout_data(self):
        # asset_details = self.wml_client.data_assets.store({
        #     self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.connection_id,
        #     self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Iris - training asset",
        #     self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: self.data_cos_path
        # })
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Iris - holdout/test asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: f"/{self.bucket_name}/{self.test_data_cos_path}/{self.sheet_name}"
        })

        TestAutoAIRemote.test_asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.asset_id, str)

    def test_02_DataConnection_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(data_asset_id=self.asset_id)
        TestAutoAIRemote.test_data_connection = DataConnection(data_asset_id=self.test_asset_id)
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    def test_03_initialize_optimizer(self):
        AbstractTestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.MULTICLASS,
            prediction_column=self.prediction_column,
            holdout_size=0.15,
            max_number_of_estimators=1,
            # csv_separator=self.custom_separator,
            autoai_pod_version=self.pod_version,
            notebooks=True,
            n_parallel_data_connections=self.max_connection_nb,
            retrain_on_holdout=True,
            excel_sheet=self.sheet_name,
            test_data_excel_sheet=self.test_sheet_name
        )


    def test_06b_get_test_data(self):
        from pandas import read_excel

        if self.test_data_connection:
            test_data = self.remote_auto_pipelines.get_test_data_connections()[0].read(excel_sheet=self.sheet_name)

            print("test data sample:")
            print(test_data.head())
            self.assertGreater(len(test_data), 0)
            # self.assertEqual(len(test_data), len(read_excel(self.test_data_location, sheet_name=self.test_sheet_name)))

    def test_29_delete_connection_and_connected_data_asset(self):
        self.wml_client.data_assets.delete(self.asset_id)
        self.wml_client.data_assets.delete(self.test_asset_id)
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.data_assets.get_details(self.asset_id)
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
