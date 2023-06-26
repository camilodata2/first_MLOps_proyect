#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from os.path import join
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import bucket_exists, create_bucket, is_cp4d, save_data_to_cos_bucket
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAISync,\
    AbstractTestWebservice

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics


@unittest.skipIf(not is_cp4d(), "Excel files are not supported yet on Cloud")
class TestAutoAIRemote(AbstractTestAutoAISync, AbstractTestWebservice, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = './autoai/data/xlsx/CarPrice_bank__two_sheets.xlsx'
    sheet_name = 'bank'
    sheet_number = 1

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'data/CarPrice_bank__two_sheets.xlsx'

    SPACE_ONLY = True

    OPTIMIZER_NAME = "Bank test sdk"

    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc='test description',
        prediction_type=PredictionType.BINARY,
        prediction_column='y',
        positive_label='yes',
        scoring=Metrics.ACCURACY_SCORE,
        holdout_size=0.1,
        max_number_of_estimators=1,
        excel_sheet=sheet_name
    )

    def test_00b_prepare_COS_instance(self):
        TestAutoAIRemote.bucket_name = save_data_to_cos_bucket(self.data_location,
                                                               self.data_cos_path,
                                                               access_key_id=self.cos_credentials['cos_hmac_keys'][
                                                                   'access_key_id'],
                                                               secret_access_key=self.cos_credentials['cos_hmac_keys'][
                                                                   'secret_access_key'],
                                                               cos_endpoint=self.cos_endpoint,
                                                               bucket_name=self.bucket_name)

    def test_00c_prepare_connection_to_COS(self):
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

        TestAutoAIRemote.connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.connection_id, str)

    def test_00d_prepare_connected_data_asset(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Bank - training asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: join(self.bucket_name,
                                                                                       self.data_cos_path,
                                                                                       self.experiment_info['excel_sheet'])
        })

        TestAutoAIRemote.asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.asset_id, str)

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(data_asset_id=self.asset_id)
        TestAutoAIRemote.results_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.results_cos_path
            )
        )

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNotNone(obj=TestAutoAIRemote.results_connection)

    def test_99_delete_connection_and_connected_data_asset(self):
        self.wml_client.data_assets.delete(self.asset_id)
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.data_assets.get_details(self.asset_id)
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
