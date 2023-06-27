#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import ibm_boto3
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import create_connection_to_cos, save_data_to_cos_bucket
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestTSADAsync, \
    AbstractTestWebservice

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics
from ibm_watson_machine_learning.utils.autoai.errors import NoAutomatedHoldoutSplit


class TestAutoAIRemote(AbstractTestTSADAsync, unittest.TestCase):
    """
    The test can be run on Cloud only
    """

    cos_resource = None
    data_location = './autoai/data/real_17.csv'

    data_cos_path = 'real_17.csv'

    batch_payload_location = data_location
    batch_payload_cos_location = data_cos_path

    SPACE_ONLY = False
    ONLINE_DEPLOYMENT = False
    BATCH_DEPLOYMENT = False
    NOTEBOOK_CHECK = False
    LALE_CHECK = False

    OPTIMIZER_NAME = "Timeseries anomaly prediction real_17"
    DEPLOYMENT_NAME = OPTIMIZER_NAME + "Deployment"

    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.TIMESERIES_ANOMALY_PREDICTION,
        feature_columns=['value'],
        timestamp_column_name='timestamp',
        lookback_window=10,
        max_number_of_estimators=2,
        retrain_on_holdout=False,
        notebooks=True
    )

    def test_00b_prepare_COS_instance_and_connection(self):
        TestAutoAIRemote.connection_id, TestAutoAIRemote.bucket_name = create_connection_to_cos(
            wml_client=self.wml_client,
            cos_credentials=self.cos_credentials,
            cos_endpoint=self.cos_endpoint,
            bucket_name=self.bucket_name,
            save_data=True,
            data_path=self.data_location,
            data_cos_path=self.data_cos_path)

        self.assertIsInstance(self.connection_id, str)

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.data_cos_path
            )
        )
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNone(obj=TestAutoAIRemote.results_connection)

    def test_05a_param_in_configuration(self):
        optimizer_id = self.remote_auto_pipelines._workspace.wml_client.training.get_details(
            training_uid=self.run_id
        ).get('entity')['pipeline']['id']
        optimizer_config = self.remote_auto_pipelines._workspace.wml_client.pipelines.get_details(pipeline_uid=optimizer_id)
        ts_parameters = optimizer_config['entity']['document']['pipelines'][0]['nodes'][0]['parameters']['optimization']
        print(ts_parameters)
        self.assertIn('retrain_on_holdout', ts_parameters)

    def test_06_get_train_data(self):
        TestAutoAIRemote.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

        TestAutoAIRemote.train_X = self.train_data.drop(columns=[self.experiment_info.get('timestamp_column_name')])
        TestAutoAIRemote.scoring_df = self.train_X[:10]
        print("Scoring data sample:")
        print(self.scoring_df.head())

    def test_06a_get_train_data(self):
        TestAutoAIRemote.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

        TestAutoAIRemote.train_X = self.train_data.drop(columns=[self.experiment_info.get('timestamp_column_name')])
        TestAutoAIRemote.scoring_df = self.train_X[:10]
        print("Scoring data sample:")
        print(self.scoring_df.head())

    def test_06a_get_train_data_with_holdout_raises_error(self):
        TestAutoAIRemote.train_X, TestAutoAIRemote.test_X, TestAutoAIRemote.train_y, TestAutoAIRemote.test_y = \
            self.remote_auto_pipelines.get_data_connections()[0].read(with_holdout_split=True)
        print("train data sample:")
        print(self.train_X.head())
        self.assertGreater(len(self.train_X), 0)

    def test_99_delete_connection(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)

        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
