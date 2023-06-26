#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import ibm_boto3
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import bucket_exists, create_bucket, is_cp4d
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestTSAsync, \
     BaseTestStoreModel

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, PipelineTypes


class TestAutoAIRemote(AbstractTestTSAsync, BaseTestStoreModel, unittest.TestCase):
    """
    The test can be run on Cloud only
    """

    cos_resource = None
    data_location = './autoai/data/Twitter_volume_AMZN.csv'

    data_cos_path = 'Twitter_volume_AMZN.csv'

    batch_payload_location = data_location
    batch_payload_cos_location = data_cos_path

    SPACE_ONLY = True
    BATCH_DEPLOYMENT = False

    OPTIMIZER_NAME = "Twitter_volume_AMZN test sdk"
    DEPLOYMENT_NAME = OPTIMIZER_NAME + "Deployment"

    pipeline_to_deploy = None

    target_space_id = None


    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.FORECASTING,
        prediction_columns=['value'],
        timestamp_column_name='timestamp',
        backtest_num=4,
        lookback_window=-1,
        forecast_window=2,
        holdout_size=100,
        max_number_of_estimators=1,
        notebooks=True
    )

    def test_00d_prepare_data_asset(self):
        asset_details = self.wml_client.data_assets.create(
            name=self.data_location.split('/')[-1],
            file_path=self.data_location)

        TestAutoAIRemote.asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.asset_id, str)

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(data_asset_id=self.asset_id)
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNone(obj=TestAutoAIRemote.results_connection)

    def test_09a_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.scoring_df[:5].values)
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_09ab_predict_using_fitted_pipeline(self):
        pipelines_summary = self.remote_auto_pipelines.summary()
        pipeline_name = list(pipelines_summary.index)[-1]
        TestAutoAIRemote.pipeline_to_deploy = pipeline_name
        pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name, persist=True, astype=PipelineTypes.LALE)
        print(pipeline.pretty_print())
        predictions = pipeline.predict(X=self.scoring_df[:5].values)
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_09b_get_pipeline_optimizer_from_experiment_metdata(self):
        from ibm_watson_machine_learning.helpers import ContainerLocation

        training_data_references = [
            DataConnection(
                data_asset_id=self.asset_id
            ),
        ]
        training_result_reference = DataConnection(
            location=ContainerLocation(
                path=f'default_autoai_out/{self.run_id}/data/autoai-ts',
                model_location=f'default_autoai_out/{self.run_id}/data/autoai-ts/model.zip',
                training_status=f'default_autoai_out/{self.run_id}/training-status.json'
            )
        )

        experiment_metadata = dict(
            prediction_type='forecasting',
            prediction_columns=['value'],
            csv_separator=',',
            holdout_size=100,
            training_data_reference=training_data_references,
            training_result_reference=training_result_reference,
            timestamp_column_name='timestamp',
            backtest_num=4,
            lookback_window=5,
            forecast_window=2,
            max_num_daub_ensembles=1,
            deployment_url='https://yp-qa.ml.cloud.ibm.com',
            space_id=self.space_id
        )
        summary = self.remote_auto_pipelines.summary()
        best_pipeline_name = list(summary.index)[0]

        from ibm_watson_machine_learning.deployment import WebService

        service = WebService(
            source_wml_credentials=self.wml_credentials,
            target_wml_credentials=self.wml_credentials,
            source_space_id=experiment_metadata['space_id'],
            target_space_id=experiment_metadata['space_id']
        )
        service.create(
            model=best_pipeline_name,
            metadata=experiment_metadata,
            deployment_name='Best_pipeline_webservice'
        )

        self.assertIsNotNone(service)

    def test_09c_get_pipeline_optimizer_from_experiment_metdata(self):
        from ibm_watson_machine_learning.helpers import ContainerLocation

        training_data_references = [
            DataConnection(
                data_asset_id=self.asset_id
            ),
        ]
        training_result_reference = DataConnection(
            location=ContainerLocation(
                path=f'default_autoai_out/{self.run_id}/data/automl',
                model_location=f'default_autoai_out/{self.run_id}/data/automl/model.zip',
                training_status=f'default_autoai_out/{self.run_id}/training-status.json'
            )
        )

        experiment_metadata = dict(
            prediction_type='forecasting',
            prediction_columns=['value'],
            csv_separator=',',
            holdout_size=100,
            training_data_reference=training_data_references,
            training_result_reference=training_result_reference,
            timestamp_column_name='timestamp',
            backtest_num=4,
            lookback_window=5,
            forecast_window=2,
            max_num_daub_ensembles=1,
            deployment_url='https://yp-qa.ml.cloud.ibm.com',
            space_id=self.space_id
        )
        pipeline_optimizer = AutoAI().runs.get_optimizer(metadata=experiment_metadata,
                                                         wml_client=self.remote_auto_pipelines._workspace.wml_client)

        self.assertIsNotNone(pipeline_optimizer)

    def test_99_delete_data_asset(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)

        self.wml_client.data_assets.delete(self.asset_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.data_assets.get_details(self.asset_id)


if __name__ == '__main__':
    unittest.main()
