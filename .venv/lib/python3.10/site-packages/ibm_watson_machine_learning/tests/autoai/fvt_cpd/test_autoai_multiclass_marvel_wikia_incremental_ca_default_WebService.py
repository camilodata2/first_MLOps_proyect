#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import pandas as pd
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestWebservice, \
    AbstractTestAutoAIAsync

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms, \
    SamplingTypes
from ibm_watson_machine_learning.tests.utils import create_connection_to_cos


class TestAutoAIRemote(AbstractTestAutoAIAsync, AbstractTestWebservice, unittest.TestCase):
    """
    The test can be run on CP4D (data asset with csv file)
    """

    cos_resource = None
    data_location = './autoai/data/marvel-wikia-data.csv'

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'data/marvel-wikia-data.csv'

    SPACE_ONLY = False

    OPTIMIZER_NAME = "MarvelWikia AutoAI SDK  test"
    INCREMENTAL_TRAINING = True

    target_space_id = None
    asset_id = None

    expected_iterations = 11
    expected_classes = 3
    INCREMENTAL_PIPELINES_EXPECTED = True

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc='test description',
        prediction_type=PredictionType.MULTICLASS,
        prediction_column='ALIGN',
        scoring=Metrics.F1_SCORE_MACRO,
        sampling_type=SamplingTypes.STRATIFIED,
        sample_size_limit=1 * 1024 * 1024,  # 10 MB
        early_stop_enabled=True,
        early_stop_window_size=5,
        max_number_of_estimators=2,
        include_only_estimators=[ClassificationAlgorithms.RF, ClassificationAlgorithms.EX_TREES,
                                 ClassificationAlgorithms.LGBM, ClassificationAlgorithms.XGB,
                                 ClassificationAlgorithms.SnapRF, ClassificationAlgorithms.SnapBM]
        # only estimators with with batche ensamble
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
        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

        TestAutoAIRemote.results_connection = None
        self.assertIsNone(obj=TestAutoAIRemote.results_connection)

    def test_12a_test_pipeline_incremental_learning_curve(self):
        summary = self.remote_auto_pipelines.summary()
        incremental_pipelines_names = summary.index[
            summary['Enhancements'].str.contains('Incremental_learning')].tolist()

        required_elements = ['ml_metrics', 'learning_curve', 'features_importance', 'roc_curve', 'confusion_matrix']

        for pipeline_name in incremental_pipelines_names:
            pipeline_details = self.remote_auto_pipelines.get_pipeline_details(pipeline_name)

            for key in required_elements:
                self.assertIn(key, pipeline_details,
                              msg=f"Missing key {key} in {pipeline_name}'s details: {str(pipeline_details)}")
                self.assertIsInstance(pipeline_details[key], pd.DataFrame,
                                      f"The {key} is not pd.DataFrame, it's: {pipeline_details[key]}")
                self.assertFalse(pipeline_details[key].empty,
                                 f"The {key} was not generated/ read correct for the pipeline: {pipeline_name}")

            learning_curve_table = pipeline_details['learning_curve']
            self.assertTrue(len(learning_curve_table) == self.expected_iterations)
            self.assertIn('partial_fit_time', learning_curve_table.columns)
            self.assertIn('iteration', learning_curve_table.columns)
            self.assertIn('batch_size', learning_curve_table.columns)

            confusion_matrix = pipeline_details['confusion_matrix']
            self.assertEqual(confusion_matrix.shape, (self.expected_classes, 4))

    def test_99_delete_connection_and_connected_data_asset(self):
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
