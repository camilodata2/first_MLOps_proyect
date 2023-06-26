#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from os.path import join
import uuid

from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import bucket_exists, create_bucket, is_cp4d, create_connection_to_cos
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAIAsync, \
    AbstractTestWebservice, AbstractTestBatch

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, RegressionAlgorithms, \
    BatchedRegressionAlgorithms


class TestAutoAIRemote(AbstractTestAutoAIAsync, AbstractTestWebservice, AbstractTestBatch, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = './autoai/data/Hotel_Reviews_1MB.csv'
    data_encoding = "utf_8"

    data_cos_path = 'data/Hotel_Reviews.csv'

    batch_payload_location = './autoai/data/scoring_payload/Hotel_Reviews_1MB_scoring_payload.csv'
    batch_payload_cos_location = 'scoring_payload/Hotel_Reviews_1MB_scoring_payload.csv'

    SPACE_ONLY = True

    OPTIMIZER_NAME = "Hotel Reviews text data test sdk"

    BATCH_DEPLOYMENT_WITH_DF = True
    BATCH_DEPLOYMENT_WITH_DA = False
    HISTORICAL_RUNS_CHECK = False

    target_space_id = None


    experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc='Text Transformer experiment',
        prediction_type=PredictionType.REGRESSION,
        prediction_column='Reviewer_Score',
        scoring=Metrics.ROOT_MEAN_SQUARED_ERROR,
        max_number_of_estimators=2,
        text_processing=True,

        text_columns_names=['Negative_Review', 'Positive_Review'],
        word2vec_feature_number=5,
        include_only_estimators=[RegressionAlgorithms.SnapBM, RegressionAlgorithms.XGB],
        include_batched_ensemble_estimators=[BatchedRegressionAlgorithms.SnapBM,
                                             BatchedRegressionAlgorithms.XGB],
        # encoding=data_encoding
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

    def test_00d_prepare_connected_data_asset(self):
        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Hotel Reviewers utf16 - training asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: join(self.bucket_name,
                                                                                       self.data_cos_path)
        })
        TestAutoAIRemote.asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.asset_id, str)

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(data_asset_id=self.asset_id)
        TestAutoAIRemote.results_connection = None

        # TestAutoAIRemote.results_connection = DataConnection(
        #     connection_asset_id=self.connection_id,
        #     location=S3Location(
        #         bucket=self.bucket_name,
        #         path=self.results_cos_path
        #     )
        # )

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        # self.assertIsNotNone(obj=TestAutoAIRemote.results_connection)

    def test_08a_get_feature_importance(self):
        feature_importance_df = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_4').get(
            'features_importance')

        self.assertIsNotNone(feature_importance_df)

        str_feature_importance_index_list = str(list(feature_importance_df.sort_index().index))

        text_columns = self.experiment_info['text_columns_names']

        for column in text_columns:
            self.assertIn(f'word2vec({column})', str_feature_importance_index_list,
                          msg=f"word2vec({column}) is not in features importance table. Full table: {feature_importance_df}")

        self.assertIn('NewTextFeature_0', str_feature_importance_index_list,
                      msg="Text features were numerated incorrectly.")
        self.assertIn(f"NewTextFeature_{self.experiment_info['word2vec_feature_number'] * len(text_columns) - 1}",
                      str_feature_importance_index_list, msg="Text features were numerated incorrectly.")

    def test_99_delete_connection_and_connected_data_asset(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)
        self.wml_client.connections.delete(self.connection_id)
        self.wml_client.data_assets.delete(self.asset_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
