#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import uuid

from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import create_connection_to_cos
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAIAsync, BaseTestStoreModel

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms


class TestAutoAIRemote(AbstractTestAutoAIAsync, BaseTestStoreModel, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = './autoai/data/drug_train_data_updated.csv'

    data_cos_path = 'data/drug_train_data_updated.csv'

    batch_payload_location = './autoai/data/scoring_payload/drug_train_data_updated_scoring_payload.csv'
    batch_payload_cos_location = 'scoring_payload/drug_train_data_updated_scoring_payload.csv'

    pipeline_to_deploy = "Pipeline_1"

    SPACE_ONLY = False

    OPTIMIZER_NAME = "Drug data test sdk"

    BATCH_DEPLOYMENT_WITH_DF = True
    BATCH_DEPLOYMENT_WITH_DA = False
    HISTORICAL_RUNS_CHECK = False

    target_space_id = None

    fairness_info = {
        "favorable_labels": ["drugA", "drugC"]
    }

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc='FAIRNESS experiment',
        prediction_type=PredictionType.MULTICLASS,
        prediction_column='DRUG',
        scoring=Metrics.RECALL_SCORE_WEIGHTED,
        include_only_estimators=[ClassificationAlgorithms.LGBM,
                                 ClassificationAlgorithms.XGB],
        fairness_info=fairness_info,
        max_number_of_estimators=2,
        text_processing=False,
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
        TestAutoAIRemote.results_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.results_cos_path
            )
        )

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNotNone(obj=TestAutoAIRemote.results_connection)

    def test_10_summary_listing_all_pipelines_from_wml(self):
        TestAutoAIRemote.summary = self.remote_auto_pipelines.summary()
        print(TestAutoAIRemote.summary)

        for col in self.summary.columns:
            print(self.summary[col])

        self.assertIn('holdout_disparate_impact', list(TestAutoAIRemote.summary.columns))
        self.assertIn('holdout_disparate_impact_SEX', list(TestAutoAIRemote.summary.columns))
        self.assertIn('holdout_disparate_impact_AGE', list(TestAutoAIRemote.summary.columns))
        self.assertIn('training_disparate_impact_AGE', list(TestAutoAIRemote.summary.columns))


    def test_99_delete_connection_and_connected_data_asset(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
