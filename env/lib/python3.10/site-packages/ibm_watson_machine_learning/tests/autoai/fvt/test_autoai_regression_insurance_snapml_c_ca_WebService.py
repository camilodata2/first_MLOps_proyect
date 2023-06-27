#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location, ContainerLocation
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import save_data_to_container, create_bucket, is_cp4d, \
    create_connection_to_cos
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAIAsync, \
    AbstractTestWebservice, AbstractTestBatch

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, RegressionAlgorithms


class TestAutoAIRemote(AbstractTestAutoAIAsync, AbstractTestWebservice, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = './autoai/data/insurance.csv'

    data_cos_path = 'insurance.csv'
    results_cos_path = "results"
    # batch_payload_location = './autoai/data/scoring_payload/drug_train_data_updated_scoring_payload.csv'
    # batch_payload_cos_location = 'scoring_payload/drug_train_data_updated_scoring_payload.csv'

    SPACE_ONLY = True

    OPTIMIZER_NAME = "Insurance data Fairness test sdk"

    # BATCH_DEPLOYMENT_WITH_DF = True
    # BATCH_DEPLOYMENT_WITH_DA = False
    HISTORICAL_RUNS_CHECK = True

    target_space_id = None

    fairness_info = {'favorable_labels': [[5000, 50000]],
                     'unfavorable_labels': [[0, 4999], [50001, 1000000]],
                     'protected_attributes': [
                         {'feature': 'age', 'monitored_group': [[18, 38]], 'reference_group': [[39, 64]]},
                         {'feature': 'sex', 'monitored_group': ['female'], 'reference_group': ['male']}]
                     }

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc='FAIRNESS experiment',
        prediction_type=PredictionType.REGRESSION,
        prediction_column='charges',
        scoring=Metrics.R2_AND_DISPARATE_IMPACT_SCORE,
        include_only_estimators=[RegressionAlgorithms.SnapRF,
                                 RegressionAlgorithms.SnapBM],
        fairness_info=fairness_info,
        max_number_of_estimators=2,
        notebooks=True,
        text_processing=True,
        retrain_on_holdout=False
    )

    def test_00b_write_data_to_container(self):
        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

        save_data_to_container(self.data_location, self.data_cos_path, self.wml_client)

    def test_00c_prepare_COS_instance_and_connection(self):
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
            location=ContainerLocation(path=self.data_cos_path
                                       ))

        TestAutoAIRemote.results_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.results_cos_path
            )
        )

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNotNone(obj=TestAutoAIRemote.results_connection)

    def test_02a_read_data_before_fit(self):
        self.data_connection.set_client(self.wml_client)
        data = self.data_connection.read()
        print(data)

        self.assertGreater(len(data), 0)

    def test_06_get_train_data(self):
        X_train, X_holdout, y_train, y_holdout = self.remote_auto_pipelines.get_data_connections()[0].read(
            with_holdout_split=True)

        print("train data sample:")
        print(X_train)
        print(y_train)
        print("holdout data sample:")
        print(X_holdout)
        print(y_holdout)

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_holdout), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_holdout), len(y_holdout))

        AbstractTestAutoAIAsync.X_df = X_holdout
        AbstractTestAutoAIAsync.X_values = AbstractTestAutoAIAsync.X_df.values
        AbstractTestAutoAIAsync.y_values = y_holdout

    def test_10_summary_listing_all_pipelines_from_wml(self):
        TestAutoAIRemote.summary = self.remote_auto_pipelines.summary()
        print(TestAutoAIRemote.summary)

        for col in self.summary.columns:
            print(self.summary[col])

        if self.experiment_info['scoring'] == self.experiment.Metrics.R2_AND_DISPARATE_IMPACT_SCORE:
            self.assertIn('training_r2_and_disparate_impact_(optimized)', list(TestAutoAIRemote.summary.columns))
            self.assertIn('holdout_r2_and_disparate_impact', list(TestAutoAIRemote.summary.columns))

        self.assertIn('holdout_disparate_impact', list(TestAutoAIRemote.summary.columns))
        # self.assertIn('training_disparate_impact', list(TestAutoAIRemote.summary.columns))


        for feature in ['sex', 'age']:
            self.assertIn(f'holdout_disparate_impact_{feature}', list(TestAutoAIRemote.summary.columns))
            # self.assertIn(f'training_disparate_impact_{feature}', list(TestAutoAIRemote.summary.columns))

    def test_16a_get_params_of_last_historical_run(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")
        experiment_run_params = self.experiment.runs.get_params(run_id=self.run_id)

        print(experiment_run_params)

        for param, value in self.experiment_info.items():
            if param == 'include_only_estimators':
                experiment_info_estimators = [estimator.value for estimator in value if not isinstance(estimator, str)]
                self.assertEqual(experiment_info_estimators, experiment_run_params[param])
            elif param == 'max_number_of_estimators':
                self.assertEqual(value, experiment_run_params['max_num_daub_ensembles'])
            elif param in ['notebooks', 'autoai_pod_version']:
                continue
            else:
                self.assertIn(param, experiment_run_params,
                              msg=f"{param} field not fount in run_params. Run_params are: {experiment_run_params}")
                self.assertEqual(value, experiment_run_params[param])

    def test_16b_get_params_of_last_historical_run(self):
        self.skipTest("Not ready.")
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")

        optimizer_run_params = self.remote_auto_pipelines.get_params()
        experiment_run_params = self.experiment.runs.get_params(run_id=self.run_id)
        print("Optimizer params self.remote_auto_pipelines.get_params(): ")
        print(optimizer_run_params)

        print("\nParams from experiment AutoAI  sself.experiment.runs.get_params(run_id=self.run_id): ")
        print(experiment_run_params)

        skip_validation_params = ['t_shirt_size']

        for param, value in optimizer_run_params.items():
            if param not in skip_validation_params:
                self.assertIn(param, experiment_run_params,
                              msg=f"{param} field not fount in run_params. Run_params are: {experiment_run_params}")
                if param == 'include_only_estimators':
                    experiment_info_estimators = [estimator.value for estimator in value if not isinstance(estimator, str)]
                    self.assertEqual(experiment_info_estimators, experiment_run_params[param])
                else:
                    self.assertEqual(value, experiment_run_params[param])

    def test_99_delete_connection_and_connected_data_asset(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)

if __name__ == '__main__':
    unittest.main()
