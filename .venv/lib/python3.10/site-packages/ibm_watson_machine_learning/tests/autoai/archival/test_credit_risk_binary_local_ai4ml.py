#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import pandas as pd

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d, bucket_exists, bucket_name_gen
from ibm_watson_machine_learning.experiment.autoai.optimizers import LocalAutoPipelines

from sklearn.pipeline import Pipeline


class TestAutoAILocal(unittest.TestCase):
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    local_optimizer = None
    data_location = './autoai/data/bank.csv'
    target_column = 'y'
    train_data = None
    test_X = None
    test_y = None
    pipeline_name = 'Pipeline_3'
    OPTIMIZER_NAME = 'Local ai4ml'

    @classmethod
    def setUp(cls) -> None:
        cls.experiment = AutoAI()
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop([cls.target_column], axis=1)
        cls.y = cls.data[cls.target_column]

    def test_01_initialize_optimizer(self):
        TestAutoAILocal.local_optimizer = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            prediction_type=AutoAI.PredictionType.MULTICLASS,
            prediction_column=self.target_column,
            scoring=AutoAI.Metrics.ACCURACY_SCORE,
            _force_local_scenario=True
        )

        self.assertIsInstance(self.local_optimizer, LocalAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_02_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.local_optimizer.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_03_fit_optimizer(self):
        best_model = self.local_optimizer.fit(X=TestAutoAILocal.X, y=TestAutoAILocal.y)
        self.assertIsInstance(best_model, Pipeline)

    def test_04_get_pipeline_model_details(self):
        details = self.local_optimizer.get_pipeline_details(pipeline_name='Pipeline_2')
        self.assertIn('pipeline_nodes', details)

    def test_05_get_holdout_data(self):
        TestAutoAILocal.test_X, TestAutoAILocal.test_y = self.local_optimizer.get_holdout_data()

        print("holdout data sample:")
        print(self.test_X.head())
        self.assertGreater(len(self.test_X), 0)

    def test_06_get_pipeline_model(self):
        pipeline_model = self.local_optimizer.get_pipeline(pipeline_name=TestAutoAILocal.pipeline_name, astype='sklearn')
        results = pipeline_model.predict(X=TestAutoAILocal.test_X.values)

        self.assertIsInstance(pipeline_model, Pipeline)
        print(results)
        self.assertGreater(len(results), 0)

    def test_06_predict_with_optimizer(self):
        results = self.local_optimizer.predict(X=TestAutoAILocal.test_X.values)
        self.assertGreater(len(results), 0)


    #################################
    #      DEPLOYMENT SECTION       #
    #################################

    # def test_21_deployment_setup_and_preparation(self):
    #     if is_cp4d():
    #         TestAutoAILocal.service = WebService(wml_credentials=self.wml_credentials.copy(),
    #                                              project_id=self.project_id,
    #                                              space_id=self.space_id)
    #         self.wml_client.set.default_space(self.space_id)
    #     else:
    #         TestAutoAILocal.service = WebService(wml_credentials=self.wml_credentials)
    #
    # def test_22__deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
    #     self.service.create(
    #         experiment_run_id=self.run_id,
    #         model=self.new_pipeline,
    #         deployment_name=self.DEPLOYMENT_NAME)
    #
    # def test_24_score_deployed_model(self):
    #     nb_records = 10
    #     predictions = self.service.score(payload=self.holdout_data.drop([self.target_column], axis=1)[:nb_records])
    #     print(predictions)
    #     self.assertIsNotNone(predictions)
    #     self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)
    #
    #
    # def test_25_delete_deployment(self):
    #     print("Delete current deployment: {}".format(self.service.deployment_id))
    #     self.service.delete()

if __name__ == '__main__':
    unittest.main()
