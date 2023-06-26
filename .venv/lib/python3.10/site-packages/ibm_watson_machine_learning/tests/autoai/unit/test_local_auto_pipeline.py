#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import pandas as pd
from sklearn.pipeline import Pipeline
from lale.operators import TrainablePipeline

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.utils.autoai.enums import (
    PredictionType, Metrics, ClassificationAlgorithms, PipelineTypes)
from ibm_watson_machine_learning.experiment.autoai.optimizers import LocalAutoPipelines


class TestLocalAutoPipelines(unittest.TestCase):
    data_location = './autoai/data/bank.csv'
    local_optimizer: 'LocalAutoPipelines' = None
    best_pipeline_model: 'Pipeline' = None
    pipeline = None

    @classmethod
    def setUp(cls) -> None:
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['y'], axis=1)
        cls.y = cls.data['y']

    def test_01_initialize_LocalAutoPipelines__LocalAutoPipelines_initialized(self):
        TestLocalAutoPipelines.local_optimizer = AutoAI().optimizer(
            name='Local optimization',
            desc='Test local optimization',
            prediction_type=PredictionType.BINARY,
            prediction_column='state_code',
            scoring=Metrics.ROC_AUC_SCORE,
            test_size=0.2,
            max_number_of_estimators=1,
            daub_include_only_estimators=[
                ClassificationAlgorithms.LGBM,
                ClassificationAlgorithms.DT,
                ClassificationAlgorithms.XGB,
                ClassificationAlgorithms.GB,
                ClassificationAlgorithms.EX_TREES
            ]
        )

        self.assertIsInstance(self.local_optimizer, LocalAutoPipelines, msg="Local Pipeline Optimizer not initialized")

    def test_02__fit__start_local_training__local_training_finished_and_pipeline_model_returned(self):
        TestLocalAutoPipelines.best_pipeline_model = self.local_optimizer.fit(X=self.X, y=self.y)

        self.assertIsInstance(self.best_pipeline_model, Pipeline, msg="Computed best pipeline is not of sklearn type.")

    def test_03__get_params__get_configuration_parameters_of_local_auto_pipeline__parameters_listed(self):
        parameters = self.local_optimizer.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_04_predict__predict_using_fitted_pipeline__predictions_are_computed(self):
        predictions = self.local_optimizer.predict(X=self.X.values)
        print(predictions)

    def test_05__results__listing_all_pipelines__all_pipelines_listed_with_metrics(self):
        pipelines_details = self.local_optimizer.summary()
        print(pipelines_details)

    def test_06__get_pipeline_params__list_specific_pipeline_parameters__parameters_fetched_as_dict(self):
        pipeline_params = self.local_optimizer.get_pipeline_details(pipeline_name='Pipeline_1')
        print(pipeline_params)

    def test_07__get_pipeline_params__fetch_best_pipeline_parameters__parameters_fetched_as_dict(self):
        best_pipeline_params = self.local_optimizer.get_pipeline_details()
        print(best_pipeline_params)

    def test_08__get_pipeline__load_specified_pipeline__pipeline_loaded(self):
        TestLocalAutoPipelines.pipeline = self.local_optimizer.get_pipeline(pipeline_name='Pipeline_3')
        print(f"Fetched pipeline type: {type(self.pipeline)}")

        self.assertIsInstance(self.pipeline, TrainablePipeline,
                              msg="Fetched pipeline is not of TrainablePipeline instance.")

    def test_09__predict__do_the_predict_on_lale_pipeline__results_computed(self):
        predictions = self.pipeline.predict(self.X.values)
        print(predictions)

    def test_10__get_pipeline__load_specified_pipeline__sklearn_pipeline_loaded(self):
        TestLocalAutoPipelines.pipeline = self.local_optimizer.get_pipeline(
            pipeline_name='Pipeline_3', astype=PipelineTypes.SKLEARN)
        print(f"Fetched pipeline type: {type(self.pipeline)}")

        self.assertIsInstance(self.pipeline, Pipeline, msg="Fetched pipeline is not of sklearn type.")


if __name__ == '__main__':
    unittest.main()
