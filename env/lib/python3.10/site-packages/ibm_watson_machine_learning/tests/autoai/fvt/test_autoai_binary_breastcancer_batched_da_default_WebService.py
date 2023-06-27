#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from sklearn.metrics import get_scorer
from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAIAsync, \
    AbstractTestWebservice

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms, \
    BatchedClassificationAlgorithms, SamplingTypes
from ibm_watson_machine_learning.tests.utils import is_cp4d


@unittest.skipIf(not is_cp4d(), "Batched Tree Ensembles not supported yet on cloud")
class TestAutoAIRemote(AbstractTestAutoAIAsync, AbstractTestWebservice, unittest.TestCase):
    """
    The test can be run on CPD only
    """

    cos_resource = None
    data_location = './autoai/data/breast_cancer.csv'

    data_cos_path = 'data/breast_cancer.csv'

    SPACE_ONLY = False

    OPTIMIZER_NAME = "breast_cancer test sdk"

    target_space_id = None
    df = None
    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.BINARY,
        prediction_column='diagnosis',
        positive_label='M',
        scoring=Metrics.AVERAGE_PRECISION_SCORE,
        max_number_of_estimators=1,
        sampling_type=SamplingTypes.STRATIFIED,
        number_of_batch_rows=100,
        sample_size_limit=1000000, #1000k bytes
        include_only_estimators=[ClassificationAlgorithms.SnapRF, ClassificationAlgorithms.LGBM],
        include_batched_ensemble_estimators=[BatchedClassificationAlgorithms.SnapRF,
                                             BatchedClassificationAlgorithms.LGBM],
        use_flight=True
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

    def test_02a_read_saved_remote_data_before_fit(self):
        self.data_connection.set_client(self.wml_client)
        data = self.data_connection.read(raw=True)
        print("data sample:")
        print(data.head())
        self.assertGreater(len(data), 0)

    def test_10_summary_listing_all_pipelines_from_wml(self):
        TestAutoAIRemote.summary = self.remote_auto_pipelines.summary()
        print(TestAutoAIRemote.summary)
        self.assertIn('Ensemble', TestAutoAIRemote.summary.Enhancements['Pipeline_5'])

    def test_90_partial_fit_BatchedTreeEnsmblePipeline(self):
        pipeline_model = self.remote_auto_pipelines.get_pipeline('Pipeline_5', astype='lale')
        estimator = pipeline_model.steps[-1][1]
        pipeline_model = pipeline_model.remove_last().freeze_trained() >> estimator
        scorer = get_scorer(TestAutoAIRemote.experiment_info['scoring'])

        for i in range(3):
            pipeline_model = pipeline_model.partial_fit(self.X_values, self.y_values, classes=['M', 'B'])
            print('score: ', scorer(pipeline_model, self.X_values, self.y_values))

    def test_91_batch_sizes(self):
        self.assertEqual(self.remote_auto_pipelines.params['number_of_batch_rows'], self.experiment_info['number_of_batch_rows'])
        self.assertEqual(self.remote_auto_pipelines.params['sample_size_limit'], self.experiment_info['sample_size_limit'])

    def test_92_data_loader(self):
        from ibm_watson_machine_learning.data_loaders import experiment as data_loaders
        from ibm_watson_machine_learning.data_loaders.datasets import experiment as datasets
        from ibm_watson_machine_learning.helpers import DataConnection

        asset_id = 'd289da78-0ccf-413d-a24f-10fea2dedd90'
        training_large_data_conn = DataConnection(data_asset_id=asset_id)
        self.experiment_info['project_id'] = '6d0af73b-575a-44e6-b18e-751a7591591d'

        dataset = datasets.ExperimentIterableDataset(
            connection=training_large_data_conn,
            with_subsampling=False,
            experiment_metadata=self.experiment_info,
            _wml_client=self.wml_client
        )

        data_loader = data_loaders.ExperimentDataLoader(dataset=dataset)

    def test_99_delete_data_asset(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)

        self.wml_client.data_assets.delete(self.asset_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.data_assets.get_details(self.asset_id)


if __name__ == '__main__':
    unittest.main()
