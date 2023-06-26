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
    BatchedClassificationAlgorithms
from ibm_watson_machine_learning.tests.utils import is_cp4d


@unittest.skipIf(not is_cp4d(), "Batched Tree Ensembles not supported yet on cloud")
class TestAutoAIRemote(AbstractTestAutoAIAsync, unittest.TestCase):
    """
    The test can be run on CPD only
    """

    cos_resource = None
    data_location = './autoai/data/make_class_header.csv'

    data_cos_path = 'data/make_class_header.csv'

    SPACE_ONLY = False

    OPTIMIZER_NAME = "make_class_header test sdk"

    target_space_id = None
    df = None
    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.BINARY,
        prediction_column='class',
        scoring=Metrics.ROC_AUC_SCORE,
        max_number_of_estimators=1,
        include_only_estimators=["SnapRandomForestClassifier", "XGBClassifier", "LogisticRegression",
                                 "DecisionTreeClassifier"],
        include_batched_ensemble_estimators=["BatchedTreeEnsembleClassifier(SnapRandomForestClassifier)",
                                             "BatchedTreeEnsembleClassifier(XGBClassifier)"]
        #use_flight=True
    )

    data_loader = None

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
            pipeline_model = pipeline_model.partial_fit(self.X_values, self.y_values, classes=[0, 1])
            print('score: ', scorer(pipeline_model, self.X_values, self.y_values))

    def test_91_data_loader(self):
        from ibm_watson_machine_learning.data_loaders import experiment as data_loaders
        from ibm_watson_machine_learning.data_loaders.datasets import experiment as datasets

        TestAutoAIRemote.experiment_info = dict(
            prediction_type=self.remote_auto_pipelines.get_params()['prediction_type'],
            prediction_column=self.remote_auto_pipelines.get_params()['prediction_column'],
            scoring=self.remote_auto_pipelines.get_params()['scoring'],
            training_data_references=[self.data_connection],
            project_id=self.wml_credentials['project_id'],
            positive_label=self.remote_auto_pipelines.get_params()['positive_label'],
            classes=[0, 1],
        )

        dataset = datasets.ExperimentIterableDataset(
            connection=self.data_connection,
            with_subsampling=False,
            experiment_metadata=self.experiment_info,
            _wml_client=self.wml_client
        )

        TestAutoAIRemote.data_loader = data_loaders.ExperimentDataLoader(dataset=dataset)

    def test_92_partial_fit(self):
        from sklearn.metrics import get_scorer

        scorer = get_scorer(self.experiment_info['scoring'])

        TestAutoAIRemote.experiment_info['classes'] = [0, 1]

        X_train_first_10 = None

        pipeline_model = self.remote_auto_pipelines.get_pipeline('Pipeline_5', astype='lale')

        for batch_df in self.data_loader:
            print(batch_df.shape)
            X_train = batch_df.drop([self.experiment_info['prediction_column']], axis=1).values
            y_train = batch_df[self.experiment_info['prediction_column']].values
            if not X_train_first_10:
                X_train_first_10 = X_train[:10]
            pipeline_model = pipeline_model.partial_fit(X_train, y_train, classes=self.experiment_info['classes'],
                                                        freeze_trained_prefix=True)
            print('score: ', scorer(pipeline_model, X_train, y_train))

        pipeline_model.predict(X_train_first_10)

    def test_99_delete_data_asset(self):
        if not self.SPACE_ONLY:
            self.wml_client.set.default_project(self.project_id)

        self.wml_client.data_assets.delete(self.asset_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.data_assets.get_details(self.asset_id)


if __name__ == '__main__':
    unittest.main()
