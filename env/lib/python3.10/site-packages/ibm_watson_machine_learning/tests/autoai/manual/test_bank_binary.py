#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from sklearn.pipeline import Pipeline
from pprint import pprint
import pandas as pd
import traceback

from ibm_watson_machine_learning import APIClient

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService

from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, AssetLocation

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines

from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d, get_env
from ibm_watson_machine_learning.tests.utils.cleanup import delete_model_deployment

from lale import wrap_imported_operators
from lale.operators import TrainablePipeline
from lale.lib.lale import Hyperopt


@unittest.skipIf(is_cp4d(), "Not supported on CP4D")
class TestAutoAIRemote(unittest.TestCase):
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    service: 'WebService' = None

    data_location = './autoai/data/bank.csv'

    trained_pipeline_details = None
    run_id = None

    data_connection = None
    results_connection = None
    sklearn_pipeline = None

    train_data = None
    holdout_data = None

    cos_endpoint = None
    bucket_name = None
    data_cos_path = 'data/bank.csv'

    project_id = None
    space_id = None

    results_cos_path = None

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None

    OPTIMIZER_NAME = 'Bank wml_autoai binary test'
    DEPLOYMENT_NAME = "Bank AutoAI Deployment tests"

    asset_id = None

    @classmethod
    def setUp(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['y'], axis=1)
        cls.y = cls.data['y']

        cls.wml_credentials = get_wml_credentials()
        if not is_cp4d():
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials['endpoint_url']
            cls.bucket_name = cls.cos_credentials['bucket_name']
            cls.results_cos_path = cls.cos_credentials['results_cos_path']

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        if is_cp4d():
            TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                 project_id=self.project_id,
                                                 space_id=self.space_id)
        else:
            TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_save_remote_data_and_DataConnection_setup(self):
        if is_cp4d():
            self.wml_client.set.default_project(self.project_id)
            asset_details = self.wml_client.data_assets.create(
                name=self.data_location.split('/')[-1],
                file_path=self.data_location)
            asset_id = asset_details['metadata']['guid']

            TestAutoAIRemote.data_connection = DataConnection(
                location=AssetLocation(asset_id=asset_id))
            TestAutoAIRemote.results_connection = None

        else:  # for cloud and COS
            TestAutoAIRemote.data_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['access_key_id'],
                                        secret_access_key=self.cos_credentials['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path)
            )
            TestAutoAIRemote.results_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['access_key_id'],
                                        secret_access_key=self.cos_credentials['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.results_cos_path)
            )
            TestAutoAIRemote.data_connection.write(data=self.data, remote_name=self.data_cos_path)

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    def __make_experiment(self, name, scoring, test_size, max_est, daub_est):
        print('###############################################################')
        print(f"Preparing experiment: {name}")
        print('###############################################################')
        optimizer = self.experiment.optimizer(
            name=name,
            desc='',
            prediction_type=AutoAI.PredictionType.BINARY,
            prediction_column='y',
            scoring=scoring,
            test_size=test_size,
            max_number_of_estimators=max_est,
            daub_include_only_estimators=daub_est
        )

        print('###############################################################')
        print('Fetching experiment parameters...')
        print('###############################################################')
        parameters = optimizer.get_params()
        print(f'Optimizer parameters: {parameters}')
        print(parameters)

        print('###############################################################')
        print('Starting training...')
        print('###############################################################')
        optimizer.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=False)

        print('###############################################################')
        print('Fetching training data...')
        print('###############################################################')
        TestAutoAIRemote.train_data, TestAutoAIRemote.holdout_data = optimizer.get_data_connections()[
            0].read(with_holdout_split=True)

        print("train data sample:")
        print(self.train_data.head())

        print("holdout data sample:")
        print(self.holdout_data.head())

        print('###############################################################')
        print('Fetching training status...')
        print('###############################################################')
        status = optimizer.get_run_status()
        print(f'Training status: {status}')

        print('###############################################################')
        print('Fetching training details...')
        print('###############################################################')
        parameters = optimizer.get_run_details()
        print(f'Training run details: {parameters}')

        print('###############################################################')
        print('Make prediction on the best pipeline...')
        print('###############################################################')
        predictions = optimizer.predict(X=self.holdout_data.drop(['y'], axis=1).values[:5])
        print(f'Optimizer predict: {predictions}')

        print('###############################################################')
        print('Fetching summary...')
        print('###############################################################')
        summary = optimizer.summary()
        print(f'Summary: {summary}')

        print('###############################################################')
        print('Fetching best pipeline details...')
        print('###############################################################')
        best_pipeline_params = optimizer.get_pipeline_details()
        print(f'Best pipeline parameters: {best_pipeline_params}')

        print('###############################################################')
        print('Starting exploring each computed pipeline...')
        print('###############################################################')
        for pipeline_name in summary.index.tolist():
            print('###############################################################')
            print(f'Fetching {pipeline_name} details...')
            print('###############################################################')
            pipeline_params = optimizer.get_pipeline_details(pipeline_name=pipeline_name)
            print(f'{pipeline_name} parameters: {pipeline_params}')

            print('###############################################################')
            print(f'Fetching {pipeline_name} as LALE...')
            print('###############################################################')
            lale_pipeline = optimizer.get_pipeline(pipeline_name=pipeline_name)
            print(f"Fetched pipeline type: {type(lale_pipeline)}")

            print('###############################################################')
            print('Make predictions...')
            print('###############################################################')
            predictions = lale_pipeline.predict(X=self.X.values[:10])
            print(f'Lale {pipeline_name} predicitons: {predictions}')

            print('###############################################################')
            print('Checking lale pretty print...')
            print('###############################################################')
            pipeline_code = lale_pipeline.pretty_print(combinators=False)
            try:
                exec(pipeline_code)
            except Exception as e:
                print('!!!!!!!!!!!!!!!!!!!!!!!EXCEPTION!!!!!!!!!!!!!!!!!!!!!!!!!!')
                traceback.print_exc()


            from lale.lib.sklearn import KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression
            print('###############################################################')
            print('LALE new pipeline preparation...')
            print('###############################################################')
            prefix = lale_pipeline.remove_last().freeze_trainable()
            TestAutoAIRemote.new_pipeline = prefix >> (
                    KNeighborsClassifier | DecisionTreeClassifier | LogisticRegression)

            hyperopt = Hyperopt(estimator=TestAutoAIRemote.new_pipeline, cv=3, max_evals=20)
            try:
                print('###############################################################')
                print('Hyperopt fit...')
                print('###############################################################')
                hyperopt_pipelines = hyperopt.fit(self.X.values, self.y.values)
                new_pipeline_model = hyperopt_pipelines.get_pipeline()
                new_sklearn_pipeline = new_pipeline_model.export_to_sklearn_pipeline()
                predictions = new_sklearn_pipeline.predict(X=self.X.values[:20])
                print(f'sklearn hyperopt pipeline predictions: {predictions}')
            except Exception as e:
                traceback.print_exc()
                hyperopt_results = hyperopt._impl._trials.results
                print(f'Hyperopt results: {hyperopt_results}')

            print('###############################################################')
            print(f'Getting {pipeline_name} as SKLEARN... and store locally')
            print('###############################################################')
            TestAutoAIRemote.sklearn_pipeline = optimizer.get_pipeline(pipeline_name=pipeline_name,
                                                                       astype=self.experiment.PipelineTypes.SKLEARN,
                                                                       persist=True)
            print(f"Fetched pipeline type: {type(TestAutoAIRemote.sklearn_pipeline)}")

            print('###############################################################')
            print('Make predictions...')
            print('###############################################################')
            predictions = TestAutoAIRemote.sklearn_pipeline.predict(X=self.X.values[:10])
            print(f'Sklearn {pipeline_name} predicitons: {predictions}')

    def test_03__explore_experiments(self):

        cases = dict(
            names=[
                'bank_binary_RF',
                'bank_binary_XGB',
                'bank_binary_EX_TREES',
                'bank_binary_LGBM',
                'bank_binary_DT',
                'bank_binary_GB',
                'bank_binary_LR',
                'bank_binary_ALL_1',
                'bank_binary_ALL_2',
                'bank_binary_ALL_3',
                'bank_binary_ALL_4',

            ],
            scorings=[
                AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR,
                AutoAI.Metrics.ROC_AUC_SCORE,
                AutoAI.Metrics.LOG_LOSS,
                AutoAI.Metrics.F1_SCORE_WEIGHTED,
                AutoAI.Metrics.ACCURACY_SCORE,
                AutoAI.Metrics.RECALL_SCORE_MACRO,
                AutoAI.Metrics.R2_SCORE,
                AutoAI.Metrics.MEAN_SQUARED_LOG_ERROR,
                AutoAI.Metrics.EXPLAINED_VARIANCE_SCORE,
                AutoAI.Metrics.RECALL_SCORE,
                AutoAI.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR,
            ],
            test_sizes=[
                0.1,
                0.2,
                0.15,
                0.05,
                0.18,
                0.09,
                0.12,
                0.07,
                0.1,
                0.1,
                0.1,
            ],
            max_estimators=[
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                3,
                4,
            ],
            daub_estimators=[
                [AutoAI.ClassificationAlgorithms.RF],
                [AutoAI.ClassificationAlgorithms.XGB],
                [AutoAI.ClassificationAlgorithms.EX_TREES],
                [AutoAI.ClassificationAlgorithms.LGBM],
                [AutoAI.ClassificationAlgorithms.DT],
                [AutoAI.ClassificationAlgorithms.GB],
                [AutoAI.ClassificationAlgorithms.LR],
                None,
                None,
                None,
                None,
            ]
        )

        for name, scoring, test_size, max_est, daub_est in zip(cases['names'],
                                                               cases['scorings'],
                                                               cases['test_sizes'],
                                                               cases['max_estimators'],
                                                               cases['daub_estimators'],
                                                               ):
            try:
                self.__make_experiment(name, scoring, test_size, max_est, daub_est)

            except Exception as e:
                print(f'@@@@@@@@@@@@@@@@@@@@ EXPERIMENT ERROR @@@@@@@@@@@@@@@@@@@@@@@@@\n'
                      f'NAME: {name}\n'
                      f'{e}\n'
                      f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
                traceback.print_exc()

    #################################
    #      DEPLOYMENT SECTION       #
    #################################

    def test_04_deployment_setup_and_preparation(self):
        if is_cp4d():
            TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
                                                  source_project_id=self.project_id,
                                                  source_space_id=self.space_id)
        else:
            TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials)

        delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)

    def test_05__deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
        self.service.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model=self.pipeline,
            deployment_name=self.DEPLOYMENT_NAME)

    def test_06_score_deployed_model(self):
        nb_records = 10
        predictions = self.service.score(payload=self.holdout_data.drop(['y'], axis=1)[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_07_list_deployments(self):
        self.service.list()
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_08_delete_deployment(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()


if __name__ == '__main__':
    unittest.main()
