#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc
import traceback
from os import environ
from random import randrange
from pandas import read_csv

from sklearn.pipeline import Pipeline

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id,
                                                     is_cp4d)
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup, delete_model_deployment
from ibm_watson_machine_learning.utils.autoai.enums import TShirtSize, SamplingTypes


class AbstractTestAutoAIRemote(abc.ABC):
    """
    The abstract tests which covers:
    - training AutoAI model on iris dataset
    - downloading training data from data assets
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline
    - deployment deletion
    In order to execute test connection definitions must be provided
    in inheriting classes.
    """
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    service: 'WebService' = None
    service_batch: 'Batch' = None

    data_location = './autoai/data/iris_dataset.csv'
    test_data_location = './autoai/data/iris_dataset_test.csv'
    # custom_separator = ','

    sheet_name = "iris"

    prediction_column = 'species'

    trained_pipeline_details = None
    run_id = None

    data_connection = None
    test_data_connection = None
    results_connection = None

    train_data = None

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    pod_version = environ.get('KB_VERSION', None)
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'iris_dataset.csv'
    test_data_cos_path = 'iris_dataset_test.csv'
    cos_resource_instance_id = None
    excel_sheet = None

    results_cos_path = 'results_wml_autoai'

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None
    X_values = None
    y_values = None

    OPTIMIZER_NAME = 'Iris wml_autoai multiclass test'
    DEPLOYMENT_NAME = "Iris AutoAI Deployment tests"

    project_id = None
    space_id = None

    asset_id = None
    test_asset_id = None
    connection_id = None

    max_connection_nb = 5

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())

        cls.cos_credentials = get_cos_credentials()
        cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
        cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.project_id = cls.wml_credentials.get('project_id')

    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        AbstractTestAutoAIRemote.space_id = get_space_id(self.wml_client, self.space_name,
                                                         cos_resource_instance_id=self.cos_resource_instance_id)
        self.wml_client.set.default_project(self.project_id)

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        AbstractTestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                     project_id=self.project_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02a_read_before_fit(self):
        self.data_connection.set_client(self.wml_client)

        data_df = self.data_connection.read(excel_sheet=self.excel_sheet)
        print(data_df.head())
        self.assertGreater(len(data_df), 0)


    def test_03_initialize_optimizer(self):
        AbstractTestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.MULTICLASS,
            prediction_column=self.prediction_column,
            holdout_size=0.15,
            max_number_of_estimators=1,
            # csv_separator=self.custom_separator,
            t_shirt_size=TShirtSize.S if self.wml_client.ICP else TShirtSize.L,
            autoai_pod_version=self.pod_version,
            notebooks=True,
            n_parallel_data_connections=self.max_connection_nb,
            retrain_on_holdout=True,
            excel_sheet=self.sheet_name
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml(self):
        # Do not pass test data connection if not set up (for example in CPD tests)
        if self.test_data_connection:
            AbstractTestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
                training_data_references=[self.data_connection],
                test_data_references=[self.test_data_connection],
                training_results_reference=self.results_connection,
                background_mode=False)
        else:
            AbstractTestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
                training_data_reference=[self.data_connection],
                training_results_reference=self.results_connection,
                background_mode=False)

        AbstractTestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']
        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')

    def test_06a_get_train_data(self):
        AbstractTestAutoAIRemote.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

        AbstractTestAutoAIRemote.X_values = self.train_data.drop([self.prediction_column], axis=1)[:10].values
        AbstractTestAutoAIRemote.y_values = self.train_data[self.prediction_column][:10]

    def test_06b_get_test_data(self):
        if self.test_data_connection:
            test_data = self.remote_auto_pipelines.get_test_data_connections()[0].read()

            print("test data sample:")
            print(test_data.head())
            self.assertGreater(len(test_data), 0)
            self.assertEqual(len(test_data), len(read_csv(self.test_data_location)))
        else:
            from ibm_watson_machine_learning.utils.autoai.errors import TestDataNotPresent
            with self.assertRaises(TestDataNotPresent):
                _ = self.remote_auto_pipelines.get_test_data_connections()

    def test_06c_get_train_data_with_split(self):
        if self.test_data_connection:
            from ibm_watson_machine_learning.utils.autoai.errors import NoAutomatedHoldoutSplit
            with self.assertRaises(NoAutomatedHoldoutSplit):
                _ = self.remote_auto_pipelines.get_data_connections()[0].read(with_holdout_split=True)
        else:
            X_train, X_holdout, y_train, y_holdout = self.remote_auto_pipelines.get_data_connections()[0].read(with_holdout_split=True)

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

    def test_07_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        self.assertEqual(status, "completed", msg="AutoAI run didn't finished successfully. "
                                                  "Status: {}\n run details: {}".format(status,
                                                                                        self.remote_auto_pipelines.get_run_details()))

    def test_08_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        import json
        print(json.dumps(self.wml_client.training.get_details(training_uid=parameters['metadata']['id']), indent=4))
        print(parameters)
        self.assertIsNotNone(parameters)

    def test_08b_get_metrics(self):
        self.wml_client.training.get_metrics(self.run_id)

    def test_09_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.X_values[:5])
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_10_summary_listing_all_pipelines_from_wml(self):
        pipelines_details = self.remote_auto_pipelines.summary()
        print(pipelines_details.to_string())

    def test_11__get_data_connections__return_a_list_with_data_connections_with_optimizer_params(self):
        data_connections = self.remote_auto_pipelines.get_data_connections()
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    def test_12_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(pipeline_params)

    ####
    # LALE #
    ########

    def test_13_get_pipeline__load_lale_pipeline__pipeline_loaded(self):
        AbstractTestAutoAIRemote.lale_pipeline = self.remote_auto_pipelines.get_pipeline()
        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")
        from lale.operators import TrainablePipeline
        self.assertIsInstance(self.lale_pipeline, TrainablePipeline)

    def test_14_get_all_pipelines_as_lale(self):
        from lale.operators import TrainablePipeline

        summary = self.remote_auto_pipelines.summary()
        print(summary)
        failed_pipelines = []
        error_messages = []
        for pipeline_name in summary.reset_index()['Pipeline Name']:
            print(f"Getting pipeline: {pipeline_name}")
            lale_pipeline = None
            try:
                lale_pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name=pipeline_name)
                self.assertIsInstance(lale_pipeline, TrainablePipeline)
                predictions = lale_pipeline.predict(
                    X=self.X_values[:1])
                print(predictions)
                self.assertGreater(len(predictions), 0, msg=f"Returned prediction for {pipeline_name} are empty")
            except:
                print(f"Failure: {pipeline_name}")
                failed_pipelines.append(pipeline_name)

                error_message = traceback.format_exc()
                print(error_message)
                error_messages.append(error_message)

            if not AbstractTestAutoAIRemote.lale_pipeline:
                AbstractTestAutoAIRemote.lale_pipeline = lale_pipeline
                print(f"{pipeline_name} loaded for next test cases")

        self.assertEqual(len(failed_pipelines), 0, msg=f"Some pipelines failed. Full list: {failed_pipelines} \n "
                                                       f"Errors: {error_messages}")

    #################################
    #        HISTORICAL RUNS        #
    #################################

    def test_15_list_historical_runs_and_get_run_ids(self):
        runs_df = self.experiment.runs(filter=self.OPTIMIZER_NAME).list()
        print(runs_df)
        self.assertIsNotNone(runs_df)
        self.assertGreater(len(runs_df), 0)

        runs_completed_df = runs_df[runs_df.state == 'completed']

        AbstractTestAutoAIRemote.random_run_id = runs_completed_df.run_id.iloc[
            randrange(0, min(1, max(len(runs_completed_df) - 1, 1)))]  # random run_id first 2 rows
        print("Random historical run_id: {}".format(AbstractTestAutoAIRemote.random_run_id))
        self.assertIsNotNone(AbstractTestAutoAIRemote.random_run_id)

    def test_16_get_params_of_last_historical_run(self):
        run_params = self.experiment.runs.get_params(run_id=self.run_id)
        self.assertIn('prediction_type', run_params,
                      msg="prediction_type field not fount in run_params. Run_params are: {}".format(run_params))

        AbstractTestAutoAIRemote.historical_opt = self.experiment.runs.get_optimizer(self.run_id)
        self.assertIsInstance(self.historical_opt, RemoteAutoPipelines,
                              msg="historical_optimizer is not type RemoteAutoPipelines. It's type of {}".format(
                                  type(self.historical_opt)))

        AbstractTestAutoAIRemote.train_data = self.historical_opt.get_data_connections()[0].read()

    def test_17_get_last_historical_pipeline_and_predict_on_historical_pipeline(self):
        print("Getting pipeline for last run_id={}".format(self.run_id))
        summary = self.historical_opt.summary()
        pipeline_name = summary.index.values[0]
        historical_pipeline = self.historical_opt.get_pipeline(pipeline_name,
                                                               astype=self.experiment.PipelineTypes.SKLEARN)
        print(type(historical_pipeline))
        predictions = historical_pipeline.predict(self.X_values[-2:])
        print(predictions)
        self.assertGreater(len(predictions), 0, msg="Empty predictions")

    #################################
    #      DEPLOYMENT SECTION       #
    #################################

    def test_19_deployment_setup_and_preparation(self):
        AbstractTestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
                                                      source_project_id=self.project_id,
                                                      target_wml_credentials=self.wml_credentials,
                                                      target_space_id=self.space_id)
        self.wml_client.set.default_space(self.space_id)
        delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)
        self.wml_client.set.default_project(self.project_id) if self.project_id else None

        self.assertIsInstance(self.service, WebService, msg="Deployment is not of WebService type.")
        self.assertIsInstance(self.service._source_workspace, WorkSpace, msg="Workspace set incorrectly.")
        self.assertEqual(self.service.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(self.service.name, None, msg="Deployment name initialized incorrectly")

    def test_20__deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
        best_pipeline = self.remote_auto_pipelines.summary()._series['Enhancements'].keys()[0]
        print('Deploying', best_pipeline)
        self.service.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model=best_pipeline,
            deployment_name=self.DEPLOYMENT_NAME)

        self.assertIsNotNone(self.service.id, msg="Online Deployment creation - missing id")
        self.assertIsNotNone(self.service.name, msg="Online Deployment creation - name not set")
        self.assertIsNotNone(self.service.scoring_url,
                             msg="Online Deployment creation - mscoring url  missing")

    def test_21_score_deployed_model(self):
        nb_records = 5
        predictions = self.service.score(payload=self.train_data.drop([self.prediction_column], axis=1)[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_22_list_deployments(self):
        self.service.list()
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_23_delete_deployment(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(self.service.asset_id)
        self.wml_client.set.default_project(self.project_id) if is_cp4d() else None
        self.assertEqual(self.service.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(self.service.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(self.service.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")

    #########################
    #  Batch deployment
    #########################

    def test_24_batch_deployment_setup_and_preparation(self):
        AbstractTestAutoAIRemote.service_batch = Batch(source_wml_credentials=self.wml_credentials.copy(),
                                                       source_project_id=self.project_id,
                                                       target_wml_credentials=self.wml_credentials,
                                                       target_space_id=self.space_id)

        self.assertIsInstance(self.service_batch, Batch, msg="Deployment is not of Batch type.")
        self.assertIsInstance(self.service_batch._source_workspace, WorkSpace, msg="Workspace set incorrectly.")
        self.assertEqual(self.service_batch.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(self.service_batch.name, None, msg="Deployment name initialized incorrectly")

    def test_25__deploy__batch_deploy_best_computed_pipeline_from_autoai_on_wml(self):
        self.service_batch.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model="Pipeline_2",
            deployment_name=self.DEPLOYMENT_NAME + ' BATCH')

        self.assertIsNotNone(self.service_batch.id, msg="Batch Deployment creation - missing id")
        self.assertIsNotNone(self.service_batch.id, msg="Batch Deployment creation - name not set")
        self.assertIsNotNone(self.service_batch.asset_id,
                             msg="Batch Deployment creation - model (asset) id missing, incorrect model storing")

    def test_26_score_batch_deployed_model(self):
        scoring_params = self.service_batch.run_job(
            payload=self.train_data.drop(columns=[self.prediction_column]),
            background_mode=False)
        print(scoring_params)
        print(self.service_batch.get_job_result(scoring_params['metadata']['id']))
        self.assertIsNotNone(scoring_params)
        self.assertIn('predictions', str(scoring_params).lower())

    def test_27_list_batch_deployments(self):
        deployments = self.service_batch.list()
        print(deployments)
        self.assertIn('created_at', str(deployments).lower())
        self.assertIn('status', str(deployments).lower())

        params = self.service_batch.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_28_delete_deployment_batch(self):
        print("Delete current deployment: {}".format(self.service_batch.deployment_id))
        self.service_batch.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(self.service_batch.asset_id)
        self.wml_client.set.default_project(self.project_id)
        self.assertEqual(self.service_batch.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(self.service_batch.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(self.service_batch.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")
