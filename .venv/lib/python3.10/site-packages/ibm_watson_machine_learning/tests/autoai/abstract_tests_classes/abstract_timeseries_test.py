#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc
import os
from os import environ

import subprocess
import sys
import pkg_resources
import time

import pandas as pd
from sklearn.pipeline import Pipeline

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.helpers.connections import DataConnection, DeploymentOutputAssetLocation, S3Location
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id,
                                                     save_data_to_cos_bucket)
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup, delete_model_deployment
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, RunStateTypes

from ibm_watson_machine_learning.tests.utils.assertions import get_and_predict_all_pipelines_as_lale, \
    validate_autoai_timeseries_experiment

from ibm_watson_machine_learning.utils.autoai.utils import chose_model_output
from ibm_watson_machine_learning.utils.autoai.errors import NoAvailableNotebookLocation


class AbstractTestTSAsync(abc.ABC):
    """
    The abstract tests which covers:
    - training AutoAI Forecasting model on a dataset
    - downloading all generated pipelines to lale pipeline
    In order to execute test connection definitions must be provided
    in inheriting classes.
    """

    BATCH_DEPLOYMENT = True
    BATCH_DEPLOYMENT_WITH_DF = True  # batch input passed as Pandas.DataFrame
    BATCH_DEPLOYMENT_WITH_DA = True  # batch input passed as DataConnection type data-assets with csv files
    BATCH_DEPLOYMENT_WITH_CDA = True  # batch input passed as DataConnection type data-assets with connection_id(COS)
    BATCH_DEPLOYMENT_WITH_CA = True  # batch input passed as DataConnection type connected assets (COS)

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    pod_version = environ.get('KB_VERSION', None)
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    results_cos_path = 'results_wml_autoai'

    # to be set in every child class:
    OPTIMIZER_NAME = "AutoAI forecasting test"

    SPACE_ONLY = True
    HISTORICAL_RUNS_CHECK = True

    experiment_info = dict(name=OPTIMIZER_NAME,
                           desc='test description',
                           prediction_type=PredictionType.FORECASTING,
                           prediction_columns=['species'],
                           autoai_pod_version=pod_version
                           )

    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None

    batch_output_filename = "TS_Batch_output.csv"

    cos_resource_instance_id = None
    experiment_info: dict = None

    data_assets_to_delete = []

    trained_pipeline_details = None
    run_id = None
    prev_run_id = None
    data_connection = None
    results_connection = None
    train_data = None

    pipeline: 'Pipeline' = None
    winning_pipelines_summary = None
    discarded_pipelines_summary = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None
    scoring_df = None

    project_id = None
    space_id = None

    asset_id = None
    connection_id = None

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

    @staticmethod
    def install(package: str):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    def test_00a_space_cleanup(self):
        space_checked = False
        while not space_checked:
            space_cleanup(self.wml_client,
                          get_space_id(self.wml_client, self.space_name,
                                       cos_resource_instance_id=self.cos_resource_instance_id),
                          days_old=7)
            space_id = get_space_id(self.wml_client, self.space_name,
                                    cos_resource_instance_id=self.cos_resource_instance_id)
            try:
                self.assertIsNotNone(space_id, msg="space_id is None")
                space_checked = True
            except AssertionError:
                space_checked = False

        AbstractTestTSAsync.space_id = space_id

        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):

        if self.SPACE_ONLY:
            AbstractTestTSAsync.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                    space_id=self.space_id)
        else:
            AbstractTestTSAsync.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                    project_id=self.project_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    @abc.abstractmethod
    def test_02_data_reference_setup(self):
        pass

    def test_03_initialize_optimizer(self):
        AbstractTestTSAsync.experiment_info = validate_autoai_timeseries_experiment(self.experiment_info,
                                                                                    self.pod_version)

        AbstractTestTSAsync.remote_auto_pipelines = self.experiment.optimizer(**AbstractTestTSAsync.experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)

        # TODO: params validation
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml(self):
        AbstractTestTSAsync.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=False)

        AbstractTestTSAsync.run_id = self.trained_pipeline_details['metadata']['id']
        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')

    def test_06_get_train_data(self):
        # AbstractTestTSAsync.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()
        #
        # print("train data sample:")
        # print(self.train_data.head())
        # self.assertGreater(len(self.train_data), 0)

        AbstractTestTSAsync.train_X, AbstractTestTSAsync.test_X, AbstractTestTSAsync.train_y, AbstractTestTSAsync.test_y = \
            self.remote_auto_pipelines.get_data_connections()[0].read(with_holdout_split=True)

        AbstractTestTSAsync.scoring_df = self.train_X[:10]

        print("train data sample:")
        print(self.train_X.head())
        self.assertGreater(len(self.train_X), 0)

    def test_07_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        run_details = self.remote_auto_pipelines.get_run_details()
        self.assertEqual(status, "completed",
                         msg="AutoAI run didn't finished successfully. Status: {},\n\n Run details {}".format(status,
                                                                                                              run_details))

    def test_08_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        import json
        print(json.dumps(self.wml_client.training.get_details(training_uid=parameters['metadata']['id']), indent=4))
        print(parameters)
        self.assertIsNotNone(parameters)

    def test_08b_get_metrics(self):
        metrics = self.wml_client.training.get_metrics(self.run_id)
        self.assertIsNotNone(metrics)
        self.assertGreater(len(metrics), 0)

    def test_08c_get_pipeline_notebook(self):
        def test_notebook(*args):
            path = ''

            try:
                path = self.remote_auto_pipelines.get_pipeline_notebook(*args)

                if len(args) == 2:
                    self.assertEqual(path, args[1])

                with open(path, 'r') as f:
                    self.assertGreater(len(f.read()), 0)
            except Exception as e:
                raise e
            finally:
                try:
                    os.remove(path)
                except Exception as e:
                    pass

        test_notebook()

        summary = self.remote_auto_pipelines.summary()

        ## note: test if NoAvailableNotebookLocation is raised
        details = self.wml_client.training.get_details(training_uid=self.remote_auto_pipelines.get_run_details()['metadata']['id'])
        winners = summary._series['Enhancements'].keys()
        no_winner_pipelines = ['Pipeline_' + m['context']['intermediate_model']['name'][1:] for m in details['entity']['status']['metrics'] if 'Pipeline_' + m['context']['intermediate_model']['name'][1:] not in winners]
        if len(no_winner_pipelines)>0:
            with self.assertRaises(NoAvailableNotebookLocation):
                test_notebook(no_winner_pipelines[0])
        # end note

        pipeline_name = summary.index[0]

        test_notebook(pipeline_name)

        test_notebook(pipeline_name, 'test_notebook.ipynb')

    def test_09_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.train_X)
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_10_summary_listing_all_pipelines_from_wml(self):
        pipelines_summary = self.remote_auto_pipelines.summary()
        print(pipelines_summary.to_string())

        AbstractTestTSAsync.winning_pipelines_summary = pipelines_summary[pipelines_summary['Winner']]
        AbstractTestTSAsync.discarded_pipelines_summary = pipelines_summary[pipelines_summary['Winner'] == False]

        self.assertGreater(len(AbstractTestTSAsync.winning_pipelines_summary), 0,
                           msg=f"Summary is empty:\n {pipelines_summary.to_string()}")
        if 'max_number_of_estimators' in self.experiment_info and self.experiment_info:
            nb_gen_pipelines = self.experiment_info['max_number_of_estimators']
            self.assertEqual(len(AbstractTestTSAsync.winning_pipelines_summary), nb_gen_pipelines,
                             msg=f"Incorrect winning pipelines in summary:\n {pipelines_summary.to_string()}")

    def test_11__get_data_connections__return_a_list_with_data_connections_with_optimizer_params(self):
        data_connections = self.remote_auto_pipelines.get_data_connections()
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    def test_12_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(pipeline_params)

    # GET PIPELINE FOR TS IS NOT SUPPORTED
    # ########
    # # LALE #
    # ########
    #
    def test_13_get_pipeline__load_lale_pipeline__pipeline_loaded(self):
        AbstractTestTSAsync.lale_pipeline = self.remote_auto_pipelines.get_pipeline()
        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")
        self.assertIsNotNone(self.lale_pipeline)
    #
    # def test_14_get_all_pipelines_as_lale(self):
    #     get_and_predict_all_pipelines_as_lale(self.remote_auto_pipelines, self.X_values)

    #################################
    #        HISTORICAL RUNS        #
    #################################

    def test_15_list_historical_runs_and_get_run_ids(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")
        runs_df = self.experiment.runs(filter=None).list()
        print(runs_df)
        runs_df = self.experiment.runs(filter=self.OPTIMIZER_NAME).list()
        print(runs_df)
        self.assertIsNotNone(runs_df)
        self.assertGreater(len(runs_df), 0)

        runs_completed_df = runs_df[runs_df.state == 'completed']

        if len(runs_completed_df) > 1:
            AbstractTestTSAsync.prev_run_id = runs_completed_df.run_id.iloc[1]  # prev run_id
            print("Random historical run_id: {}".format(AbstractTestTSAsync.prev_run_id))
            self.assertIsNotNone(AbstractTestTSAsync.prev_run_id)

    def test_16_get_params_of_last_historical_run(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")

        run_params = self.experiment.runs.get_params(run_id=self.run_id)
        self.assertIn('prediction_type', run_params,
                      msg="prediction_type field not fount in run_params. Run_params are: {}".format(run_params))

        AbstractTestTSAsync.historical_opt = self.experiment.runs.get_optimizer(self.run_id)
        self.assertIsInstance(self.historical_opt, RemoteAutoPipelines,
                              msg="historical_optimizer is not type RemoteAutoPipelines. It's type of {}".format(
                                  type(self.historical_opt)))

        train_data = self.historical_opt.get_data_connections()[0].read()

    # # GET PIPELINE FOR TS IS NOT SUPPORTED
    # def test_17_get_last_historical_pipeline_and_predict_on_historical_pipeline(self):
    #     if not self.HISTORICAL_RUNS_CHECK:
    #         self.skipTest("Skipping historical runs check.")
    #
    #     print("Getting pipeline for last run_id={}".format(self.run_id))
    #     summary = self.historical_opt.summary()
    #     pipeline_name = summary.index.values[0]
    #     historical_pipeline = self.historical_opt.get_pipeline(pipeline_name,
    #                                                            astype=self.experiment.PipelineTypes.SKLEARN)
    #     print(type(historical_pipeline))
    #     predictions = historical_pipeline.predict(self.X_values)
    #     print(predictions)
    #     self.assertGreater(len(predictions), 0, msg="Empty predictions")

    ###########################################################
    #      DEPLOYMENT SECTION    tests numbers start from 31  #
    ###########################################################

    def test_31_deployment_online_setup_and_preparation(self):
        # note: if target_space_id is not set, use the space_id
        if self.target_space_id is None:
            AbstractTestTSAsync.target_space_id = self.space_id
        else:
            AbstractTestTSAsync.target_space_id = self.target_space_id
        # end note

        self.wml_client.set.default_space(AbstractTestTSAsync.target_space_id)
        delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)

        if self.SPACE_ONLY:
            AbstractTestTSAsync.service = WebService(source_wml_credentials=self.wml_credentials,
                                                     source_space_id=self.space_id,
                                                     target_wml_credentials=self.wml_credentials,
                                                     target_space_id=AbstractTestTSAsync.target_space_id)
        else:
            AbstractTestTSAsync.service = WebService(source_wml_credentials=self.wml_credentials,
                                                     source_project_id=self.project_id,
                                                     target_wml_credentials=self.wml_credentials,
                                                     target_space_id=AbstractTestTSAsync.target_space_id)

        self.assertIsInstance(AbstractTestTSAsync.service, WebService, msg="Deployment is not of WebService type.")
        self.assertIsInstance(AbstractTestTSAsync.service._source_workspace, WorkSpace,
                              msg="Workspace set incorrectly.")
        self.assertEqual(AbstractTestTSAsync.service.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(AbstractTestTSAsync.service.name, None, msg="Deployment name initialized incorrectly")

    def test_32__deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
        pipeline_name = self.winning_pipelines_summary.reset_index()['Pipeline Name'][0]
        print('Deploying', pipeline_name)
        AbstractTestTSAsync.service.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model=pipeline_name,
            deployment_name=self.DEPLOYMENT_NAME)

        self.assertIsNotNone(AbstractTestTSAsync.service.id, msg="Online Deployment creation - missing id")
        self.assertIsNotNone(AbstractTestTSAsync.service.name, msg="Online Deployment creation - name not set")
        self.assertIsNotNone(AbstractTestTSAsync.service.scoring_url,
                             msg="Online Deployment creation - mscoring url  missing")

    def test_33_score_deployed_model_with_empty_payload(self):
        predictions = AbstractTestTSAsync.service.score()
        print(predictions)
        self.assertIsNotNone(predictions)

        forecast_window = self.experiment_info.get('forecast_window', 1)
        self.assertEqual(len(predictions['predictions'][0]['values']), forecast_window)
        self.assertEqual(len(predictions['predictions'][0]['values'][0][0]),
                         len(self.experiment_info.get('prediction_columns', [])))

    def test_34_score_deployed_model_with_df(self):
        predictions = AbstractTestTSAsync.service.score(payload=self.scoring_df)
        print(predictions)
        self.assertIsNotNone(predictions)

        forecast_window = self.experiment_info.get('forecast_window', 1)
        self.assertEqual(len(predictions['predictions'][0]['values']), forecast_window)
        self.assertEqual(len(predictions['predictions'][0]['values'][0][0]),
                         len(self.experiment_info.get('prediction_columns', [])))

    def test_35_list_deployments(self):
        AbstractTestTSAsync.service.list()
        params = AbstractTestTSAsync.service.get_params()
        print(params)
        self.assertIsNotNone(params)

        self.assertIn(AbstractTestTSAsync.service.deployment_id, str(params))

    def test_36_delete_deployment(self):
        print("Delete current deployment: {}".format(AbstractTestTSAsync.service.deployment_id))
        AbstractTestTSAsync.service.delete()
        self.wml_client.repository.delete(AbstractTestTSAsync.service.asset_id)
        self.assertEqual(AbstractTestTSAsync.service.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(AbstractTestTSAsync.service.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(AbstractTestTSAsync.service.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")

        ########################################################
        #  Batch deployment (possible test numbers are: 40-54) #
        ########################################################

    def test_40_deployment_target_space_setup(self):
        if not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment")

        # note: if target_space_id is not set, use the space_id
        if self.target_space_id is None:
            AbstractTestTSAsync.target_space_id = self.space_id
        else:
            AbstractTestTSAsync.target_space_id = self.target_space_id
        # end note

        self.wml_client.set.default_space(AbstractTestTSAsync.target_space_id)

    def test_41_batch_deployment_setup_and_preparation(self):
        if not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment")

        self.assertIsNotNone(AbstractTestTSAsync.target_space_id, "Test issue: target space not set.")

        if self.SPACE_ONLY:
            AbstractTestTSAsync.service_batch = Batch(source_wml_credentials=self.wml_credentials,
                                                      source_space_id=self.space_id,
                                                      target_wml_credentials=self.wml_credentials,
                                                      target_space_id=AbstractTestTSAsync.target_space_id)
        else:
            AbstractTestTSAsync.service_batch = Batch(source_wml_credentials=self.wml_credentials,
                                                      source_project_id=self.project_id,
                                                      target_wml_credentials=self.wml_credentials,
                                                      target_space_id=AbstractTestTSAsync.target_space_id)

        self.assertIsInstance(AbstractTestTSAsync.service_batch, Batch, msg="Deployment is not of Batch type.")
        self.assertIsInstance(AbstractTestTSAsync.service_batch._source_workspace, WorkSpace,
                              msg="Workspace set incorrectly.")
        self.assertEqual(AbstractTestTSAsync.service_batch.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(AbstractTestTSAsync.service_batch.name, None, msg="Deployment name initialized incorrectly")

    def test_42_deploy__batch_deploy_best_computed_pipeline_from_autoai_on_wml(self):
        if not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment")

        pipeline_name = self.winning_pipelines_summary.reset_index()['Pipeline Name'][0]
        print('Deploying', pipeline_name)
        AbstractTestTSAsync.service_batch.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model=pipeline_name,
            deployment_name=self.DEPLOYMENT_NAME + ' BATCH')

        self.assertIsNotNone(AbstractTestTSAsync.service_batch.id, msg="Batch Deployment creation - missing id")
        self.assertIsNotNone(AbstractTestTSAsync.service_batch.id, msg="Batch Deployment creation - name not set")
        self.assertIsNotNone(AbstractTestTSAsync.service_batch.asset_id,
                             msg="Batch Deployment creation - model (asset) id missing, incorrect model storing")

    def test_43_list_batch_deployments(self):
        if not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment")

        deployments = AbstractTestTSAsync.service_batch.list()
        print(deployments)
        self.assertIn('created_at', str(deployments).lower())
        self.assertIn('status', str(deployments).lower())

        params = AbstractTestTSAsync.service_batch.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_44_run_job__batch_deployed_model_with_data_frame(self):
        if not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment")

        if self.BATCH_DEPLOYMENT_WITH_DF:
            scoring_params = AbstractTestTSAsync.service_batch.run_job(
                payload=self.scoring_df,
                background_mode=False)
            print(scoring_params)
            print(AbstractTestTSAsync.service_batch.get_job_result(scoring_params['metadata']['id']))
            self.assertIsNotNone(scoring_params)
            self.assertIn('predictions', str(scoring_params).lower())
        else:
            self.skipTest("Skip batch deployment run job with data frame")

    def test_45_run_job_batch_deployed_model_with_data_assets(self):
        if not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment")

        data_connections_space_only = []
        # self.wml_client.set.default_space(self.target_space_id)

        test_case_batch_output_filename = "da_" + self.batch_output_filename

        results_reference = DataConnection(
            location=DeploymentOutputAssetLocation(name=test_case_batch_output_filename))

        if not self.BATCH_DEPLOYMENT_WITH_DA:
            self.skipTest("Skip batch deployment run job with data asset type")
        else:
            self.assertIsNotNone(self.batch_payload_location,
                                 "Test configuration failure: Batch payload location is missing")

            asset_details = self.wml_client.data_assets.create(
                name=self.batch_payload_location.split('/')[-1],
                file_path=self.batch_payload_location)
            asset_id = self.wml_client.data_assets.get_id(asset_details)
            AbstractTestTSAsync.data_assets_to_delete.append(asset_id)
            data_connections_space_only = [DataConnection(data_asset_id=asset_id)]

        scoring_params = AbstractTestTSAsync.service_batch.run_job(
            payload=data_connections_space_only,
            output_data_reference=results_reference,
            background_mode=False)

        print(scoring_params)
        self.assertIsNotNone(scoring_params)

        deployment_job_id = self.wml_client.deployments.get_job_uid(scoring_params)

        self.wml_client.data_assets.list()

        data_asset_details = self.wml_client.data_assets.get_details()
        self.assertIn(test_case_batch_output_filename, str(data_asset_details),
                      f"Batch output file: {test_case_batch_output_filename} missed in data assets")

        predictions = AbstractTestTSAsync.service_batch.get_job_result(deployment_job_id)
        print(predictions)

        self.assertIsNotNone(predictions)

    def test_46_run_job_batch_deployed_model_with_data_assets_with_cos_connection(self):
        if not self.BATCH_DEPLOYMENT_WITH_CDA or not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment run job with data asset with cos connection type")

        data_connections_space_only = []
        # self.wml_client.set.default_space(self.target_space_id)

        test_case_batch_output_filename = "cos_da_" + self.batch_output_filename

        results_reference =  DataConnection(
            location=DeploymentOutputAssetLocation(name=test_case_batch_output_filename))

        AbstractTestTSAsync.bucket_name = save_data_to_cos_bucket(self.batch_payload_location,
                                                                  self.batch_payload_cos_location,
                                                                  access_key_id=self.cos_credentials['cos_hmac_keys'][
                                                                      'access_key_id'],
                                                                  secret_access_key=
                                                                  self.cos_credentials['cos_hmac_keys'][
                                                                      'secret_access_key'],
                                                                  cos_endpoint=self.cos_endpoint,
                                                                  bucket_name=self.bucket_name)

        # prepare connection
        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(
                'bluemixcloudobjectstorage'),
            'name': 'Connection to COS for tests',
            'properties': {
                'bucket': self.bucket_name,
                'access_key': self.cos_credentials['cos_hmac_keys']['access_key_id'],
                'secret_key': self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                'iam_url': self.wml_client.service_instance._href_definitions.get_iam_token_url(),
                'url': self.cos_endpoint
            }
        })

        AbstractTestTSAsync.connection_id = self.wml_client.connections.get_uid(connection_details)

        self.assertIsNotNone(self.batch_payload_location,
                             "Test configuration failure: Batch payload location is missing")

        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: AbstractTestTSAsync.connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Batch deployment asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: f"{self.bucket_name}/{self.batch_payload_cos_location}"
        })

        asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(asset_id, str)
        AbstractTestTSAsync.data_assets_to_delete.append(asset_id)

        data_connections_space_only.append(
            DataConnection(data_asset_id=asset_id))

        self.assertEqual(len(data_connections_space_only), 1)

        scoring_params = AbstractTestTSAsync.service_batch.run_job(
            payload=data_connections_space_only,
            output_data_reference=results_reference,
            background_mode=False)

        print(scoring_params)
        self.assertIsNotNone(scoring_params)

        deployment_job_id = self.wml_client.deployments.get_job_uid(scoring_params)

        self.wml_client.data_assets.list()

        data_asset_details = self.wml_client.data_assets.get_details()
        self.assertIn(test_case_batch_output_filename, str(data_asset_details),
                      f"Batch output file: {test_case_batch_output_filename} missed in data assets")

        predictions = AbstractTestTSAsync.service_batch.get_job_result(deployment_job_id)
        print(predictions)

        self.assertIsNotNone(predictions)

    def test_47_run_job_batch_deployed_model_with_connected_data_asset(self):

        if not self.BATCH_DEPLOYMENT_WITH_CA or not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment run job with connected asset with cos connection type")
        data_connections_space_only = []

        self.assertIsNotNone(self.wml_client.default_space_id, "TEST Error: default space was not set correctly")

        test_case_batch_output_filename = "cos_ca_" + self.batch_output_filename

        results_reference = DataConnection(
            location=DeploymentOutputAssetLocation(name=test_case_batch_output_filename))

        # prepare connection
        if not AbstractTestTSAsync.connection_id:
            connection_details = self.wml_client.connections.create({
                'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(
                    'bluemixcloudobjectstorage'),
                'name': 'Connection to COS for tests',
                'properties': {
                    'bucket': self.bucket_name,
                    'access_key': self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    'secret_key': self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                    'iam_url': self.wml_client.service_instance._href_definitions.get_iam_token_url(),
                    'url': self.cos_endpoint
                }
            })

            AbstractTestTSAsync.connection_id = self.wml_client.connections.get_uid(connection_details)

        self.assertIsNotNone(self.batch_payload_location,
                             "Test configuration failure: Batch payload location is missing")

        AbstractTestTSAsync.bucket_name = save_data_to_cos_bucket(self.batch_payload_location,
                                                                  self.batch_payload_cos_location,
                                                                  access_key_id=self.cos_credentials['cos_hmac_keys'][
                                                                      'access_key_id'],
                                                                  secret_access_key=
                                                                  self.cos_credentials['cos_hmac_keys'][
                                                                      'secret_access_key'],
                                                                  cos_endpoint=self.cos_endpoint,
                                                                  bucket_name=self.bucket_name)
        conn_space = DataConnection(
            connection_asset_id=AbstractTestTSAsync.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.batch_payload_cos_location
            )
        )

        data_connections_space_only.append(conn_space)

        scoring_params = AbstractTestTSAsync.service_batch.run_job(
            payload=data_connections_space_only,
            output_data_reference=results_reference,
            background_mode=False)

        print(scoring_params)
        self.assertIsNotNone(scoring_params)

        deployment_job_id = self.wml_client.deployments.get_job_uid(scoring_params)

        self.wml_client.data_assets.list()

        data_asset_details = self.wml_client.data_assets.get_details()
        self.assertIn(test_case_batch_output_filename, str(data_asset_details),
                      f"Batch output file: {test_case_batch_output_filename} missed in data assets")

        predictions = AbstractTestTSAsync.service_batch.get_job_result(deployment_job_id)

        print(predictions)
        self.assertIsNotNone(predictions)

    def test_48_run_job_batch_deployed_model_with_data_connection_container(self):
        if not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment")

        if self.wml_client.ICP or self.wml_client.WSD:
            self.skipTest("Batch Deployment with container data connection is available only for Cloud")
        else:
            self.skipTest("not ready")

    def test_49_delete_deployment_batch(self):
        if not self.BATCH_DEPLOYMENT:
            self.skipTest("Skip batch deployment")

        print("Delete current deployment: {}".format(AbstractTestTSAsync.service_batch.deployment_id))
        AbstractTestTSAsync.service_batch.delete()
        self.wml_client.set.default_space(self.target_space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(AbstractTestTSAsync.service_batch.asset_id)

        print(f"Delete all created assets in Batch deployment tests. {AbstractTestTSAsync.data_assets_to_delete}")
        for asset_id in AbstractTestTSAsync.data_assets_to_delete:
            self.wml_client.data_assets.delete(asset_id)

        self.assertEqual(AbstractTestTSAsync.service_batch.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(AbstractTestTSAsync.service_batch.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(AbstractTestTSAsync.service_batch.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")
