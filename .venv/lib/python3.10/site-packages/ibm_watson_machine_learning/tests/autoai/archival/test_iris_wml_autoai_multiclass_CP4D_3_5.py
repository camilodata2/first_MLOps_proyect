#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import pandas as pd
from sklearn.pipeline import Pipeline
from lale.operators import TrainablePipeline
from lale.lib.lale import Hyperopt
import traceback
from os import environ

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, FSLocation, \
    CP4DAssetLocation
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.utils.autoai.enums import TShirtSize
from ibm_watson_machine_learning.utils.autoai.errors import AutoAIComputeError

from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d, bucket_exists, \
    bucket_name_gen, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import delete_model_deployment, bucket_cleanup, space_cleanup


@unittest.skipIf(not (is_cp4d()), "Supported only on CP4D")
class TestAutoAIRemote(unittest.TestCase):
    """
    The test covers:
    - COS set-up (if run on Cloud): checking if bucket exists for the cos instance, if not new bucket is create
    - Saving data `datasets_different_delimeter/iris_dataset_different_delimeter.csv` to COS/data assets
    - multiclass experiment for datasets with different delimeter dataset and `;` separator
    - downloading all generated pipelines to lale pipeline
    - list historical runs
    - get one historical run and download its pipeline
    - predict on historical run
    - deployment by `pipeline name`
    - deployment deletion
    """
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    service: 'WebService' = None
    service_batch: 'Batch' = None
    wml_credentials = None

    train_data = None
    data = None

    pipeline_opt: 'RemoteAutoPipelines' = None
    historical_opt: 'RemoteAutoPipelines' = None

    trained_pipeline_details = None
    run_id = None
    random_run_id = None

    best_pipeline = None
    lale_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None

    data_connection = None
    results_connection = None

    data_location = './autoai/data/iris_dataset.csv'
    custom_separator = ','

    OPTIMIZER_NAME = 'Iris wml_autoai multiclass test'
    DEPLOYMENT_NAME = "Iris AutoAI Deployment tests"

    project_id = "4ae1998c-87be-49df-9baf-c79ce3edbd4b"
    space_id = 'c6f02cdc-ca64-4285-b248-b5ad788be456'

    asset_id = '458d167a-cecd-4d41-8204-338cf732de7c'

    @classmethod
    def setUp(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """

        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())
        cls.wml_client.set.default_project(cls.project_id)

    # def test_00_prepare_COS_instance(self):
    #     if self.wml_client.ICP or self.wml_client.WSD:
    #         self.skipTest("Prepare COS is available only for Cloud")
    #     import ibm_boto3
    #     cos_resource = ibm_boto3.resource(
    #         service_name="s3",
    #         endpoint_url=self.cos_endpoint,
    #         aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
    #         aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
    #     )
    #     # Prepare bucket
    #     if not bucket_exists(cos_resource, self.bucket_name):
    #         try:
    #             bucket_cleanup(cos_resource, prefix=f"{self.bucket_name}-")
    #         except Exception as e:
    #             print(f"Bucket cleanup with prefix {self.bucket_name}- failed due to:\n{e}\n skipped")
    #
    #         import datetime
    #         TestAutoAIRemote.bucket_name = bucket_name_gen(prefix=f"{self.bucket_name}-{str(datetime.date.today())}")
    #         print(f"Creating COS bucket: {TestAutoAIRemote.bucket_name}")
    #         cos_resource.Bucket(TestAutoAIRemote.bucket_name).create()
    #
    #         self.assertIsNotNone(TestAutoAIRemote.bucket_name)
    #         self.assertTrue(bucket_exists(cos_resource, TestAutoAIRemote.bucket_name))
    #
    #     print(f"Using COS bucket: {TestAutoAIRemote.bucket_name}")

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                             project_id=self.project_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_save_remote_data_and_DataConNection_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(location=CP4DAssetLocation(asset_id=self.asset_id))
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    def test_02a_read_saved_remote_data_before_fit(self):
        TestAutoAIRemote.data = self.data_connection.read(csv_separator=self.custom_separator)
        print("Data sample:")
        print(self.data.head())
        self.assertGreater(len(self.data), 0)

    #################################
    #       REMOTE OPTIMIZER        #
    #################################

    def test_03_initialize_optimizer(self):
        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.MULTICLASS,
            prediction_column='species',
            scoring=self.experiment.Metrics.ACCURACY_SCORE,
            test_size=0.15,
            max_number_of_estimators=1,
            csv_separator=self.custom_separator,
            t_shirt_size=TShirtSize.S
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml(self):
        failure_count = 0
        while failure_count < 4:
            try:
                TestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
                    training_data_reference=[self.data_connection],
                    training_results_reference=self.results_connection,
                    background_mode=False)
                break
            except AutoAIComputeError as e:
                import time
                failure_count += 1
                print(f"Failure {failure_count}\n error:{e}\n\n")
                time.sleep(10)

        TestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']
        self.assertIsInstance(self.trained_pipeline_details, dict,
                              msg='Trained pipeline details are not a dictionary instance.')
        self.assertTrue(bool(self.trained_pipeline_details))  # Checking if details are not empty.

        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')

    def test_06_get_train_data(self):
        TestAutoAIRemote.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()
        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

    def test_07_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        self.assertEqual(status, "completed", msg="AutoAI run didn't finished successfully. Status: {}".format(status))

    def test_08_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        print(parameters)

    def test_09_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.train_data.drop(['species'], axis=1).values[:5])
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_10_summary_listing_all_pipelines_from_wml(self):
        pipelines_details = self.remote_auto_pipelines.summary()
        print(pipelines_details)

    def test_11_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_1')
        self.assertIsNotNone(pipeline_params)
        self.assertIn('composition_steps', pipeline_params)
        print(pipeline_params['composition_steps'])
        self.assertIn('pipeline_nodes', pipeline_params)
        print(pipeline_params['pipeline_nodes'])
        self.assertIn('features_importance', pipeline_params)
        print(pipeline_params['features_importance'])
        self.assertIn('roc_curve', pipeline_params)
        print(pipeline_params['roc_curve'])
        self.assertIn('confusion_matrix', pipeline_params)
        print(pipeline_params['confusion_matrix'])
        self.assertIn('ml_metrics', pipeline_params)
        print(pipeline_params['ml_metrics'])

    def test_12_get_pipeline_params_best_pipeline_parameters(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        self.assertIsNotNone(best_pipeline_params)
        print(best_pipeline_params)

    def test_13_get_pipeline_load_specified_pipeline(self):
        TestAutoAIRemote.lale_pipeline = self.remote_auto_pipelines.get_pipeline(
            pipeline_name='Pipeline_1')
        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")
        self.assertIsInstance(self.lale_pipeline, TrainablePipeline)

    def test_14_get_all_pipelines_as_lale(self):
        summary = self.remote_auto_pipelines.summary()
        print(summary)
        failed_pipelines = []
        for pipeline_name in summary.reset_index()['Pipeline Name']:
            print(f"Getting pipeline: {pipeline_name}")
            lale_pipeline = None
            try:
                lale_pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name=pipeline_name)
                self.assertIsInstance(lale_pipeline, TrainablePipeline)
                predictions = lale_pipeline.predict(
                    X=self.train_data.values[:1])
                print(predictions)
                self.assertGreater(len(predictions), 0, msg=f"Returned prediction for {pipeline_name} are empty")
            except:
                print(f"Failure: {pipeline_name}")
                failed_pipelines.append(pipeline_name)
                traceback.print_exc()

            if not TestAutoAIRemote.lale_pipeline:
                TestAutoAIRemote.lale_pipeline = lale_pipeline
                print(f"{pipeline_name} loaded for next test cases")

        self.assertEqual(len(failed_pipelines), 0, msg=f"Some pipelines failed. Full list: {failed_pipelines}")

    #################################
    #        HISTORICAL RUNS        #
    #################################

    def test_15_list_historical_runs_and_get_run_ids(self):
        runs_df = self.experiment.runs.list()
        print(runs_df)
        self.assertIsNotNone(runs_df)
        self.assertGreater(len(runs_df), 0)

        runs_completed_df = runs_df[runs_df.state == 'completed']

        from random import randint
        TestAutoAIRemote.random_run_id = runs_completed_df.run_id.iloc[
            randint(len(runs_completed_df) // 2,
                    len(runs_completed_df) - 1)]  # random run_id from the 2nd part of runs table
        print("Random run_id from 2nd half of df: {}".format(TestAutoAIRemote.random_run_id))
        self.assertIsNotNone(TestAutoAIRemote.random_run_id)

    def test_16_get_params_of_last_historical_run(self):
        run_params = self.experiment.runs.get_params(run_id=self.run_id)
        self.assertIn('prediction_type', run_params,
                      msg="prediction_type field not fount in run_params. Run_params are: {}".format(run_params))

        TestAutoAIRemote.historical_opt = self.experiment.runs.get_optimizer(self.run_id)
        self.assertIsInstance(self.historical_opt, RemoteAutoPipelines,
                              msg="historical_optimizer is not type RemoteAutoPipelines. It's type of {}".format(
                                  type(self.historical_opt)))

        TestAutoAIRemote.train_data = self.historical_opt.get_data_connections()[0].read()

    def test_17_get_last_historical_pipeline_and_predict_on_historical_pipeline(self):
        print("Getting pipeline for last run_id={}".format(self.run_id))
        summary = self.historical_opt.summary()
        pipeline_name = summary.index.values[0]
        historical_pipeline = self.historical_opt.get_pipeline(pipeline_name,
                                                               astype=self.experiment.PipelineTypes.SKLEARN)
        print(type(historical_pipeline))
        predictions = historical_pipeline.predict(self.train_data.values[-2:])
        print(predictions)
        self.assertGreater(len(predictions), 0, msg="Empty predictions")

    def test_18_get_random_historical_optimizer_and_its_pipeline(self):
        run_params = self.experiment.runs.get_params(run_id=self.random_run_id)
        self.assertIn('prediction_type', run_params,
                      msg="prediction_type field not fount in run_params. Run_params are: {}".format(run_params))
        historical_opt = self.experiment.runs.get_optimizer(self.random_run_id)
        self.assertIsInstance(historical_opt, RemoteAutoPipelines,
                              msg="historical_optimizer is not type RemoteAutoPipelines. It's type of {}".format(
                                  type(historical_opt)))
        summary = self.historical_opt.summary()

        self.assertGreater(len(summary), 0, msg=f"No pipelines found for optimizer with run_id = {self.random_run_id},"
                                                f" and parameters: {run_params}")
        pipeline_name = summary.index.values[0]
        pipeline = historical_opt.get_pipeline(pipeline_name, self.experiment.PipelineTypes.SKLEARN)
        print(type(pipeline))
        self.assertIsInstance(pipeline, Pipeline)

    #################################
    #      DEPLOYMENT SECTION       #
    #################################

    def test_20_deployment_setup_and_preparation(self):
        # TestAutoAIRemote.run_id = '149b23da-aef1-4568-a20c-f426368b6063'
        if self.wml_client.ICP:
            TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
                                                  source_project_id=self.project_id,
                                                  source_space_id=self.space_id)
            self.wml_client.set.default_space(self.space_id)
        else:
            TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials,
                                                  source_space_id=self.space_id)

        delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)

        self.wml_client.set.default_project(self.project_id) if self.wml_client.ICP else None

    def test_21_deploy_pipepline_by_pipeline_name(self):
        TestAutoAIRemote.service.create(
            model='Pipeline_4',
            deployment_name=self.DEPLOYMENT_NAME,
            experiment_run_id=self.run_id
        )
        self.assertIsNotNone(self.service.id, msg="Online Deployment creation - missing id")
        self.assertIsNotNone(self.service.name, msg="Online Deployment creation - name not set")
        self.assertIsNotNone(self.service.scoring_url,
                             msg="Online Deployment creation - mscoring url  missing")

    def test_22_score_deployed_model(self):
        nb_records = 5
        predictions = self.service.score(payload=self.train_data.drop(['species'], axis=1)[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_23_list_deployments(self):
        self.service.list()
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_24_delete_deployment(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()

    def test_25_deployment_setup_and_preparation_different_wml(self):
        if self.wml_client.ICP:
            TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
                                                  source_project_id=self.project_id,
                                                  source_space_id=self.space_id,
                                                  target_wml_credentials=self.wml_credentials.copy(),
                                                  target_project_id=self.project_id,
                                                  target_space_id=self.space_id
                                                  )
            self.wml_client.set.default_space(self.space_id)
        else:
            TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials,
                                                  source_space_id=self.space_id,
                                                  target_wml_credentials=self.wml_credentials,
                                                  target_space_id=self.space_id,
                                                  )

        delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)

        self.wml_client.set.default_project(self.project_id) if self.wml_client.ICP else None

    def test_26_deploy_pipepline_by_pipeline_name_different_wml(self):
        TestAutoAIRemote.service.create(
            model='Pipeline_4',
            deployment_name=self.DEPLOYMENT_NAME,
            experiment_run_id=self.run_id
        )

    def test_27_score_deployed_model_different_wml(self):
        nb_records = 5
        predictions = self.service.score(payload=self.train_data.drop(['species'], axis=1)[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_28_list_deployments_different_wml(self):
        self.service.list()
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_29_delete_deployment_different_wml(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()

    #########################
    #  Batch deployment
    #########################

    def test_30_batch_deployment_setup_and_preparation(self):
        TestAutoAIRemote.service_batch = Batch(source_wml_credentials=self.wml_credentials.copy(),
                                               source_project_id=self.project_id,
                                               target_wml_credentials=self.wml_credentials,
                                               target_space_id=self.space_id)

        self.assertIsInstance(self.service_batch, Batch, msg="Deployment is not of Batch type.")
        self.assertIsInstance(self.service_batch._source_workspace, WorkSpace, msg="Workspace set incorrectly.")
        self.assertEqual(self.service_batch.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(self.service_batch.name, None, msg="Deployment name initialized incorrectly")

    def test_32__deploy__batch_deploy_best_computed_pipeline_from_autoai_on_wml(self):
        self.service_batch.create(
            experiment_run_id=self.run_id,
            model=self.lale_pipeline,
            deployment_name=self.DEPLOYMENT_NAME + 'BATCH')

        self.assertIsNotNone(self.service_batch.id, msg="Batch Deployment creation - missing id")
        self.assertIsNotNone(self.service_batch.name, msg="Batch Deployment creation - name not set")
        self.assertIsNotNone(self.service_batch.asset_id,
                             msg="Batch Deployment creation - model (asset) id missing, incorrect model storing")

    def test_33_score_batch_deployed_model(self):
        scoring_params = self.service_batch.run_job(
            payload=self.train_data.drop(columns=['species']),
            background_mode=False)
        print(scoring_params)

    def test_34_list_batch_deployments(self):
        deployments = self.service_batch.list()
        print(deployments)
        params = self.service_batch.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_35_delete_deployment_batch(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service_batch.delete()
        self.assertEqual(self.service_batch.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(self.service_batch.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(self.service_batch.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")
