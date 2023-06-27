#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from sklearn.pipeline import Pipeline
from pprint import pprint
import pandas as pd
import traceback
from os import environ

from ibm_watson_machine_learning import APIClient

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.workspace import WorkSpace

from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, FSLocation, \
    AssetLocation

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines

from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, bucket_exists,
                                                     create_bucket, get_space_id, is_cp4d)
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup, delete_model_deployment

from ibm_watson_machine_learning.utils.autoai.enums import TShirtSize, PipelineTypes

from lale.operators import TrainablePipeline


class TestAutoAIRemote(unittest.TestCase):
    """
    The test can be run on CLOUD, WMLS and CPD (not tested)
    The test covers:
    - COS set-up (if run on Cloud): checking if bucket exists for the cos instance, if not new bucket is create
    - Saving data `/bank.cdv` to COS/data assets
    - downloading training data from cos/data assets
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline
    - deployment deletion
    """
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    service: 'WebService' = None
    service_batch: 'Batch' = None

    trained_pipeline_details = None
    run_id = None

    data_connection = None
    results_connection = None

    train_data = None

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    pod_version = environ.get('KB_VERSION', None)

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'data/donorchoose_cos.csv'
    cos_resource_instance_id = None

    results_cos_path = 'results_wml_autoai'

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None
    X_values = None
    y_values = None
    elapsed_time = None
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    data_location = './autoai/data/donorchoose_1MB.csv'

    prediction_column = "project_is_approved"

    with_text_transformer = True

    OPTIMIZER_NAME = f"TEXT TRANSFORMER {with_text_transformer} binary {data_location.split('/')[-1]} "
    DEPLOYMENT_NAME = "TEXT TRANSFORMER AutoAI Deployment tests"

    project_id = None
    space_id = None

    train_sample_columns_index_list = [1, 2, 4, 5, 7,  8, 9]
    text_processing = True
    word2vec_feature_number = 15
    drop_duplicates = True

    def _check_optimizer_parameters(self, parameters):
        # check_pipeline_parameters:
        self.assertIsNotNone(parameters.get('name'))

        self.assertEqual(parameters.get('prediction_column'), self.prediction_column)
        self.assertEqual(parameters.get('text_processing'), self.text_processing)
        self.assertEqual(parameters.get('drop_duplicates'), self.drop_duplicates)
        self.assertEqual(parameters.get('word2vec_feature_number'), self.word2vec_feature_number)

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())

        if not cls.wml_client.ICP:
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.project_id = cls.wml_credentials.get('project_id')

    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestAutoAIRemote.space_id = get_space_id(self.wml_client, self.space_name,
                                                 cos_resource_instance_id=self.cos_resource_instance_id)
        if self.wml_client.ICP:
            self.wml_client.set.default_project(self.project_id)
        else:
            self.wml_client.set.default_space(self.space_id)

    def test_00b_prepare_COS_instance(self):
        if self.wml_client.ICP:
            self.skipTest("Prepare COS is available only for Cloud")
        import ibm_boto3
        cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # Prepare bucket
        if not bucket_exists(cos_resource, self.bucket_name):
            TestAutoAIRemote.bucket_name = create_bucket(cos_resource, self.bucket_name)

            self.assertIsNotNone(TestAutoAIRemote.bucket_name)
            self.assertTrue(bucket_exists(cos_resource, TestAutoAIRemote.bucket_name))

        print(f"Using COS bucket: {TestAutoAIRemote.bucket_name}")

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                             project_id=self.project_id)
        # space_id=self.space_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_save_remote_data_and_DataConnection_setup(self):
        if self.wml_client.ICP:
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
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path)
            )
            TestAutoAIRemote.results_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.results_cos_path)
            )
            TestAutoAIRemote.data_connection.write(data=self.data_location, remote_name=self.data_cos_path)

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    def test_03_initialize_optimizer(self):
        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.BINARY,
            prediction_column=self.prediction_column,
            scoring=self.experiment.Metrics.LOG_LOSS,
            train_sample_columns_index_list=self.train_sample_columns_index_list,
            text_processing=self.text_processing,
            word2vec_feature_number=self.word2vec_feature_number,
            autoai_pod_version=self.pod_version
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        self._check_optimizer_parameters(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml(self):
        import time
        start_time = time.time()
        TestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=False)

        TestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']
        TestAutoAIRemote.elapsed_time = time.time() - start_time
        print(self.elapsed_time)

        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')
        # TestAutoAIRemote.run_id = '95327361-d68a-4f3b-ba62-1a46d6d34b54'
        # TestAutoAIRemote.remote_auto_pipelines = self.experiment.runs.get_optimizer(self.run_id)

    def test_06_get_train_data(self):
        TestAutoAIRemote.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

        TestAutoAIRemote.X_values = self.train_data.drop([self.prediction_column], axis=1)[:10].values
        TestAutoAIRemote.y_values = self.train_data[self.prediction_column][:10]

    def test_07_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        self.assertEqual(status, "completed", msg="AutoAI run didn't finished successfully. Status: {}".format(status))

    def test_08_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
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
        print(pipelines_details)

    def test_11__get_data_connections__return_a_list_with_data_connections_with_optimizer_params(self):
        data_connections = self.remote_auto_pipelines.get_data_connections()
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    def test_12a_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_1')
        print(pipeline_params)

    def test_12a_get_pipeline_params__fetch_best_pipeline_parameters__parameters_fetched_as_dict(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(best_pipeline_params)

    def test_13_check_params_of_realoaded_pipeline_optimizer(self):
        historical_optimizer = self.experiment.runs.get_optimizer(self.run_id)
        hist_opt_params = historical_optimizer.get_params()
        print(hist_opt_params)
        self._check_optimizer_parameters(hist_opt_params)

    ####
    # LALE #
    ########

    def test_14a_get_pipeline__load_sklearn_pipeline__pipeline_loaded(self):
        TestAutoAIRemote.pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name='Pipeline_4',
                                                                            astype=PipelineTypes.SKLEARN)
        print(f"Fetched pipeline type: {type(self.pipeline)}")

        self.assertIsInstance(self.pipeline, Pipeline,
                              msg="Fetched pipeline is not of Sklearn Pipeline instance.")

        print(self.pipeline.steps)
        print("\nPredictions:")
        predictions = self.pipeline.predict(
            X=self.X_values[:5])
        print(predictions)

    def test_14b_get_pipeline__load_lale_pipeline__pipeline_loaded(self):
        TestAutoAIRemote.pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name='Pipeline_4')
        print(f"Fetched pipeline type: {type(self.pipeline)}")

        self.assertIsInstance(self.pipeline, TrainablePipeline,
                              msg="Fetched pipeline is not of TrainablePipeline instance.")
        predictions = self.pipeline.predict(
            X=self.X_values[:5])
        print(predictions)

    def test_15_get_all_pipelines_as_lale(self):
        summary = self.remote_auto_pipelines.summary()
        print(summary)
        failed_pipelines = []
        for pipeline_name in summary.reset_index()['Pipeline Name']:
            print(f"Getting pipeline: {pipeline_name}")
            lale_pipeline = None
            try:
                lale_pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name=pipeline_name)
                self.assertIsInstance(lale_pipeline, TrainablePipeline)
                print(lale_pipeline)
                print(self.X_values[:5])
                predictions = lale_pipeline.predict(
                    X=self.X_values[:2])
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

    # @unittest.skip("Skipped lale pretty print")
    def test_16__pretty_print_lale__checks_if_generated_python_pipeline_code_is_correct(self):
        pipeline_code = self.lale_pipeline.pretty_print()
        try:
            exec(pipeline_code)

        except Exception as exception:
            self.assertIsNone(exception,
                              msg=f"Pretty print from lale pipeline was not successful \n\n Full pipeline code:\n {pipeline_code}")

    #################################
    #      DEPLOYMENT SECTION       #
    #################################

    def test_21_deployment_setup_and_preparation(self):
        TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
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

    def test_22__deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
        self.service.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model="Pipeline_1",
            deployment_name=self.DEPLOYMENT_NAME)

        self.assertIsNotNone(self.service.id, msg="Online Deployment creation - missing id")
        self.assertIsNotNone(self.service.name, msg="Online Deployment creation - name not set")
        self.assertIsNotNone(self.service.scoring_url,
                             msg="Online Deployment creation - mscoring url  missing")

    def test_23_score_deployed_model(self):
        nb_records = 5
        predictions = self.service.score(payload=self.train_data.drop([self.prediction_column], axis=1)[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_24_list_deployments(self):
        self.service.list()
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_25_delete_deployment(self):
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
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model="Pipeline_2",
            deployment_name=self.DEPLOYMENT_NAME + ' BATCH')

        self.assertIsNotNone(self.service_batch.id, msg="Batch Deployment creation - missing id")
        self.assertIsNotNone(self.service_batch.id, msg="Batch Deployment creation - name not set")
        self.assertIsNotNone(self.service_batch.asset_id,
                             msg="Batch Deployment creation - model (asset) id missing, incorrect model storing")

    def test_33_score_batch_deployed_model(self):
        scoring_params = self.service_batch.run_job(
            payload=self.train_data.drop(columns=[self.prediction_column]),
            background_mode=False)
        print(scoring_params)
        print(self.service_batch.get_job_result(scoring_params['metadata']['id']))
        self.assertIsNotNone(scoring_params)
        self.assertIn('predictions', str(scoring_params).lower())

    def test_34_list_batch_deployments(self):
        deployments = self.service_batch.list()
        print(deployments)
        self.assertIn('created_at', str(deployments).lower())
        self.assertIn('status', str(deployments).lower())

        params = self.service_batch.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_35_delete_deployment_batch(self):
        print("Delete current deployment: {}".format(self.service_batch.deployment_id))
        self.service_batch.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(self.service_batch.asset_id)
        self.wml_client.set.default_project(self.project_id) if is_cp4d() else None
        self.assertEqual(self.service_batch.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(self.service_batch.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(self.service_batch.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")


if __name__ == '__main__':
    unittest.main()
