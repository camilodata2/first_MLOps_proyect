#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc
import os
from os import environ

import time

from sklearn.pipeline import Pipeline

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id)
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, RunStateTypes, BatchedRegressionAlgorithms, \
    BatchedClassificationAlgorithms, RegressionAlgorithms, ClassificationAlgorithms

from ibm_watson_machine_learning.tests.utils.assertions import get_and_predict_all_pipelines_as_lale, \
    validate_autoai_experiment

from ibm_watson_machine_learning.utils.autoai.utils import chose_model_output
from ibm_watson_machine_learning.wml_client_error import WMLClientError


class AbstractTestAutoAIAsync(abc.ABC):
    """
    The abstract tests which covers:
    - training AutoAI model on a dataset
    - downloading all generated pipelines to lale pipeline
    In order to execute test connection definitions must be provided
    in inheriting classes.
    """

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    pod_version = environ.get('KB_VERSION', None)
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    results_cos_path = 'results_wml_autoai'

    # to be set in every child class:
    OPTIMIZER_NAME = "AutoAI regression test"

    SPACE_ONLY = True
    HISTORICAL_RUNS_CHECK = True
    INCREMENTAL_PIPELINES_EXPECTED = False

    experiment_info = dict(name=OPTIMIZER_NAME,
                           desc='test description',
                           prediction_type=PredictionType.MULTICLASS,
                           prediction_column='species',
                           autoai_pod_version=pod_version
                           )

    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    service: 'WebService' = None
    service_batch: 'Batch' = None

    cos_resource_instance_id = None
    experiment_info: dict = None

    trained_pipeline_details = None
    run_id = None
    prev_run_id = None
    data_connection = None
    results_connection = None
    train_data = None

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None
    X_df = None
    X_values = None
    y_values = None

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

        AbstractTestAutoAIAsync.space_id = space_id

        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):

        if self.SPACE_ONLY:
            AbstractTestAutoAIAsync.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                        space_id=self.space_id)
        else:
            AbstractTestAutoAIAsync.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                        project_id=self.project_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

        AbstractTestAutoAIAsync.batched = ('include_batched_ensemble_estimators' not in self.experiment_info and self.wml_client.ICP) or \
                                          self.experiment_info.get('include_batched_ensemble_estimators', False)

    @abc.abstractmethod
    def test_02_data_reference_setup(self):
        pass

    def test_03_initialize_optimizer(self):
        AbstractTestAutoAIAsync.experiment_info = validate_autoai_experiment(self.experiment_info, self.pod_version)

        AbstractTestAutoAIAsync.remote_auto_pipelines = self.experiment.optimizer(**AbstractTestAutoAIAsync.experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)

        # TODO: params validation
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml(self):
        AbstractTestAutoAIAsync.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=False)

        AbstractTestAutoAIAsync.run_id = self.trained_pipeline_details['metadata']['id']
        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')

    def test_06_get_train_data(self):
        AbstractTestAutoAIAsync.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

        AbstractTestAutoAIAsync.X_df = self.train_data.drop([self.experiment_info['prediction_column']], axis=1)[
                                           :10]
        AbstractTestAutoAIAsync.X_values = AbstractTestAutoAIAsync.X_df.values
        AbstractTestAutoAIAsync.y_values = self.train_data[self.experiment_info['prediction_column']][:10]

    def test_07_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        run_details = self.remote_auto_pipelines.get_run_details()
        self.assertEqual(status, "completed",
                         msg="AutoAI run didn't finished successfully. Status: {},\n\n Run details {}".format(status,
                                                                                                              run_details))

    def test_08_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        import json
        training_details = self.wml_client.training.get_details(training_uid=parameters['metadata']['id'])
        AbstractTestAutoAIAsync.batched = 'Batched' in str(training_details)
        print(json.dumps(training_details, indent=4))
        print(parameters)
        self.assertIsNotNone(parameters)

    def test_08b_get_metrics(self):
        metrics = self.wml_client.training.get_metrics(self.run_id)
        self.assertIsNotNone(metrics)
        self.assertGreater(len(metrics), 0)

    def test_08c_get_pipeline_notebook(self):
        def test_notebook(*args, additional_check=None):
            path = ''

            try:
                path = self.remote_auto_pipelines.get_pipeline_notebook(*args)

                if len(args) == 2:
                    self.assertEqual(path, args[1])

                with open(path, 'r') as f:
                    content = f.read()
                    self.assertGreater(len(content), 0)

                    if additional_check:
                        additional_check(content)
            except Exception as e:
                raise e
            finally:
                try:
                    os.remove(path)
                except Exception as e:
                    pass

        test_notebook()

        test_notebook('Pipeline_1')

        test_notebook('Pipeline_1', 'test_notebook.ipynb')

        if self.batched:
            summary = self.remote_auto_pipelines._engine.summary()

            ensemble_pipeline_name = summary['Enhancements'].keys()[
                [i for i, v in enumerate(summary['Enhancements'].values) if 'Ensemble' in v][0]]

            def check_if_ensemble_content(content):
                self.assertIn('Notebook for training continuation', content)
                self.assertIn('pipeline_model.partial_fit', content)

            test_notebook(ensemble_pipeline_name, 'test_notebook_ensemble.ipynb', additional_check=check_if_ensemble_content)

    def test_09_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.X_values)
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_10_summary_listing_all_pipelines_from_wml(self):
        pipelines_summary = self.remote_auto_pipelines.summary()
        print(pipelines_summary.to_string())

        if self.batched:
            if 'include_only_estimators' in self.experiment_info:
                available_estimators = self.experiment_info['include_only_estimators']
            elif self.experiment_info['prediction_type'] == PredictionType.REGRESSION:
                available_estimators = list(RegressionAlgorithms)
            else:
                available_estimators = list(ClassificationAlgorithms)

            if 'include_batched_ensemble_estimators' in self.experiment_info:
                available_batch_estimators = self.experiment_info['include_batched_ensemble_estimators']
            elif self.experiment_info['prediction_type'] == PredictionType.REGRESSION:
                available_batch_estimators = list(BatchedRegressionAlgorithms)
            else:
                available_batch_estimators = list(BatchedClassificationAlgorithms)

            max_batched = len(
                [e for e in available_estimators if e.name in [e1.name for e1 in available_batch_estimators]])
        else:
            max_batched = 0

        self.assertGreater(len(pipelines_summary), 0, msg=f"Summary is empty:\n {pipelines_summary.to_string()}")
        if self.experiment_info:
            max_number_of_estimators = self.experiment_info.get('max_number_of_estimators')
        else:
            max_number_of_estimators = 2

        expected_batched_pipelines = min(max_batched, max_number_of_estimators)

        if self.INCREMENTAL_PIPELINES_EXPECTED:
            nb_gen_pipelines = 4 * max_number_of_estimators + 2 * expected_batched_pipelines
        else:
            nb_gen_pipelines = 4 * max_number_of_estimators + expected_batched_pipelines

        self.assertEqual(len(pipelines_summary), nb_gen_pipelines,
                         msg=f"Incorrect pipelines in summary:\n {pipelines_summary.to_string()}")

    def test_10b_summary_sorting_all_pipelines_by_specific_score(self):

        sort_by_holdout = {
            "training": False,
            "holdout": True
        }

        binary_classification_scorers = [
            "accuracy",
            "average_precision",
            "f1",
            "precision",
            "recall",
            "roc_auc"
        ]

        multi_classification_scorers = [
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "precision_micro",
            "precision_macro",
            "precision_weighted",
            "recall_micro",
            "recall_macro",
            "recall_weighted"
        ]

        regression_scorers = [
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
            "neg_root_mean_squared_error",
            "neg_root_mean_squared_log_error"
        ]

        scorers = {
            PredictionType.BINARY: binary_classification_scorers,
            PredictionType.MULTICLASS: multi_classification_scorers,
            PredictionType.REGRESSION: regression_scorers
        }

        prediction_type = self.remote_auto_pipelines.params["prediction_type"]

        if prediction_type in scorers:
            for score_type in sort_by_holdout.keys():
                for scoring in scorers[prediction_type]:
                    pipelines_summary = self.remote_auto_pipelines.summary(scoring=scoring,
                                                                           sort_by_holdout_score=sort_by_holdout[score_type])
                    print(pipelines_summary.to_string())
                    is_regression = prediction_type == PredictionType.REGRESSION

                    if scoring == self.remote_auto_pipelines.params['scoring'] and score_type == 'training':
                        scoring += '_(optimized)'
                    if is_regression:
                        metric_values = pipelines_summary[f"{score_type}_{scoring[4:]}"]  # take score name without 'neg_'
                    else:
                        metric_values = pipelines_summary[f"{score_type}_{scoring}"]
                    # else:
                    #     if is_regression:
                    #         metric_values = pipelines_summary[f"{score_type}_{scoring[4:]}_(optimized)"]
                    #     else:
                    #         metric_values = pipelines_summary[f"{score_type}_{scoring}_(optimized)"]
                    metric_values = metric_values.fillna(1.7976931348623157e+308)
                    print(metric_values)

                    self.assertEqual((metric_values.values == metric_values.sort_values(ascending=is_regression,
                                                                                        ignore_index=True).values).all(),
                                     True, msg=f"Incorrect Pipelines sorting by {score_type}_{scoring}\n")

    def test_11__get_data_connections__return_a_list_with_data_connections_with_optimizer_params(self):
        data_connections = self.remote_auto_pipelines.get_data_connections()
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    def test_12_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(pipeline_params)

    ########
    # LALE #
    ########

    def test_13_get_pipeline__load_lale_pipeline__pipeline_loaded(self):
        AbstractTestAutoAIAsync.lale_pipeline = self.remote_auto_pipelines.get_pipeline()
        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")
        from lale.operators import TrainablePipeline
        self.assertIsInstance(self.lale_pipeline, TrainablePipeline)

    def test_14_get_all_pipelines_as_lale(self):
        get_and_predict_all_pipelines_as_lale(self.remote_auto_pipelines, self.X_values)

    #################################
    #        HISTORICAL RUNS        #
    #################################

    def test_15_list_historical_runs_and_get_run_ids(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")
        runs_df = self.experiment.runs(filter=self.OPTIMIZER_NAME).list()
        print(runs_df)
        self.assertIsNotNone(runs_df)
        self.assertGreater(len(runs_df), 0)

        runs_completed_df = runs_df[runs_df.state == 'completed']

        if len(runs_completed_df) > 1:
            AbstractTestAutoAIAsync.prev_run_id = runs_completed_df.run_id.iloc[1]  # prev run_id
            print("Random historical run_id: {}".format(AbstractTestAutoAIAsync.prev_run_id))
            self.assertIsNotNone(AbstractTestAutoAIAsync.prev_run_id)

    def test_16_get_params_of_last_historical_run(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")

        run_params = self.experiment.runs.get_params(run_id=self.run_id)
        self.assertIn('prediction_type', run_params,
                      msg="prediction_type field not fount in run_params. Run_params are: {}".format(run_params))

        AbstractTestAutoAIAsync.historical_opt = self.experiment.runs.get_optimizer(self.run_id)
        self.assertIsInstance(self.historical_opt, RemoteAutoPipelines,
                              msg="historical_optimizer is not type RemoteAutoPipelines. It's type of {}".format(
                                  type(self.historical_opt)))

        train_data = self.historical_opt.get_data_connections()[0].read()

    def test_17_get_last_historical_pipeline_and_predict_on_historical_pipeline(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")

        print("Getting pipeline for last run_id={}".format(self.run_id))
        summary = self.historical_opt.summary()
        pipeline_name = summary.index.values[0]
        historical_pipeline = self.historical_opt.get_pipeline(pipeline_name,
                                                               astype=self.experiment.PipelineTypes.SKLEARN)
        print(type(historical_pipeline))
        predictions = historical_pipeline.predict(self.X_values)
        print(predictions)
        self.assertGreater(len(predictions), 0, msg="Empty predictions")

    def test_93_delete_experiment(self):
        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

        run_details = self.wml_client.training.get_details(self.run_id)
        pipeline_id = run_details['entity']['pipeline']['id']
        self.wml_client.pipelines.delete(pipeline_id)
        self.wml_client.training.cancel(self.run_id, hard_delete=True)

        with self.assertRaises(WMLClientError):
            self.wml_client.training.get_details(self.run_id)
            self.wml_client.pipelines.get_details(pipeline_id)


class AbstractTestAutoAISync(abc.ABC):
    """
    The abstract tests which covers:
    - training AutoAI model on a dataset
    - downloading training data 
    - downloading all generated pipelines to lale pipeline

    In order to execute test connection definitions must be provided
    in inheriting classes.
    """

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    pod_version = environ.get('KB_VERSION', None)
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    results_cos_path = 'results_wml_autoai'

    # to be set in every child class:
    OPTIMIZER_NAME = "AutoAI regression test"

    SPACE_ONLY = True
    HISTORICAL_RUNS_CHECK = True

    experiment_info = dict(name=OPTIMIZER_NAME,
                           desc='test description',
                           prediction_type=PredictionType.MULTICLASS,
                           prediction_column='species',
                           autoai_pod_version=pod_version
                           )

    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    service: 'WebService' = None
    service_batch: 'Batch' = None

    cos_resource_instance_id = None
    experiment_info: dict = None

    trained_pipeline_details = None
    run_id = None
    prev_run_id = None
    data_connection = None
    results_connection = None
    train_data = None

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None
    X_values = None
    y_values = None

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

        AbstractTestAutoAISync.space_id = space_id

        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        if self.SPACE_ONLY:
            AbstractTestAutoAISync.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                       space_id=self.space_id)
        else:
            AbstractTestAutoAISync.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                       project_id=self.project_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    @abc.abstractmethod
    def test_02_data_reference_setup(self):
        pass

    def test_03_initialize_optimizer(self):
        AbstractTestAutoAISync.experiment_info = validate_autoai_experiment(self.experiment_info, self.pod_version)

        AbstractTestAutoAISync.remote_auto_pipelines = self.experiment.optimizer(**AbstractTestAutoAISync.experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)

        # TODO: params validation
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml(self):
        AbstractTestAutoAISync.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=True)

        AbstractTestAutoAISync.run_id = self.trained_pipeline_details['metadata']['id']
        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')

    def test_06a_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        self.assertNotEqual(status, RunStateTypes.COMPLETED)

    def test_06b_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        # print(parameters)
        self.assertIsNotNone(parameters)
        self.assertIn(self.run_id, str(parameters))

    def test_07_get_summary(self):

        print(f"Run status = {self.remote_auto_pipelines.get_run_status()}")
        # note: check if first pipeline was generated

        metrics = self.wml_client.training.get_details(self.run_id)['entity']['status'].get('metrics', [])
        while chose_model_output("1") not in str(metrics) and self.remote_auto_pipelines.get_run_status() not in [
            'failed', 'canceled']:
            time.sleep(5)
            print(".", end=" ")
            metrics = self.wml_client.training.get_details(self.run_id)['entity']['status'].get('metrics', [])
        # end note

        status = self.remote_auto_pipelines.get_run_status()
        run_details = self.remote_auto_pipelines.get_run_details().get('entity')
        self.assertNotIn(status, ['failed', 'canceled'], msg=f"Training finished with status {status}. \n"
                                                             f"Details: {run_details.get('status')}")
        print("\n 1st pipeline completed")
        summary_df = self.remote_auto_pipelines.summary()
        print(summary_df)

        self.assertGreater(len(summary_df), 0,
                           msg=f"Summary DataFrame is empty. While {len(metrics)} pipelines are in training_details['entity']['status']['metrics']")

        # check if pipelines are not duplicated
        self.assertEqual(len(summary_df.index.unique()), len(summary_df),
                         msg="Some pipeline names are duplicated in the summary")

    def test_08_get_train_data(self):
        AbstractTestAutoAISync.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

        AbstractTestAutoAISync.X_df = self.train_data.drop([self.experiment_info['prediction_column']], axis=1)[
                                          :10]
        AbstractTestAutoAISync.X_values = AbstractTestAutoAISync.X_df.values
        AbstractTestAutoAISync.y_values = self.train_data[self.experiment_info['prediction_column']][:10]

    def test_09_get_best_pipeline_so_far(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(best_pipeline_params)
        self.assertIn(f"holdout_{self.experiment_info.get('scoring')}", str(best_pipeline_params))

        summary_df = self.remote_auto_pipelines.summary()
        print(summary_df)

        AbstractTestAutoAISync.best_pipeline_name_so_far = summary_df.index[0]
        print("\nGetting best calculated pipeline: ", self.best_pipeline_name_so_far)

        pipeline = self.remote_auto_pipelines.get_pipeline()
        print(f"Fetched pipeline type: {type(pipeline)}")

    def test_09b_get_metrics(self):
        self.wml_client.training.get_metrics(self.run_id)

    #################################
    #      DEPLOYMENT SECTION       #
    #################################

    # def test_10_deployment_setup_and_preparation(self):
    #     AbstractTestAutoAISync.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
    #                                                 source_project_id=self.project_id,
    #                                                 target_wml_credentials=self.wml_credentials,
    #                                                 target_space_id=self.space_id)
    #
    #     self.wml_client.set.default_space(self.space_id)
    #     delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)
    #     self.wml_client.set.default_project(self.project_id) if self.project_id else None
    #
    # def test_11__deploy__online_deploy_pipeline_from_autoai_on_wml(self):
    #     self.service.create(
    #         experiment_run_id=self.run_id,
    #         model=self.best_pipeline_name_so_far,
    #         deployment_name=self.DEPLOYMENT_NAME + self.best_pipeline_name_so_far)
    #
    #     self.assertIsNotNone(self.service.id, msg="Online Deployment creation - missing id")
    #     self.assertIsNotNone(self.service.name, msg="Online Deployment creation - name not set")
    #     self.assertIsNotNone(self.service.scoring_url,
    #                          msg="Online Deployment creation - mscoring url  missing")
    #
    # def test_12_score_deployed_model(self):
    #     nb_records = 5
    #     predictions = self.service.score(payload=self.train_data.drop(['y'], axis=1)[:nb_records])
    #     print(predictions)
    #     self.assertIsNotNone(predictions)
    #     self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)
    #
    # def test_13_list_deployments(self):
    #     self.service.list()
    #     params = self.service.get_params()
    #     print(params)
    #     self.assertIsNotNone(params)

    ###########################################
    #     TRAINING SECTION - FINISH RUN       #
    ###########################################

    def test_55_waiting_for_fitted_completed(self):
        while self.remote_auto_pipelines.get_run_status() == 'running':
            time.sleep(10)

        status = self.remote_auto_pipelines.get_run_status()

        self.assertEqual(status, RunStateTypes.COMPLETED,
                         msg="AutoAI run didn't finished successfully. Status: {}".format(status))

    def test_56_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.X_values[:5])
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_57_summary_listing_all_pipelines_from_wml(self):
        pipelines_details = self.remote_auto_pipelines.summary()
        print(pipelines_details)
        if 'max_number_of_estimators' in self.experiment_info:
            nb_generated_pipelines = 4 * self.experiment_info.get('max_number_of_estimators')
        else:
            nb_generated_pipelines = 8

        self.assertGreaterEqual(len(pipelines_details), nb_generated_pipelines)

    def test_58__get_data_connections__return_a_list_with_data_connections_with_optimizer_params(self):
        data_connections = self.remote_auto_pipelines.get_data_connections()
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    def test_59_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_1')
        print(pipeline_params)

    def test_60__get_pipeline_params__fetch_best_pipeline_parameters__parameters_fetched_as_dict(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(best_pipeline_params)

    ####
    # LALE #
    ########

    def test_61__get_pipeline__load_lale_pipeline__pipeline_loaded(self):
        AbstractTestAutoAISync.lale_pipeline = self.remote_auto_pipelines.get_pipeline()
        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")

        from lale.operators import TrainablePipeline
        self.assertIsInstance(self.lale_pipeline, TrainablePipeline,
                              msg="Fetched pipeline is not of TrainablePipeline instance.")
        predictions = self.lale_pipeline.predict(
            X=self.X_values[:5])
        print(predictions)

    def test_62_get_all_pipelines_as_lale(self):
        get_and_predict_all_pipelines_as_lale(self.remote_auto_pipelines, self.X_values)

    #################################
    #        HISTORICAL RUNS        #
    #################################

    def test_65_list_historical_runs_and_get_run_ids(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")
        runs_df = self.experiment.runs(filter=self.OPTIMIZER_NAME).list()
        print(runs_df)
        self.assertIsNotNone(runs_df)
        self.assertGreater(len(runs_df), 0)

        runs_completed_df = runs_df[runs_df.state == 'completed']

        if len(runs_completed_df) > 1:
            AbstractTestAutoAISync.prev_run_id = runs_completed_df.run_id.iloc[1]  # prev run_id
            print("Random historical run_id: {}".format(AbstractTestAutoAISync.prev_run_id))
            self.assertIsNotNone(AbstractTestAutoAISync.prev_run_id)

    def test_66_get_params_of_last_historical_run(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")

        run_params = self.experiment.runs.get_params(run_id=self.run_id)
        self.assertIn('prediction_type', run_params,
                      msg="prediction_type field not fount in run_params. Run_params are: {}".format(run_params))

        AbstractTestAutoAISync.historical_opt = self.experiment.runs.get_optimizer(self.run_id)
        self.assertIsInstance(self.historical_opt, RemoteAutoPipelines,
                              msg="historical_optimizer is not type RemoteAutoPipelines. It's type of {}".format(
                                  type(self.historical_opt)))

        rain_data = self.historical_opt.get_data_connections()[0].read()

    def test_67_get_last_historical_pipeline_and_predict_on_historical_pipeline(self):
        if not self.HISTORICAL_RUNS_CHECK:
            self.skipTest("Skipping historical runs check.")

        print("Getting pipeline for last run_id={}".format(self.run_id))
        summary = self.historical_opt.summary()
        pipeline_name = summary.index.values[0]
        historical_pipeline = self.historical_opt.get_pipeline(pipeline_name,
                                                               astype=self.experiment.PipelineTypes.SKLEARN)
        print(type(historical_pipeline))
        predictions = historical_pipeline.predict(self.X_values)
        print(predictions)
        self.assertGreater(len(predictions), 0, msg="Empty predictions")
    #
    # def test_68_get_random_historical_optimizer_and_its_pipeline(self):
    #     if not self.HISTORICAL_RUNS_CHECK:
    #         self.skipTest("Skipping historical runs check.")
    #
    #     run_params = self.experiment.runs(filter=self.OPTIMIZER_NAME).get_params(run_id=self.prev_run_id)
    #     self.assertIn('prediction_type', run_params,
    #                   msg="prediction_type field not fount in run_params. Run_params are: {}".format(run_params))
    #     historical_opt = self.experiment.runs.get_optimizer(self.prev_run_id)
    #     self.assertIsInstance(historical_opt, RemoteAutoPipelines,
    #                           msg="historical_optimizer is not type RemoteAutoPipelines. It's type of {}".format(
    #                               type(historical_opt)))
    #
    #     summary = historical_opt.summary()
    #     self.assertGreater(len(summary), 0, msg=f"No pipelines found for optimizer with run_id = {self.prev_run_id},"
    #                                             f" and parameters: {run_params}")
    #
    #     pipeline_name = summary.index.values[0]
    #     pipeline = historical_opt.get_pipeline(pipeline_name, self.experiment.PipelineTypes.SKLEARN)
    #     self.assertIsInstance(pipeline, Pipeline)

    def test_93_delete_experiment(self):
        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

        run_details = self.wml_client.training.get_details(self.run_id)
        pipeline_id = run_details['entity']['pipeline']['id']
        self.wml_client.pipelines.delete(pipeline_id)
        self.wml_client.training.cancel(self.run_id, hard_delete=True)

        with self.assertRaises(WMLClientError):
            self.wml_client.training.get_details(self.run_id)
            self.wml_client.pipelines.get_details(pipeline_id)