#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from os import environ
import ibm_watson_machine_learning._wrappers.requests as requests
import json

import ibm_boto3

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.helpers.connections import DataConnection, ContainerLocation
from ibm_watson_machine_learning.tests.utils import is_cp4d, save_data_to_container, get_wml_credentials, \
    get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestWebservice, \
    AbstractTestAutoAIAsync, AbstractTestBatch
from ibm_watson_machine_learning.tests.utils.assertions import validate_autoai_experiment
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms, \
    ImputationStrategy
from ibm_watson_machine_learning.utils.autoai.errors import StrategyIsNotApplicable, InvalidImputationParameterNonTS, \
    InvalidImputationParameterTS, NumericalImputationStrategyValueMisused, ImputationListNotSupported, \
    InconsistentImputationListElements


class TestAutoAIRemote(unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = './autoai/data/Corona_NLP_test_utf8.csv'

    data_cos_path = 'data/Corona_NLP_test_utf8.csv'

    SPACE_ONLY = False

    OPTIMIZER_NAME = "Corona NLP Text Transformer test sdk"

    batch_payload_location = './autoai/data/scoring_payload/Corona_NLP_scoring_payload.csv'
    batch_payload_cos_location = "scoring_payload/Corona_NLP_scoring_payload.csv"

    BATCH_DEPLOYMENT_WITH_CA = True
    BATCH_DEPLOYMENT_WITH_CDA = False
    BATCH_DEPLOYMENT_WITH_DA = False
    BATCH_DEPLOYMENT_WITH_DF = True

    target_space_id = None
    pod_version = environ.get('KB_VERSION', None)

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.MULTICLASS,
        prediction_column='Sentiment',
        scoring=Metrics.F1_SCORE_MACRO,
        holdout_size=0.1,
        text_processing=True,
        text_columns_names=['OriginalTweet', 'Location'],
        word2vec_feature_number=4,
        max_number_of_estimators=1,
        daub_give_priority_to_runtime=3.0,
    )

    ts_experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc='test description',
        prediction_type=PredictionType.FORECASTING,
        prediction_columns=['value1', 'value2'],
        timestamp_column_name='timestamp',
        backtest_num=4,
        lookback_window=5,
        forecast_window=2,
        holdout_size=0.05,
        max_number_of_estimators=1,
        include_only_estimators=[AutoAI.ForecastingAlgorithms.ENSEMBLER],
        #csv_separator=','
    )

    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')
    space_id = None
    experiment = None

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

        TestAutoAIRemote.space_id = space_id

        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        if self.SPACE_ONLY:
            TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                        space_id=self.space_id)
        else:
            TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                        project_id=self.project_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_initialize_optimizer_kb_scenario(self):
        experiment_info = self.experiment_info.copy()
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0]\
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertNotIn('use_imputation', params)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_03_initialize_optimizer_kb_scenario_mean(self):
        experiment_info = self.experiment_info.copy()
        experiment_info['categorical_imputation_strategy'] = ImputationStrategy.MEAN
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertIn('preprocessor_cat_imp_strategy', params)
        self.assertEqual(params['preprocessor_cat_imp_strategy'], 'mean')
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertNotIn('use_imputation', params)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_04_initialize_optimizer_kb_scenario_median(self):
        experiment_info = self.experiment_info.copy()
        experiment_info['categorical_imputation_strategy'] = ImputationStrategy.MEDIAN
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertIn('preprocessor_cat_imp_strategy', params)
        self.assertEqual(params['preprocessor_cat_imp_strategy'], 'median')
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertNotIn('use_imputation', params)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_05_initialize_optimizer_kb_scenario_most_frequent(self):
        experiment_info = self.experiment_info.copy()
        experiment_info['categorical_imputation_strategy'] = ImputationStrategy.MOST_FREQUENT
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertIn('preprocessor_cat_imp_strategy', params)
        self.assertEqual(params['preprocessor_cat_imp_strategy'], 'most_frequent')
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertNotIn('use_imputation', params)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_06_initialize_optimizer_kb_scenario_invalid_strategies(self):
        for strategy in [ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS, ImputationStrategy.VALUE,
                         ImputationStrategy.FLATTEN_ITERATIVE, ImputationStrategy.LINEAR, ImputationStrategy.CUBIC,
                         ImputationStrategy.PREVIOUS, ImputationStrategy.NEXT, ImputationStrategy.NO_IMPUTATION]:
            with self.assertRaises(StrategyIsNotApplicable):
                experiment_info = self.experiment_info.copy()
                experiment_info['categorical_imputation_strategy'] = strategy
                experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

    def test_07_initialize_optimizer_kb_scenario_mean_num(self):
        experiment_info = self.experiment_info.copy()
        experiment_info['numerical_imputation_strategy'] = ImputationStrategy.MEAN
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertIn('preprocessor_num_imp_strategy', params)
        self.assertEqual(params['preprocessor_num_imp_strategy'], 'mean')
        self.assertNotIn('use_imputation', params)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_08_initialize_optimizer_kb_scenario_median_num(self):
        experiment_info = self.experiment_info.copy()
        experiment_info['numerical_imputation_strategy'] = ImputationStrategy.MEDIAN
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertIn('preprocessor_num_imp_strategy', params)
        self.assertEqual(params['preprocessor_num_imp_strategy'], 'median')
        self.assertNotIn('use_imputation', params)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_09_initialize_optimizer_kb_scenario_most_frequent_num(self):
        experiment_info = self.experiment_info.copy()
        experiment_info['numerical_imputation_strategy'] = ImputationStrategy.MOST_FREQUENT
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertIn('preprocessor_num_imp_strategy', params)
        self.assertEqual(params['preprocessor_num_imp_strategy'], 'most_frequent')
        self.assertNotIn('use_imputation', params)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_10_initialize_optimizer_kb_scenario_invalid_strategies_num(self):
        for strategy in [ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS, ImputationStrategy.VALUE,
                         ImputationStrategy.FLATTEN_ITERATIVE, ImputationStrategy.LINEAR, ImputationStrategy.CUBIC,
                         ImputationStrategy.PREVIOUS, ImputationStrategy.NEXT, ImputationStrategy.NO_IMPUTATION]:
            with self.assertRaises(StrategyIsNotApplicable):
                experiment_info = self.experiment_info.copy()
                experiment_info['numerical_imputation_strategy'] = strategy
                experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

    def test_11_initialize_optimizer_kb_scenario_invalid_fields_set(self):
        with self.assertRaises(InvalidImputationParameterNonTS):
            experiment_info = self.experiment_info.copy()
            experiment_info['numerical_imputation_strategy'] = ImputationStrategy.MEAN
            experiment_info['numerical_imputation_value'] = 0
            experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

            TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        with self.assertRaises(InvalidImputationParameterNonTS):
            experiment_info = self.experiment_info.copy()
            experiment_info['numerical_imputation_strategy'] = ImputationStrategy.MEAN
            experiment_info['imputation_threshold'] = 0.1
            experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

            TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

    def test_12_initialize_optimizer_ts_scenario(self):
        ts_experiment_info = self.ts_experiment_info.copy()

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0]\
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['FlattenIterative', 'Linear', 'Cubic', 'Previous'])
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_13_initialize_optimizer_ts_scenario_mean(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.MEAN

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['Fill'])
        self.assertIn('imputer_fill_type', params)
        self.assertEqual(params['imputer_fill_type'], 'mean')
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_14_initialize_optimizer_ts_scenario_median(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.MEDIAN

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['Fill'])
        self.assertIn('imputer_fill_type', params)
        self.assertEqual(params['imputer_fill_type'], 'median')
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_15_initialize_optimizer_ts_scenario_best_of_default_imputers(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['FlattenIterative', 'Linear', 'Cubic', 'Previous'])
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_16_initialize_optimizer_ts_scenario_value(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.VALUE

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['Fill'])
        self.assertIn('imputer_fill_type', params)
        self.assertEqual(params['imputer_fill_type'], 'value')
        self.assertIn('imputer_fill_value', params)
        self.assertEqual(params['imputer_fill_value'], 0)
        self.assertNotIn('imputation_threshold', params)

    def test_17_initialize_optimizer_ts_scenario_flatten_iterative(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.FLATTEN_ITERATIVE

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['FlattenIterative'])
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_18_initialize_optimizer_ts_scenario_linear(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.LINEAR

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['Linear'])
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_19_initialize_optimizer_ts_scenario_cubic(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.CUBIC

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['Cubic'])
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_20_initialize_optimizer_ts_scenario_previous(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.PREVIOUS

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['Previous'])
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_21_initialize_optimizer_ts_scenario_next(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.NEXT

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['Next'])
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_22_initialize_optimizer_ts_scenario_no_imputation(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.NO_IMPUTATION

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], False)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_23_initialize_optimizer_ts_scenario_invalid_strategies(self):
        for strategy in [ImputationStrategy.MOST_FREQUENT]:
            with self.assertRaises(StrategyIsNotApplicable):
                ts_experiment_info = self.ts_experiment_info.copy()
                ts_experiment_info['numerical_imputation_strategy'] = strategy

                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

    def test_24_initialize_optimizer_ts_scenario_invalid_fields_set(self):
        with self.assertRaises(NumericalImputationStrategyValueMisused):
            ts_experiment_info = self.ts_experiment_info.copy()
            ts_experiment_info['numerical_imputation_strategy'] = ImputationStrategy.MEAN
            ts_experiment_info['numerical_imputation_value'] = 0

            TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        with self.assertRaises(InvalidImputationParameterTS):
            ts_experiment_info = self.ts_experiment_info.copy()
            ts_experiment_info['categorical_imputation_strategy'] = ImputationStrategy.MEAN

            TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

    def test_25_initialize_optimizer_kb_scenario_string_mean(self):
        experiment_info = self.experiment_info.copy()
        experiment_info['categorical_imputation_strategy'] = 'mean'
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertIn('preprocessor_cat_imp_strategy', params)
        self.assertEqual(params['preprocessor_cat_imp_strategy'], 'mean')
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertNotIn('use_imputation', params)
        self.assertNotIn('imputer_list', params)
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_26_initialize_optimizer_ts_scenario_string_mean(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = "mean"

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertEqual(params['imputer_list'], ['Fill'])
        self.assertIn('imputer_fill_type', params)
        self.assertEqual(params['imputer_fill_type'], 'mean')
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_27_initialize_optimizer_ts_scenario_list_default(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ['FlattenIterative', 'Linear', 'Cubic', 'Previous']

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertListEqual(params['imputer_list'], ['FlattenIterative', 'Linear', 'Cubic', 'Previous'])
        self.assertNotIn('imputer_fill_type', params)
        self.assertNotIn('imputer_fill_value', params)
        self.assertNotIn('imputation_threshold', params)

    def test_28_initialize_optimizer_ts_scenario_list_inconsistent(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = [ImputationStrategy.PREVIOUS, ImputationStrategy.NO_IMPUTATION]

        with self.assertRaises(InconsistentImputationListElements):
            TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = [ImputationStrategy.MEAN, ImputationStrategy.VALUE]

        with self.assertRaises(InconsistentImputationListElements):
            TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

    def test_29_initialize_optimizer_ts_scenario_list_advanced(self):
        ts_experiment_info = self.ts_experiment_info.copy()
        ts_experiment_info['numerical_imputation_strategy'] = ['FlattenIterative', 'Linear', 'Cubic', 'Fill', 'value']
        ts_experiment_info['numerical_imputation_value'] = 5

        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

        params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
            ['nodes'][0]['parameters']['optimization']

        print(json.dumps(params, indent=4))

        self.assertNotIn('preprocessor_cat_imp_strategy', params)
        self.assertNotIn('preprocessor_num_imp_strategy', params)
        self.assertIn('use_imputation', params)
        self.assertEqual(params['use_imputation'], True)
        self.assertIn('imputer_list', params)
        self.assertListEqual(params['imputer_list'], ['FlattenIterative', 'Linear', 'Cubic', 'Fill'])
        self.assertIn('imputer_fill_type', params)
        self.assertEqual(params['imputer_fill_type'], 'value')
        self.assertIn('imputer_fill_value', params)
        self.assertEqual(params['imputer_fill_value'], 5)
        self.assertNotIn('imputation_threshold', params)

    def test_30_initialize_optimizer_kb_scenario_list_invalid(self):
        experiment_info = self.experiment_info.copy()
        experiment_info['categorical_imputation_strategy'] = [ImputationStrategy.MEAN, ImputationStrategy.MOST_FREQUENT]
        experiment_info = validate_autoai_experiment(experiment_info, self.pod_version)

        with self.assertRaises(ImputationListNotSupported):
            TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

    def test_31_initialize_optimizer_kb_scenario_string_params_all(self):
        for imputation_strategy in ImputationStrategy:
            experiment_info = self.experiment_info.copy()
            experiment_info['numerical_imputation_strategy'] = imputation_strategy

            try:
                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

                params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
                    ['nodes'][0]['parameters']['optimization']

                print('Original:', json.dumps(params, indent=4))

                experiment_info = self.experiment_info.copy()
                experiment_info['numerical_imputation_strategy'] = params['preprocessor_num_imp_strategy']

                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

                params2 = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
                    ['nodes'][0]['parameters']['optimization']

                print('Reconstructed:', json.dumps(params2, indent=4))

                self.assertNotIn('preprocessor_cat_imp_strategy', params2)
                self.assertIn('preprocessor_num_imp_strategy', params2)
                self.assertNotIn('use_imputation', params2)
                self.assertEqual(params, params2)
            except StrategyIsNotApplicable:
                print(f'Strategy {imputation_strategy} not applicable to numerical kb imputation.')

        for imputation_strategy in ImputationStrategy:
            experiment_info = self.experiment_info.copy()
            experiment_info['categorical_imputation_strategy'] = imputation_strategy

            try:
                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

                params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
                    ['nodes'][0]['parameters']['optimization']

                print('Original:', json.dumps(params, indent=4))

                experiment_info = self.experiment_info.copy()
                experiment_info['categorical_imputation_strategy'] = params['preprocessor_cat_imp_strategy']

                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**experiment_info)

                params2 = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
                    ['nodes'][0]['parameters']['optimization']

                print('Reconstructed:', json.dumps(params2, indent=4))

                self.assertIn('preprocessor_cat_imp_strategy', params2)
                self.assertNotIn('preprocessor_num_imp_strategy', params2)
                self.assertNotIn('use_imputation', params2)
                self.assertEqual(params, params2)
            except StrategyIsNotApplicable:
                print(f'Strategy {imputation_strategy} not applicable to categorical kb imputation.')

    def test_32_initialize_optimizer_ts_scenario_string_params_all(self):
        for imputation_strategy in ImputationStrategy:
            ts_experiment_info = self.ts_experiment_info.copy()
            ts_experiment_info['numerical_imputation_strategy'] = imputation_strategy

            try:
                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

                params = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
                    ['nodes'][0]['parameters']['optimization']

                print('Original:', json.dumps(params, indent=4))

                ts_experiment_info = self.ts_experiment_info.copy()
                ts_experiment_info['numerical_imputation_strategy'] = (params['imputer_list'] if params['use_imputation'] else []) + ([params['imputer_fill_type']] if 'imputer_fill_type' in params else [])

                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(**ts_experiment_info)

                params2 = self.remote_auto_pipelines._engine._wml_stored_pipeline_details['entity']['document']['pipelines'][0] \
                    ['nodes'][0]['parameters']['optimization']

                print('Reconstructed:', json.dumps(params2, indent=4))

                self.assertNotIn('preprocessor_cat_imp_strategy', params2)
                self.assertNotIn('preprocessor_num_imp_strategy', params2)
                self.assertEqual(params, params2)
            except StrategyIsNotApplicable:
                print(f'Strategy {imputation_strategy} not applicable to ts imputation.')

if __name__ == '__main__':
    unittest.main()
