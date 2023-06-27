#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import TYPE_CHECKING, List, Dict, Tuple

from pandas import DataFrame, MultiIndex
import json
import os
from time import sleep
import warnings
import logging

from lomond import WebSocket
from ibm_watson_machine_learning.utils.autoai.enums import (
    PredictionType, RegressionAlgorithms, ClassificationAlgorithms, RunStateTypes, DataConnectionTypes,
    ImputationStrategy)
from ibm_watson_machine_learning.utils.autoai.errors import PipelineNotLoaded, FitNeeded, InvalidPredictionType, \
    AutoAIComputeError, ForecastingUnsupportedOperation, NoAvailableMetrics, LibraryNotCompatible, \
    CannotConnectToWebsocket, NumericalImputationStrategyValueMisused, InvalidImputationParameterNonTS, \
    InvalidImputationParameterTS
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.utils.autoai.progress_bar import ProgressBar
from ibm_watson_machine_learning.utils.autoai.utils import (
    fetch_pipelines, is_ipython, ProgressGenerator, check_dependencies_versions, try_import_autoai_libs,
    create_summary, prepare_auto_ai_model_to_publish_normal_scenario, get_sw_spec_and_type_based_on_sklearn,
    get_values_for_imputation_strategy, try_import_autoai_ts_libs, _download_notebook)
from .base_engine import BaseEngine
from ibm_watson_machine_learning.helpers.connections import DataConnection

if TYPE_CHECKING:
    from ibm_watson_machine_learning.workspace import WorkSpace
    from ibm_watson_machine_learning.helpers.connections import DataConnection
    from sklearn.pipeline import Pipeline

__all__ = [
    "WMLEngine"
]

logging.getLogger('lomond').setLevel(logging.CRITICAL)


class WMLEngine(BaseEngine):
    """WML Engine provides unified API to work with AutoAI Pipelines trainings on WML.

    :param workspace: WorkSpace object with wml client and all needed parameters
    :type workspace: WorkSpace
    """

    def __init__(self, workspace: 'WorkSpace') -> None:
        self.workspace = workspace
        self._wml_client = workspace.wml_client
        self._auto_pipelines_parameters = None
        self._wml_pipeline_metadata = None
        self._wml_training_metadata = None
        self._wml_stored_pipeline_details = None
        self._current_run_id = None
        self._20_class_limit_removal_test = False

    def _get_node_id(self):
        result = 'automl'
        if self._auto_pipelines_parameters.get('prediction_type') == PredictionType.FORECASTING:
            result = 'autoai-ts'
        elif self._auto_pipelines_parameters.get('prediction_type') == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
            result = 'autoai-tsad'
        return result

    def _get_learning_type(self):
        result = self._auto_pipelines_parameters.get('prediction_type')
        if self._auto_pipelines_parameters.get('prediction_type') == PredictionType.FORECASTING:
            result = 'timeseries'
        elif self._auto_pipelines_parameters.get('prediction_type') == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
            result = 'timeseries_anomaly_prediction'
        return result

    def _get_runtime_name(self):
        result = 'auto_ai.kb'
        if self._auto_pipelines_parameters.get('prediction_type') == PredictionType.FORECASTING:
            result = 'auto_ai.ts'
        elif self._auto_pipelines_parameters.get('prediction_type') == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
            result = 'auto_ai.tsad'
        return result

    def _initialize_wml_pipeline_metadata(self, **kwargs) -> None:
        """Initialization of WML Pipeline Document (WML client Meta Parameter)
            with provided parameters and default ones."""

        self._wml_pipeline_metadata = {
            self._wml_client.pipelines.ConfigurationMetaNames.NAME:
                self._auto_pipelines_parameters.get('name', 'Default name.'),
            self._wml_client.pipelines.ConfigurationMetaNames.DESCRIPTION:
                self._auto_pipelines_parameters.get('desc', 'Default description'),
            self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                'doc_type': 'pipeline',
                'version': '2.0',
                'pipelines': [{
                    'id': 'autoai',
                    'runtime_ref': 'hybrid',
                    'nodes': [{
                        'id': self._get_node_id(),
                        'type': 'execution_node',
                        'parameters': {
                            'stage_flag': True,
                            'output_logs': True,
                            'input_file_separator': self._auto_pipelines_parameters.get('csv_separator'),
                            'encoding': self._auto_pipelines_parameters.get('encoding'),
                            'drop_duplicates': self._auto_pipelines_parameters.get('drop_duplicates'),
                            'outliers_columns': self._auto_pipelines_parameters.get('outliers_columns'),
                            'incremental_learning': self._auto_pipelines_parameters.get('incremental_learning'),
                            'enable_early_stop': self._auto_pipelines_parameters.get('early_stop_enabled'),
                            'early_stop_window_size': self._auto_pipelines_parameters.get('early_stop_window_size'),
                            'optimization': {
                                'learning_type': self._get_learning_type(),
                                'cognito_transform_names':
                                    self._auto_pipelines_parameters.get('cognito_transform_names'),
                                'run_cognito_flag': True,
                            }
                        },
                        'runtime_ref': 'autoai',
                        'op': 'kube'
                    }]
                }],
                'runtimes': [{
                    'id': 'autoai',
                    'name': self._get_runtime_name(),
                    'app_data': {
                        'wml_data': {
                            'hardware_spec': {}
                        }
                    }
                }],
                'primary_pipeline': 'autoai'
            }
        }

        if self._auto_pipelines_parameters.get('holdout_size') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'holdout_param'] = self._auto_pipelines_parameters.get('holdout_size')

        if self._auto_pipelines_parameters.get('max_num_daub_ensembles') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'max_num_daub_ensembles'] = self._auto_pipelines_parameters.get('max_num_daub_ensembles')

        if kwargs.get('enable_all_data_sources') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['enable_all_data_sources'
            ] = kwargs['enable_all_data_sources']

        if kwargs.get('use_flight') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['use_flight'
            ] = kwargs['use_flight']

        # New parameters for autoai-core which are not supported in CP4D 3.0 and CPD 3.5
        if not self._wml_client.ICP_35 and \
                self._auto_pipelines_parameters.get('prediction_type') != PredictionType.FORECASTING and \
                self._auto_pipelines_parameters.get('prediction_type') != PredictionType.TIMESERIES_ANOMALY_PREDICTION:
            # Text transformer params
            if self._auto_pipelines_parameters.get('text_processing') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'text_processing_flag'] = self._auto_pipelines_parameters.get('text_processing')

            if self._auto_pipelines_parameters.get('word2vec_feature_number') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'text_processing_options'] = {'word2vec':
                                                {'output_dim': self._auto_pipelines_parameters.get('word2vec_feature_number')}}
            if self._auto_pipelines_parameters.get('text_columns_names'):
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'text_columns'] = self._auto_pipelines_parameters.get('text_columns_names')
                # end of Text transformer parameters
            if self._auto_pipelines_parameters.get('retrain_on_holdout') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'retrain_on_holdout'] = self._auto_pipelines_parameters.get('retrain_on_holdout')
            if self._auto_pipelines_parameters.get('numerical_columns') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'numerical_columns'] = self._auto_pipelines_parameters.get('numerical_columns')
            if self._auto_pipelines_parameters.get('categorical_columns') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'categorical_columns'] = self._auto_pipelines_parameters.get('categorical_columns')
        # end of section

        # pass fairness info for Cloud or CP4D 4.5
        if self._wml_client.CLOUD_PLATFORM_SPACES or self._wml_client.ICP_45 or self._wml_client.ICP_46 or self._wml_client.ICP_47:
            if self._auto_pipelines_parameters.get('fairness_info') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'fairness_info'] = self._auto_pipelines_parameters.get('fairness_info')
        # end of section

        # note: Additional parameters for benchmark subsampling with data detector
        if self._auto_pipelines_parameters.get('sampling_type') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'sampling_type'] = self._auto_pipelines_parameters.get('sampling_type')

        if self._auto_pipelines_parameters.get('sample_size_limit') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'sample_size_limit'] = self._auto_pipelines_parameters.get('sample_size_limit')

        if self._auto_pipelines_parameters.get('sample_rows_limit') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'sample_rows_limit'] = self._auto_pipelines_parameters.get('sample_rows_limit')

        if self._auto_pipelines_parameters.get('sample_percentage_limit') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'sample_percentage_limit'] = self._auto_pipelines_parameters.get('sample_percentage_limit')

        if self._auto_pipelines_parameters.get('n_parallel_data_connections') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'n_parallel_data_connections'] = self._auto_pipelines_parameters.get('n_parallel_data_connections')

        if self._auto_pipelines_parameters.get('calculate_data_metrics') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'calculate_data_metrics'] = self._auto_pipelines_parameters.get('calculate_data_metrics')

        if self._auto_pipelines_parameters.get('logical_batch_size') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'logical_batch_size'] = self._auto_pipelines_parameters.get('logical_batch_size')

        if self._auto_pipelines_parameters.get('metrics_on_logical_batch') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'metrics_on_logical_batch'] = self._auto_pipelines_parameters.get('metrics_on_logical_batch')

        if self._auto_pipelines_parameters.get('number_of_batch_rows') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'number_of_batch_rows'] = self._auto_pipelines_parameters.get('number_of_batch_rows')
        # --- end note

        if self._auto_pipelines_parameters.get('prediction_type') == PredictionType.FORECASTING:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'target_columns'] = self._auto_pipelines_parameters['prediction_columns']
            if self._auto_pipelines_parameters['timestamp_column_name'] is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'timestamp_column'] = self._auto_pipelines_parameters['timestamp_column_name']
            if self._auto_pipelines_parameters['backtest_num']:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'num_backtest'] = self._auto_pipelines_parameters['backtest_num']
            if self._auto_pipelines_parameters['lookback_window']:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'lookback_window'] = self._auto_pipelines_parameters['lookback_window']
            if self._auto_pipelines_parameters['forecast_window']:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'prediction_horizon'] = self._auto_pipelines_parameters['forecast_window']
            if self._auto_pipelines_parameters['backtest_gap_length'] != None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'gap_len'] = self._auto_pipelines_parameters['backtest_gap_length']
            if self._auto_pipelines_parameters['feature_columns']:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'feature_columns'] = self._auto_pipelines_parameters['feature_columns']
            if self._auto_pipelines_parameters['pipeline_types']:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'pipeline_type'] = 'customized'
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'customized_pipelines'] = [pipeline.value for pipeline in
                                               self._auto_pipelines_parameters['pipeline_types']]
            if self._auto_pipelines_parameters['supporting_features_at_forecast'] != None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'future_exogenous_available'] = self._auto_pipelines_parameters['supporting_features_at_forecast']
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'compute_pipeline_notebooks_flag'] = self._auto_pipelines_parameters.get('notebooks', False)
            if self._auto_pipelines_parameters.get('retrain_on_holdout') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'retrain_on_holdout'] = self._auto_pipelines_parameters.get('retrain_on_holdout')
        elif self._auto_pipelines_parameters.get('prediction_type') == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'feature_columns'] = self._auto_pipelines_parameters['feature_columns']
            if self._auto_pipelines_parameters['timestamp_column_name'] is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'timestamp_column'] = self._auto_pipelines_parameters['timestamp_column_name']
            if self._auto_pipelines_parameters.get('max_num_daub_ensembles') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'max_num_pipelines'] = self._auto_pipelines_parameters.get('max_num_daub_ensembles')
            if self._auto_pipelines_parameters['pipeline_types']:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'pipelines'] = [pipeline.value for pipeline in self._auto_pipelines_parameters['pipeline_types']]
            if self._auto_pipelines_parameters['scoring']:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'evaluation_metric'] = self._auto_pipelines_parameters['scoring']
            if self._auto_pipelines_parameters.get('confidence_level') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'confidence_level'] = self._auto_pipelines_parameters.get('confidence_level')
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'compute_pipeline_notebooks_flag'] = self._auto_pipelines_parameters.get('notebooks', False)
            if self._auto_pipelines_parameters.get('retrain_on_holdout') is not None:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'retrain_on_holdout'] = self._auto_pipelines_parameters.get('retrain_on_holdout')
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'].pop('max_num_daub_ensembles', None)
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'].pop('cognito_transform_names', None)
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'].pop('run_cognito_flag', None)
        else:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'label'] = self._auto_pipelines_parameters['prediction_column']
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'compute_pipeline_notebooks_flag'] = self._auto_pipelines_parameters.get('notebooks', False)
            if 'scoring' in self._auto_pipelines_parameters:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'scorer_for_ranking'] = self._auto_pipelines_parameters.get('scoring')

        if self._20_class_limit_removal_test:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT]['pipelines'][0]['nodes'][0]['parameters'].update({
                'one_vs_all_file': True
            })

        # note: in new v4 api we have missing hw_spec name
        t_size = self._auto_pipelines_parameters['t_shirt_size']
        if 0 < len(t_size) <= 2:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT]['runtimes'][0][
                'app_data']['wml_data']['hardware_spec']['name'] = t_size.upper()

        else:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT]['runtimes'][0][
                'app_data']['wml_data']['hardware_spec']['id'] = t_size
        # --- end note

        if self._auto_pipelines_parameters.get('train_sample_rows_test_size') is not None:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'train_sample_rows_test_size'] = self._auto_pipelines_parameters['train_sample_rows_test_size']

        if self._auto_pipelines_parameters.get('t_shirt_size') == 's':
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'daub_adaptive_subsampling_max_mem_usage'] = 6e9

        elif self._auto_pipelines_parameters.get('t_shirt_size') == 'm':
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'daub_adaptive_subsampling_max_mem_usage'] = 9e9

        elif self._auto_pipelines_parameters.get('t_shirt_size') == 'l':
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'daub_adaptive_subsampling_max_mem_usage'] = 15e9

        elif self._auto_pipelines_parameters.get('t_shirt_size') == 'xl':
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'daub_adaptive_subsampling_max_mem_usage'] = 25e9

        if self._auto_pipelines_parameters.get('include_only_estimators') is not None:
            # note: transform values from Enum
            try:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'daub_include_only_estimators'] = [
                    algorithm.value for algorithm in self._auto_pipelines_parameters.get('include_only_estimators')
                ]
            # note: if user pass strings instead of enums
            except AttributeError:
                pass

        if self._auto_pipelines_parameters.get('include_batched_ensemble_estimators') is not None:
            # note: transform values from Enum
            try:
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                    'global_stage_include_batched_ensemble_estimators'] = [
                    algorithm.value for algorithm in self._auto_pipelines_parameters.get('include_batched_ensemble_estimators')
                ]
            # note: if user pass strings instead of enums
            except AttributeError:
                pass

        # note: only pass positive label when scoring is binary
        if (self._auto_pipelines_parameters.get('positive_label')
                and self._auto_pipelines_parameters.get('prediction_type') == PredictionType.BINARY):
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'positive_label'] = self._auto_pipelines_parameters.get('positive_label')
        # --- end note

        # note: only pass daub_give_priority_to_runtime when is not None
        if self._auto_pipelines_parameters.get('daub_give_priority_to_runtime'):
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'][
                'daub_runtime_ranking_power'] = self._auto_pipelines_parameters.get('daub_give_priority_to_runtime')
        # --- end note

        # note: only pass excel_sheet when it is different than 0 and is not None
        if self._auto_pipelines_parameters.get('excel_sheet') and self._auto_pipelines_parameters.get(
                'excel_sheet') != 0:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'excel_sheet'] = self._auto_pipelines_parameters.get('excel_sheet')
        # --- end note

        # note: Fill test data params if specified (not the standard ones)
        if self._auto_pipelines_parameters.get('test_data_csv_separator') != ',':
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'test_input_file_separator'] = self._auto_pipelines_parameters.get('test_data_csv_separator')

        if self._auto_pipelines_parameters.get('test_data_excel_sheet') and self._auto_pipelines_parameters.get(
                'test_data_excel_sheet') != 0:
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'test_excel_sheet'] = self._auto_pipelines_parameters.get('test_data_excel_sheet')

        if self._auto_pipelines_parameters.get('test_data_encoding') != 'utf-8':
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters'][
                'test_encoding'] = self._auto_pipelines_parameters.get('test_data_encoding')
        # --- end note

        if self._auto_pipelines_parameters.get('prediction_type') == PredictionType.FORECASTING:
            if self._auto_pipelines_parameters.get('categorical_imputation_strategy'):
                raise InvalidImputationParameterTS()

            if self._auto_pipelines_parameters.get('numerical_imputation_strategy'):
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization']\
                    .update(get_values_for_imputation_strategy(
                    self._auto_pipelines_parameters.get('numerical_imputation_strategy'),
                    self._auto_pipelines_parameters.get('prediction_type'),
                    self._auto_pipelines_parameters.get('numerical_imputation_value')))

                if self._auto_pipelines_parameters.get('numerical_imputation_value') is not None:
                    if self._auto_pipelines_parameters.get('numerical_imputation_strategy') == ImputationStrategy.VALUE\
                            or (type(self._auto_pipelines_parameters.get('numerical_imputation_strategy')) is list
                                and ImputationStrategy.VALUE in self._auto_pipelines_parameters.get('numerical_imputation_strategy')):
                        self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                            'pipelines'][0]['nodes'][0]['parameters']['optimization']['imputer_fill_value'] = \
                            self._auto_pipelines_parameters.get('numerical_imputation_value')
                    else:
                        raise NumericalImputationStrategyValueMisused()

                if self._auto_pipelines_parameters.get('imputation_threshold'):
                    self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                        'pipelines'][0]['nodes'][0]['parameters']['optimization']['imputation_threshold'] = \
                        self._auto_pipelines_parameters.get('imputation_threshold')
        elif self._auto_pipelines_parameters.get('prediction_type') == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
            # To be changed
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'][0]['parameters']['optimization'].pop('daub_adaptive_subsampling_max_mem_usage', None)
        else:
            if self._auto_pipelines_parameters.get('numerical_imputation_value') is not None or self._auto_pipelines_parameters.get('imputation_threshold'):
                raise InvalidImputationParameterNonTS()

            if self._auto_pipelines_parameters.get('categorical_imputation_strategy'):
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization']['preprocessor_cat_imp_strategy'] = \
                    get_values_for_imputation_strategy(
                        self._auto_pipelines_parameters.get('categorical_imputation_strategy'),
                        self._auto_pipelines_parameters.get('prediction_type'))

            if self._auto_pipelines_parameters.get('numerical_imputation_strategy'):
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['optimization']['preprocessor_num_imp_strategy'] = \
                    get_values_for_imputation_strategy(
                        self._auto_pipelines_parameters.get('numerical_imputation_strategy'),
                        self._auto_pipelines_parameters.get('prediction_type'))

        # note: KB part: send version autoai_pod_version if set in pipeline details details (See supported formats)
        if not self._wml_client.ICP and self._auto_pipelines_parameters['autoai_pod_version'] is not None:
            self._wml_pipeline_metadata[
                self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT]['runtimes'][0]['version'] = (
                self._auto_pipelines_parameters['autoai_pod_version'])
        # --- end note

        # note: OBM part, only if user specified OBM as a preprocess method
        if self._auto_pipelines_parameters['data_join_graph']:
            # note: remove kb params for OBM only training
            if self._auto_pipelines_parameters.get('data_join_only'):
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'] = []
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'runtimes'] = []
            # --- end note
            else:
                # note: update autoai KB POD metadata
                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['inputs'] = [{"id": "kb_input",
                                                              "links": [{"node_id_ref": "obm",
                                                                         "port_id_ref": "obm_out"
                                                                         }]
                                                              }]

                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['outputs'] = [{"id": "outputData"
                                                               }]

                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['id'] = 'kb'

                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['runtime_ref'] = 'kb'

                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'runtimes'][0]['id'] = 'kb'
                # --- end note

            # note: insert OBM experiment metadata into the first place of nodes
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'pipelines'][0]['nodes'].insert(0, self._auto_pipelines_parameters['data_join_graph'].to_dict())
            # --- end note

            # note: insert OBM runtime metadata into the first place
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'runtimes'].insert(0, {"id": "obm",
                                       "name": "auto_ai.obm",
                                       "app_data": {
                                           "wml_data": {
                                               "hardware_spec": {
                                                   "name": "M-Spark",
                                                   "num_nodes": self._auto_pipelines_parameters['data_join_graph'].worker_nodes_number
                                               }
                                           }
                                       }
                                       }
                                   )

            # note: in new v4 api we have missing hw_spec name
            t_size = self._auto_pipelines_parameters['data_join_graph'].t_shirt_size
            self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                'runtimes'][0]['app_data']['wml_data']['hardware_spec']['name'] = f"{t_size.upper()}-Spark"
            # --- end note
            # --- end note

            if not self._wml_client.ICP:
                self._wml_pipeline_metadata[
                    self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'runtimes'][0]['version'] = self._auto_pipelines_parameters['obm_pod_version']

            else:
                if self._wml_client.ICP_45 or self._wml_client.ICP_46 or self._wml_client.ICP_47:
                    template_id = 'spark-3.2-automl-cp4d-template'
                elif not self._wml_client.ICP_35:
                    template_id = 'spark-3.0.0-automl-cp4d-template'
                else:
                    template_id = 'spark-2.4.0-automl-icp4d-template'

                self._wml_pipeline_metadata[self._wml_client.pipelines.ConfigurationMetaNames.DOCUMENT][
                    'pipelines'][0]['nodes'][0]['parameters']['engine'][
                    'template_id'] = template_id
        # --- end note

    def _initialize_wml_training_metadata(self,
                                          training_data_reference: List['DataConnection'],
                                          training_results_reference: 'DataConnection',
                                          test_data_references: List['DataConnection'] = None,
                                          test_output_data: 'DataConnection' = None) -> None:
        """Initialization of wml training metadata (WML client Meta Parameter).

        :param training_data_reference: data storage connection details to inform where training data is stored
        :type training_data_reference: list[DataConnection]

        :param training_results_reference: data storage connection details to store pipeline training results
        :type training_results_reference: DataConnection

        :param test_data_references: data storage connection details to inform where test / holdout data is stored
        :type test_data_references: list[DataConnection], optional

        :param test_output_data: data storage connection details to inform where joined test / holdout data is stored
            (joined data only)
        :type test_output_data: DataConnection, optional
        """
        self._wml_training_metadata = {
            self._wml_client.training.ConfigurationMetaNames.TAGS:
                ['autoai'],

            self._wml_client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES:
                [connection._to_dict() for connection in training_data_reference],

            self._wml_client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE:
                training_results_reference._to_dict(),

            self._wml_client.training.ConfigurationMetaNames.PIPELINE:
                {'id': self._wml_stored_pipeline_details['metadata']['id']},

            self._wml_client.training.ConfigurationMetaNames.NAME:
                f"{self._auto_pipelines_parameters.get('name', 'Default name.')[:40]} - "
                f"wml pipeline: {self._wml_stored_pipeline_details['metadata']['id']}",

            self._wml_client.training.ConfigurationMetaNames.DESCRIPTION:
                f"{self._auto_pipelines_parameters.get('desc', 'Default description')[:40]} - "
                f"wml pipeline: {self._wml_stored_pipeline_details['metadata']['id']}",
        }

        if test_data_references is not None:
            self._wml_training_metadata[
                self._wml_client.training.ConfigurationMetaNames.TEST_DATA_REFERENCES
            ] = [connection._to_dict() for connection in test_data_references]

        if test_output_data is not None:
            self._wml_training_metadata[
                self._wml_client.training.ConfigurationMetaNames.TEST_OUTPUT_DATA
            ] = test_output_data._to_dict()

        # note: Delete unnecessary keys for connection_asset in training_data_reference
        for connection in \
                self._wml_training_metadata[self._wml_client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES] +\
                [self._wml_training_metadata[self._wml_client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE]]:
            if connection.get("type") == DataConnectionTypes.CA:
                try:
                    del connection["connection"]["secret_access_key"]
                    del connection["connection"]["access_key_id"]
                    del connection["connection"]["endpoint_url"]
                except KeyError:
                    pass
        # --- end note

    ##########################################################
    #   WML Pipeline Part / AutoPipelineOptimizer Init Part  #
    ##########################################################
    def initiate_remote_resources(self, params: dict, **kwargs) -> None:
        """Initializes the AutoPipelines with supplied parameters.

        :param params: AutoAi experiment parameters
        :type params: dict
        """
        self._auto_pipelines_parameters = params
        self._initialize_wml_pipeline_metadata(**kwargs)
        if 'enable_all_data_sources' in kwargs:
            del kwargs['enable_all_data_sources']

        if 'use_flight' in kwargs:
            del kwargs['use_flight']

        self._wml_stored_pipeline_details = self._wml_client.pipelines.store(meta_props=self._wml_pipeline_metadata, **kwargs)

    def get_params(self) -> dict:
        """Get configuration parameters of AutoPipelineOptimizer.

        :return: AutoPipelineOptimizer parameters
        :rtype: dict
        """
        if self._auto_pipelines_parameters is not None:
            self._auto_pipelines_parameters['run_id'] = self._current_run_id

        return self._auto_pipelines_parameters

    ##########################################################
    #   WML Training Part / AutoPipelineOptimizer Run Part   #
    ##########################################################
    def fit(self,
            training_data_reference: List['DataConnection'],
            training_results_reference: 'DataConnection',
            background_mode: bool = True,
            test_data_references: List['DataConnection'] = None,
            test_output_data: 'DataConnection' = None) -> dict:
        """Run a training process on WML of autoai on top of the training data referenced by training_data_connection.

        :param training_data_reference: data storage connection details to inform where training data is stored
        :type training_data_reference: list[DataConnection]

        :param training_results_reference: data storage connection details to store pipeline training results
        :type training_results_reference: DataConnection

        :param background_mode: indicator if fit() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :param test_data_references: data storage connection details to inform where test / holdout data is stored
        :type test_data_references: list[DataConnection], optional

        :param test_output_data: data storage connection details to store test data output (joined data only)
        :type test_output_data: DataConnection, optional

        :return: run details
        :rtype: dict
        """
        self._initialize_wml_training_metadata(training_data_reference=training_data_reference,
                                               training_results_reference=training_results_reference,
                                               test_data_references=test_data_references,
                                               test_output_data=test_output_data)

        run_params = self._wml_client.training.run(meta_props=self._wml_training_metadata,
                                                   asynchronous=True)

        self._current_run_id = run_params['metadata'].get('id', run_params['metadata'].get('guid'))

        if background_mode:
            return self._wml_client.training.get_details(training_uid=self._current_run_id, _internal=True)

        else:
            wml_pipeline_details = self.get_params()
            number_of_estimators = wml_pipeline_details.get('max_num_daub_ensembles', 2)

            base_url = self._wml_client.service_instance._href_definitions.get_trainings_href()
            url = f"{base_url.replace('https', 'wss')}/{self._current_run_id}"

            params = self._wml_client._params()

            if self._wml_client.default_project_id:
                url = f"{url}?project_id={self._wml_client.default_project_id}&version={params.get('version')}"
            elif self._wml_client.default_space_id:
                url = f"{url}?space_id={self._wml_client.default_space_id}&version={params.get('version')}"

            gen = ProgressGenerator()
            end = False

            progress_bar = ProgressBar(desc="Total", total=gen.get_total(), position=0, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]')

            progress_bar.set_description(desc='Started waiting for resources')

            attempts_no = 10
            disconnection_no = 0

            while not end:
                websocket = WebSocket(url)
                websocket.add_header(bytes("Authorization", "utf-8"),
                                     bytes("Bearer " + self._wml_client.service_instance._get_token(), "utf-8"))

                try:
                    for event in websocket.connect():
                        try:
                            websocket.send_text('Ping')
                        except:
                            pass
                        if event.name == 'text':
                            status = json.loads(event.text)['entity']['status']

                            try:
                                process = status['metrics'][0]['context']['intermediate_model']['process']
                            except:
                                process = ''

                            if status.get('state') == RunStateTypes.FAILED:
                                sleep(3)

                                message = status.get('message', {}).get('text', 'Training failed')
                                progress_bar.set_description(desc=message)

                                if 'Not Found' in str(message) or 'HeadObject' in str(message):
                                    raise AutoAIComputeError(self._current_run_id,
                                                             reason=f"Fetching training data error. Please check if COS credentials, "
                                                                    f"bucket name and path are correct or file system path is correct.")

                                elif 'Failed_Image' in str(message):
                                    raise AutoAIComputeError(
                                        self._current_run_id,
                                        reason=f"This version of WML V4 SDK with AutoAI component is no longer supported. "
                                               f"Please update your package to the newest version. "
                                               f"Error: {message}")

                                else:
                                    raise AutoAIComputeError(self._current_run_id, reason=f"Error: {message}")
                            elif 'message' in status and 'text' in status['message']:
                                text = status['message']['text']

                                if self._current_run_id in text and 'completed' in text:
                                    end = True
                                    progress_bar.set_description(desc=text)
                                    progress_bar.last_update()
                                    progress_bar.close()
                                    break
                                elif 'AutoAI model computed' not in text:
                                    prepared_text = text.split('Node automl: ')[-1].split("global:")[-1].split(" Message: ")[-1]
                                    max_len = 40

                                    if len(prepared_text) > max_len:
                                        progress_bar.set_description(desc=prepared_text[:(max_len - 3)] + '...')
                                    else:
                                        progress_bar.set_description(
                                            desc=prepared_text + ' ' * (max_len - len(prepared_text)))

                                progress_bar.increment_counter(gen.get_progress(process))
                                progress_bar.update()
                            elif process:
                                progress_bar.increment_counter(gen.get_progress(process))
                                progress_bar.update()
                        elif event.name in ['closing', 'closed']:
                            end = True
                            websocket.close()
                            break
                        elif event.name in ['disconnected']:
                            disconnection_no += 1
                            websocket.close()
                            break
                except:
                    websocket.close()
                    break

                if disconnection_no >= attempts_no:
                    raise CannotConnectToWebsocket(attempts_no)

            return self._wml_client.training.get_details(training_uid=self._current_run_id, _internal=True)

    def get_run_status(self) -> str:
        """Check status/state of initialized AutoPipelineOptimizer run if ran in background mode.

        :return: run status
        :rtype: str
        """

        if self._current_run_id is None:
            raise FitNeeded(reason="Firstly schedule a fit job by using the fit() method.")

        return self._wml_client.training.get_status(training_uid=self._current_run_id).get('state')

    def get_run_details(self, include_metrics: bool = False) -> dict:
        """Get fit/run details.

        :param include_metrics: indicates to include metrics in the training details output
        :type include_metrics: bool, optional

        :return: AutoPipelineOptimizer fit/run details
        :rtype: dict
        """

        if self._current_run_id is None:
            raise FitNeeded(reason="Firstly schedule a fit job by using the fit() method.")

        details = self._wml_client.training.get_details(training_uid=self._current_run_id, _internal=True)

        if include_metrics:
            return details

        if details['entity']['status'].get('metrics', False):
            del details['entity']['status']['metrics']
            return details
        else:
            return details

    def cancel_run(self) -> None:
        """Cancels an AutoAI run."""
        if self._current_run_id is None:
            raise FitNeeded(reason="To cancel an AutoAI run, first schedule a fit job by using the fit() method.")

        self._wml_client.training.cancel(training_uid=self._current_run_id)

    #################################################################################
    #   WML Auto_AI Trained Pipelines Part / AutoPipelineOptimizer Pipelines Part   #
    #################################################################################
    def summary(self, scoring: str = None, sort_by_holdout_score: bool = True) -> 'DataFrame':
        """Prints AutoPipelineOptimizer Pipelines details (autoai trained pipelines).

        :param scoring: scoring metric which user wants to use to sort pipelines by,
            when not provided use optimized one
        :type scoring: string, optional

        :param sort_by_holdout_score: indicates if we want to sort pipelines by holdout metric or by training one,
            by default use holdout metric
        :type sort_by_holdout_score: bool, optional

        :return: DataFrame with computed pipelines and ML metrics
        :rtype: pandas.DataFrame
        """
        if self._current_run_id is None:
            raise FitNeeded(reason="To list computed pipelines, first schedule a fit job by using the fit() method.")

        details = self._wml_client.training.get_details(training_uid=self._current_run_id, _internal=True)
        optimized_scoring = self._auto_pipelines_parameters.get('scoring', None)

        if scoring is None:
            scoring = optimized_scoring

        summary = create_summary(details=details, scoring=scoring, sort_by_holdout_score=sort_by_holdout_score)

        if isinstance(optimized_scoring, str) and optimized_scoring.startswith('neg_'):
            optimized_scoring = optimized_scoring[4:]

        return summary.rename({
            f"training_{optimized_scoring}": f"training_{optimized_scoring}_(optimized)"}, axis='columns')

    def get_pipeline_details(self, pipeline_name: str = None) -> dict:
        """Fetch specific pipeline details, eg. steps etc.

        :param pipeline_name: pipeline name eg. Pipeline_1, if not specified, best pipeline parameters will be fetched
        :type pipeline_name: str, optional

        :return: pipeline parameters
        :rtype: dict
        """

        if self._current_run_id is None:
            raise FitNeeded(reason="To list computed pipelines parameters, "
                                   "first schedule a fit job by using a fit() method.")

        if pipeline_name is None:
            pipeline_name = self.summary().index[0]
        run_params = self._wml_client.training.get_details(training_uid=self._current_run_id, _internal=True)

        pipeline_parameters = {
            "composition_steps": [],
            "pipeline_nodes": [],
        }

        # note: retrieve all additional pipeline evaluation information
        for pipeline in run_params['entity']['status'].get('metrics', [])[::-1]:
            if pipeline['context']['intermediate_model']['name'].split('P')[-1] == pipeline_name.split('_')[-1]:

                pipeline_parameters['composition_steps'] = pipeline['context']['intermediate_model'][
                    'composition_steps']
                pipeline_parameters['pipeline_nodes'] = pipeline['context']['intermediate_model']['pipeline_nodes']

                if 'fairness' in pipeline['context']:
                    pipeline_parameters['fairness_details'] = pipeline['context'].get('fairness')

                if 'timeseries' not in pipeline['context']:
                    pipeline_parameters['ml_metrics'] = self._get_metrics(pipeline)
                else:
                    if self._auto_pipelines_parameters['prediction_type'] == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
                        for pipeline_metrics in run_params['entity']['status'].get('metrics', [])[::-1]:
                            pipeline_phase = pipeline_metrics['context'].get('phase')
                            if pipeline_metrics['context']['intermediate_model']['name'].split('P')[-1] ==pipeline_name.split('_')[-1] and pipeline_phase == "after_holdout_execution":
                                pipeline_parameters['tsad_metrics'] = self._get_metrics(pipeline_metrics)
                                break
                    else:
                        if not 'ts_metrics' in pipeline_parameters:
                            pipeline_parameters['ts_metrics'] = {}
                        pipeline_parameters['ts_metrics'].update(self._get_metrics(pipeline))

                if 'incremental_training' in pipeline['context']:
                    pipeline_parameters['learning_curve'] = self._get_learning_curve(pipeline, run_params)

                if (self._auto_pipelines_parameters['prediction_type'] == 'binary'
                        or (self._auto_pipelines_parameters['prediction_type'] == 'classification'
                            and 'binary_classification' in str(run_params))):

                    pipeline_parameters['features_importance'] = self._get_features_importance(pipeline)
                    pipeline_parameters['confusion_matrix'] = self._get_confusion_matrix(pipeline, run_params)
                    pipeline_parameters['roc_curve'] = self._get_roc_curve(pipeline, run_params)

                elif (self._auto_pipelines_parameters['prediction_type'] == 'multiclass'
                      or (self._auto_pipelines_parameters['prediction_type'] == 'classification'
                          and 'multi_class_classification' in str(run_params))):

                    one_vs_all_data = self._get_data_from_property_or_file('one_vs_all',
                                                   pipeline['context'].get('multi_class_classification', {'one_vs_all': []}),
                                                   run_params['entity']['results_reference'], [{}, {}])

                    pipeline_parameters['features_importance'] = self._get_features_importance(pipeline)
                    pipeline_parameters['roc_curve'] = self._get_one_vs_all(one_vs_all_data, run_params, 'roc_curve')
                    pipeline_parameters['confusion_matrix'] = self._get_one_vs_all(one_vs_all_data, run_params, 'confusion_matrix')

                elif self._auto_pipelines_parameters['prediction_type'] == 'regression':
                    pipeline_parameters['features_importance'] = self._get_features_importance(pipeline)

                elif (self._auto_pipelines_parameters['prediction_type'] == 'forecasting' or
                      self._auto_pipelines_parameters['prediction_type'] == PredictionType.TIMESERIES_ANOMALY_PREDICTION):
                    phases_to_combine_visualization = ['after_backtest_execution', 'after_holdout_execution']
                    for pipeline in run_params['entity']['status'].get('metrics', [])[::-1]:
                        pipeline_phase = pipeline['context'].get('phase')
                        # note: end earlier if all phases were added to the visualisation
                        if len(phases_to_combine_visualization) == 0:
                            break
                        #end note
                        if pipeline['context']['intermediate_model']['name'].split('P')[-1] == pipeline_name.split('_')[
                            -1] and pipeline_phase in phases_to_combine_visualization:
                            phases_to_combine_visualization.remove(pipeline_phase)
                            data = self._get_data_from_property_or_file('daub_status', pipeline['context']['timeseries'],
                                                                             run_params['entity']['results_reference'],
                                                                             [{}, {}])
                            if 'visualization' in data:
                                if 'visualization' not in pipeline_parameters:
                                    pipeline_parameters['visualization'] = {}
                                pipeline_parameters['visualization'].update(data['visualization'])


                else:
                    raise InvalidPredictionType(self._auto_pipelines_parameters['prediction_type'],
                                                reason="Prediction type not recognized.")

                break

        # --- end note

        return pipeline_parameters

    @staticmethod
    def _get_metrics(pipeline_details: Dict) -> 'DataFrame':
        """Retrieve evaluation metrics data from particular pipeline details."""
        is_ml_metrics = 'ml_metrics' in pipeline_details
        is_ts_metrics = 'ts_metrics' in pipeline_details
        is_tsad_metrics = 'tsad_metrics' in pipeline_details

        if not is_ml_metrics and not is_ts_metrics and not is_tsad_metrics:
            raise NoAvailableMetrics()

        if is_ml_metrics:
            data = pipeline_details['ml_metrics']
        elif is_ts_metrics:
            data = pipeline_details['ts_metrics']
        elif is_tsad_metrics:
            data = pipeline_details['tsad_metrics']

        columns = [field for field in data.keys()]
        values = [[value for value in data.values()]]

        metrics = DataFrame(data=values, columns=columns)
        metrics.index = ['score']
        metrics = metrics.transpose()

        return metrics

    def _get_data_from_property_or_file(self, property_name: str, details: Dict, results_reference: Dict, default_value: List) -> List:
        if f'{property_name}_location' in details:
            path = details[f'{property_name}_location']
            conn = DataConnection._from_dict(results_reference)
            conn._wml_client = self._wml_client
            if path.endswith('.csv'):
                return conn._download_csv_file(path)
            else:
                return conn._download_json_file(path)
        elif property_name in details:  # TODO probably should be removed in future
            return details[property_name]
        else:
            return default_value

    def _get_learning_curve(self, pipeline_details: Dict, run_params: Dict) -> 'DataFrame':
        """Retrieve learning curve data from file."""

        data = self._get_data_from_property_or_file('measures',
                                                    pipeline_details['context'].get('incremental_training', {}),
                                                    run_params['entity']['results_reference'],
                                                    [{}, {}])

        return data

    def _get_confusion_matrix(self, pipeline_details: Dict, run_params: Dict) -> 'DataFrame':
        """Retrieve confusion matrix data from particular pipeline details or from file. (Binary)"""
        data = self._get_data_from_property_or_file('confusion_matrix',
                                                         pipeline_details['context'].get('binary_classification', {}),
                                                         run_params['entity']['results_reference'],
                                                         [{}, {}])
        columns = [name for name in data[0].keys()]
        values = [[value for value in item.values()] for item in data]
        confusion_matrix_data = DataFrame(data=values, columns=columns)

        if 'true_class' in confusion_matrix_data.columns:
            confusion_matrix_data.index = confusion_matrix_data['true_class']
            confusion_matrix_data.drop(['true_class'], axis=1, inplace=True)

        return confusion_matrix_data

    @staticmethod
    def _get_features_importance(pipeline_details: Dict) -> 'DataFrame':
        """Retrieve features importance data from particular pipeline details."""
        data = pipeline_details['context'].get('features_importance', [{'features': {}}])[0]['features']
        columns = [name for name in data.keys()]
        values = [[value for value in data.values()]]

        features_importance_data = DataFrame(data=values, columns=columns, index=['features_importance'])
        features_importance_data = features_importance_data.transpose()
        features_importance_data.sort_values(by='features_importance', ascending=False, inplace=True)

        return features_importance_data

    def _get_roc_curve(self, pipeline_details: Dict, run_params: Dict) -> 'DataFrame':
        """Retrieve roc curve data from particular pipeline details or from file. (Binary)"""
        data = self._get_data_from_property_or_file('roc_curve',
                                                         pipeline_details['context'].get('binary_classification', {}),
                                                         run_params['entity']['results_reference'],
                                                         [])

        true_classes = [item['true_class'] for item in data]

        if len(data) > 0:
            values = [data[0]['fpr'], data[1]['fpr'], data[0]['thresholds'], data[1]['thresholds'], data[0]['tpr'],
                      data[1]['tpr']]
        else:
            values = []

        index = MultiIndex.from_product([['fpr', 'thresholds', 'tpr'], true_classes],
                                        names=['measurement types', 'true_class'])

        roc_curve = DataFrame(data=values, index=index)

        return roc_curve

    def _get_one_vs_all(self, one_vs_all_data: List, run_params: Dict, option: str) -> 'DataFrame':
        """Retrieve roc curve or confusion matrix data from particular pipeline details. (Multiclass)"""

        one_vs_all_data = [x for x in one_vs_all_data if 'multi-class' not in x]

        if option == 'confusion_matrix':
            confusion_matrix_data = [self._get_data_from_property_or_file('confusion_matrix', item,
                                     run_params['entity']['results_reference'], [{}, {}]) for item in one_vs_all_data]

            columns = ['fn', 'fp', 'tn', 'tp', 'true_class']
            confusion_matrix_values = [[item[c] for c in columns] for item in confusion_matrix_data]
            confusion_matrix_data = DataFrame(data=confusion_matrix_values,
                                              columns=columns)
            confusion_matrix_data.index = confusion_matrix_data['true_class']
            confusion_matrix_data.drop(['true_class'], axis=1, inplace=True)

            return confusion_matrix_data

        elif option == 'roc_curve':
            roc_curve_data = [self._get_data_from_property_or_file('roc_curve', item,
                              run_params['entity']['results_reference'], []) for item in one_vs_all_data]

            true_classes = [item['true_class'] for item in roc_curve_data]
            values = [item['fpr'] for item in roc_curve_data] + [item['thresholds'] for item in
                                                                    roc_curve_data] + [item['tpr'] for item in roc_curve_data]
            index = MultiIndex.from_product([['fpr', 'thresholds', 'tpr'], true_classes],
                                            names=['measurement types', 'true_class'])
            roc_curve = DataFrame(data=values, index=index)

            return roc_curve

        else:
            raise ValueError(f"Wrong option for \"one_vs_all\": {option}. "
                             f"Available options: [confusion_matrix, roc_curve]")

    def get_pipeline(self, pipeline_name: str, local_path: str = '.',
                     persist: 'bool' = False) -> Tuple['Pipeline', bool]:
        """Download specified pipeline from WML.

        :param pipeline_name: pipeline name, if you want to see the pipelines names, please use summary() method
        :type pipeline_name: str

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param persist: indicates if selected pipeline should be stored locally
        :type persist: bool, optional

        :return: Scikit-Learn pipeline and lale check result
        :rtype: tuple[Pipeline, bool]
        """

        # note: Show warning if OBM
        if 'auto_ai.obm' in str(self._wml_stored_pipeline_details):
            warnings.warn("OBM pipeline can be used only for inspection and deployment purposes.", Warning, stacklevel=2)
        # end note

        run_params = self._wml_client.training.get_details(training_uid=self._current_run_id, _internal=True)

        # note: recreation of s3 creds from connection asset
        results_reference = DataConnection._from_dict(run_params['entity']['results_reference'])
        results_reference._wml_client = self._wml_client
        results_reference._check_if_connection_asset_is_s3()
        run_params['entity']['results_reference'] = results_reference._to_dict()
        # --- end note

        if 'timeseries' in run_params['entity']['status']['metrics'][0]['context']:
            try_import_autoai_ts_libs()
        else:
            try_import_autoai_libs()

        try:
            pipelines, check_lale = fetch_pipelines(
                run_params=run_params,
                path=local_path,
                pipeline_name=pipeline_name,
                load_pipelines=True,
                store=persist,
                wml_client=self._wml_client,
                auto_pipelines_parameters=self._auto_pipelines_parameters
            )

        except LibraryNotCompatible as e:
            raise e

        except Exception as e:
            raise PipelineNotLoaded(
                pipeline_name,
                reason=f'Fit finished but there was some error during loading a pipeline from WML. Error: {e}')

        return pipelines.get(pipeline_name), check_lale

    def get_pipeline_notebook(self, pipeline_name: str, local_path: str = '.', filename: str = None) -> str:
        """Download specified pipeline notebook from WML.

        :param pipeline_name: pipeline name, if you want to see the pipelines names, please use summary() method
        :type pipeline_name: str

        :param local_path: working directory path
        :type local_path: str, optional

        :param filename: filename under which the pipeline notebook will be saved
        :type filename: str, optional

        :return: path to saved pipeline notebook
        :rtype: str
        """

        # note: Show warning if OBM
        if 'auto_ai.obm' in str(self._wml_stored_pipeline_details):
            warnings.warn("OBM pipeline notebook can be used only for inspection and deployment purposes.", Warning, stacklevel=2)
        # end note

        run_params = self._wml_client.training.get_details(training_uid=self._current_run_id, _internal=True)

        # note: recreation of s3 creds from connection asset
        results_reference = DataConnection._from_dict(run_params['entity']['results_reference'])
        results_reference._wml_client = self._wml_client
        results_reference._check_if_connection_asset_is_s3()
        run_params['entity']['results_reference'] = results_reference._to_dict()
        # --- end note

        return _download_notebook(
            run_params=run_params,
            path=local_path,
            pipeline_name=pipeline_name,
            wml_client=self._wml_client,
            filename=filename
        )

    def get_best_pipeline(self, local_path: str = '.', persist: 'bool' = False) -> Tuple['Pipeline', bool]:
        """Download best pipeline from WML.

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param persist: indicates if selected pipeline should be stored locally
        :type persist: bool, optional

        :return: Scikit-Learn pipeline and lale check result
        :rtype: tuple[Pipeline, bool]
        """
        best_pipeline_name = self.summary().index[0]
        return self.get_pipeline(best_pipeline_name, persist=persist)
