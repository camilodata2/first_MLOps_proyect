#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import copy
from typing import List, Union
from warnings import warn

from ibm_watson_machine_learning.preprocessing import DataJoinGraph
from ibm_watson_machine_learning.utils.autoai.enums import (
    TShirtSize, ClassificationAlgorithms, RegressionAlgorithms, ForecastingAlgorithms, PredictionType, Metrics, \
    Transformers, DataConnectionTypes, PipelineTypes, PositiveLabelClass, ClassificationAlgorithmsCP4D,
    RegressionAlgorithmsCP4D, ForecastingAlgorithmsCP4D, SamplingTypes, ImputationStrategy, ForecastingPipelineTypes,
    TimeseriesAnomalyPredictionPipelineTypes, TimeseriesAnomalyPredictionAlgorithms)
from ibm_watson_machine_learning.utils.autoai.errors import LocalInstanceButRemoteParameter, MissingPositiveLabel, \
    NonForecastPredictionColumnMissing, ForecastPredictionColumnsMissing, ForecastingCannotBeRunAsLocalScenario, \
    TSNotSupported, TSADNotSupported, ParamOutOfRange, ImputationListNotSupported, \
    MissingEstimatorForExistingBatchedEstimator, TimeseriesAnomalyPredictionFeatureColumnsMissing, \
    TimeseriesAnomalyPredictionCannotBeRunAsLocalScenario, TimeseriesAnomalyPredictionUnsupportedMetric
from ibm_watson_machine_learning.utils.autoai.utils import check_dependencies_versions, \
    validate_additional_params_for_optimizer, validate_optimizer_enum_values, \
    translate_imputation_string_strategy_to_enum, translate_estimator_string_to_enum, \
    translate_batched_estimator_string_to_enum
from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.wml_client_error import ForbiddenActionForGitBasedProject, WMLClientError
from ibm_watson_machine_learning.messages.messages import Messages
from .engines import WMLEngine
from .optimizers import LocalAutoPipelines, RemoteAutoPipelines
from .runs import AutoPipelinesRuns, LocalAutoPipelinesRuns
from ..base_experiment.base_experiment import BaseExperiment

__all__ = [
    "AutoAI"
]


class AutoAI(BaseExperiment):
    """AutoAI class for pipeline models optimization automation.

    :param wml_credentials: credentials to Watson Machine Learning instance
    :type wml_credentials: dict

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio Space
    :type space_id: str, optional

    :param verify: user can pass as verify one of following:

        - the path to a CA_BUNDLE file
        - the path of directory with certificates of trusted CAs
        - `True` - default path to truststore will be taken
        - `False` - no verification will be made
    :type verify: bool or str, optional

    **Example**

    .. code-block:: python

        from ibm_watson_machine_learning.experiment import AutoAI

        experiment = AutoAI(
            wml_credentials={
                "apikey": "...",
                "iam_apikey_description": "...",
                "iam_apikey_name": "...",
                "iam_role_crn": "...",
                "iam_serviceid_crn": "...",
                "instance_id": "...",
                "url": "https://us-south.ml.cloud.ibm.com"
            },
            project_id="...",
            space_id="...")
    """
    # note: initialization of AutoAI enums as class properties

    # note: Enums with estimators can be overwritten  in _init based on environment type (CPD or Cloud)
    ClassificationAlgorithms = ClassificationAlgorithms
    RegressionAlgorithms = RegressionAlgorithms
    ForecastingAlgorithms = ForecastingAlgorithms
    # end note
    TShirtSize = TShirtSize
    PredictionType = PredictionType
    Metrics = Metrics
    Transformers = Transformers
    DataConnectionTypes = DataConnectionTypes
    PipelineTypes = PipelineTypes
    SamplingTypes = SamplingTypes

    def __init__(self,
                 wml_credentials: Union[dict, 'WorkSpace'] = None,
                 project_id: str = None,
                 space_id: str = None,
                 verify=None) -> None:
        # note: as workspace is not clear enough to understand, there is a possibility to use pure
        # wml credentials with project and space IDs, but in addition we
        # leave a possibility to use a previous WorkSpace implementation, it could be passed as a first argument
        if wml_credentials is None:
            self._workspace = None
            self.runs = LocalAutoPipelinesRuns()

        else:
            if isinstance(wml_credentials, WorkSpace):
                self._workspace = wml_credentials
            else:
                self._workspace = WorkSpace(wml_credentials=wml_credentials.copy(),
                                            project_id=project_id,
                                            space_id=space_id,
                                            verify=verify)

            self.project_id = self._workspace.project_id
            self.space_id = self._workspace.space_id
            self.runs = AutoPipelinesRuns(engine=WMLEngine(self._workspace))
            self.runs._workspace = self._workspace

        #self._block_autoai_on_git_based_project()

        self._init_estimator_enums()

        self._20_class_limit_removal_test = False
        # --- end note

    def runs(self, *, filter: str) -> Union['AutoPipelinesRuns', 'LocalAutoPipelinesRuns']:
        """Get the historical runs but with WML Pipeline name filter (for remote scenario).
        Get the historical runs but with experiment name filter (for local scenario).

        :param filter: WML Pipeline name to filter the historical runs or experiment name to filter
            the local historical runs
        :type filter: str

        :return: object managing the list of runs
        :rtype: AutoPipelinesRuns or LocalAutoPipelinesRuns

        **Example**

        .. code-block:: python

            from ibm_watson_machine_learning.experiment import AutoAI

            experiment = AutoAI(...)
            experiment.runs(filter='Test').list()
        """

        if self._workspace is None:
            return LocalAutoPipelinesRuns(filter=filter)

        else:
            return AutoPipelinesRuns(engine=WMLEngine(self._workspace.wml_client), filter=filter)

    def optimizer(self,
                  name: str,
                  *,
                  prediction_type: 'PredictionType',
                  prediction_column: str = None,
                  prediction_columns: List[str] = None,
                  timestamp_column_name: str = None,
                  scoring: 'Metrics' = None,
                  desc: str = None,
                  test_size: float = None,  # deprecated
                  holdout_size: float = None,
                  max_number_of_estimators: int = None,
                  train_sample_rows_test_size: float = None,
                  include_only_estimators: List[Union['ClassificationAlgorithms', 'RegressionAlgorithms',
                                                      'ForecastingAlgorithms', 'TimeseriesAnomalyPredictionAlgorithms']] = None,
                  daub_include_only_estimators: List[Union['ClassificationAlgorithms', 'RegressionAlgorithms']] = None,  # deprecated
                  include_batched_ensemble_estimators: List[Union['BatchedClassificationAlgorithms',
                                                                  'BatchedRegressionAlgorithms']] = None,
                  backtest_num: int = None,
                  lookback_window: int = None,
                  forecast_window: int = None,
                  backtest_gap_length: int = None,
                  feature_columns: List[str] = None,
                  pipeline_types: List[Union['ForecastingPipelineTypes', 'TimeseriesAnomalyPredictionPipelineTypes']] = None,
                  supporting_features_at_forecast: bool = None,
                  cognito_transform_names: List['Transformers'] = None,
                  data_join_graph: 'DataJoinGraph' = None,
                  csv_separator: Union[List[str], str] = ',',
                  excel_sheet: Union[str, int] = None,
                  encoding: str = 'utf-8',
                  positive_label: str = None,
                  data_join_only: bool = False,
                  drop_duplicates: bool = True,
                  outliers_columns: list = None,
                  text_processing: bool = None,
                  word2vec_feature_number: int = None,
                  daub_give_priority_to_runtime: float = None,
                  fairness_info: dict = None,
                  sampling_type: 'SamplingTypes' = None,
                  sample_size_limit: int = None,
                  sample_rows_limit: int = None,
                  sample_percentage_limit: float = None,
                  n_parallel_data_connections: int = None,
                  number_of_batch_rows: int = None,
                  categorical_imputation_strategy: ImputationStrategy = None,
                  numerical_imputation_strategy: ImputationStrategy = None,
                  numerical_imputation_value: float = None,
                  imputation_threshold: float = None,
                  retrain_on_holdout: bool = None,
                  categorical_columns: list = None,
                  numerical_columns: list = None,
                  test_data_csv_separator: Union[List[str], str] = ',',
                  test_data_excel_sheet: str = None,
                  test_data_encoding: str = 'utf-8',
                  confidence_level: float = None,
                  incremental_learning: bool = None,
                  early_stop_enabled: bool = None,
                  early_stop_window_size: int = None,
                  **kwargs) -> Union['RemoteAutoPipelines', 'LocalAutoPipelines']:
        """
        Initialize an AutoAi optimizer.

        :param name: name for the AutoPipelines
        :type name: str

        :param prediction_type: type of the prediction
        :type prediction_type: PredictionType

        :param prediction_column: name of the target/label column, required for `multiclass`, `binary` and `regression`
            prediction types
        :type prediction_column: str, optional

        :param prediction_columns: names of the target/label columns, required for `forecasting` prediction type
        :type prediction_columns: list[str], optional

        :param timestamp_column_name: name of timestamp column for time series forecasting
        :type timestamp_column_name: str, optional

        :param scoring: type of the metric to optimize with, not used for forecasting
        :type scoring: Metrics, optional

        :param desc: description
        :type desc: str, optional

        :param test_size: deprecated, use `holdout_size` instead

        :param holdout_size: percentage of the entire dataset to leave as a holdout
        :type holdout_size: float, optional

        :param max_number_of_estimators: maximum number (top-K ranked by DAUB model selection)
            of the selected algorithm, or estimator types, for example `LGBMClassifierEstimator`,
            `XGBoostClassifierEstimator`, or `LogisticRegressionEstimator` to use in pipeline composition,
            the default is `None` that means the true default value will be determined by
            the internal different algorithms, where only the highest ranked by model selection algorithm type is used
        :type max_number_of_estimators: int, optional

        :param train_sample_rows_test_size: training data sampling percentage
        :type train_sample_rows_test_size: float, optional

        :param daub_include_only_estimators: deprecated, use `include_only_estimators` instead
        
        :param include_batched_ensemble_estimators: list of batched ensemble estimators to include 
            in computation process, see: AutoAI.BatchedClassificationAlgorithms, AutoAI.BatchedRegressionAlgorithms
        :type include_batched_ensemble_estimators: 
            list[BatchedClassificationAlgorithms or BatchedRegressionAlgorithms], optional

        :param include_only_estimators: list of estimators to include in computation process, see:
            AutoAI.ClassificationAlgorithms, AutoAI.RegressionAlgorithms or AutoAI.ForecastingAlgorithms
        :type include_only_estimators: List[ClassificationAlgorithms or RegressionAlgorithms or ForecastingAlgorithms]], optional

        :param backtest_num: number of backtests used for forecasting prediction type, default value: 4,
            value from range [0, 20]
        :type backtest_num: int, optional

        :param lookback_window: length of lookback window used for forecasting prediction type,
            default value: 10, if set to -1 lookback window will be auto-detected
        :type lookback_window: int, optional

        :param forecast_window: length of forecast window used for forecasting prediction type, default value: 1,
            value from range [1, 60]
        :type forecast_window: int, optional

        :param backtest_gap_length: gap between backtests used for forecasting prediction type,
            default value: 0, value from range [0, data length / 4]
        :type backtest_gap_length: int, optional

        :param feature_columns: list of feature columns used for forecasting prediction type,
            may contain target column and/or supporting feature columns, list of columns to be detected whether there are anomalies for timeseries anomaly prediction type
        :type feature_columns: list[str], optional

        :param pipeline_types: list of pipeline types to be used for forecasting or timeseries anomaly prediction type
        :type pipeline_types: list[ForecastingPipelineTypes or TimeseriesAnomalyPredictionPipelineTypes], optional

        :param supporting_features_at_forecast: enables usage of future supporting feature values during forecast
        :type supporting_features_at_forecast: bool, optional

        :param cognito_transform_names: list of transformers to include in the feature enginnering computation process,
            see: AutoAI.Transformers
        :type cognito_transform_names: list[Transformers], optional

        :param csv_separator: the separator, or list of separators to try for separating columns in a CSV file,
            not used if the file_name is not a CSV file, default is ','
        :type csv_separator: list[str] or str, optional

        :param excel_sheet: name of the excel sheet to use, only applicable when xlsx file is an input,
            support for number of the sheet is deprecated, by default first sheet is used
        :type excel_sheet: str, optional

        :param encoding: encoding type for CSV training file
        :type encoding: str, optional

        :param positive_label: the positive class to report when binary classification, when multiclass or regression,
            this will be ignored
        :type positive_label: str, optional

        :param t_shirt_size: the size of the remote AutoAI POD instance (computing resources),
            only applicable to a remote scenario, see: AutoAI.TShirtSize
        :type t_shirt_size: TShirtSize, optional

        :param data_join_graph: a graph object with definition of join structure for multiple input data sources,
            data preprocess step for multiple files
        :type data_join_graph: DataJoinGraph, optional

        :param data_join_only: if `True` only preprocessing will be executed
        :type data_join_only: bool, optional

        :param drop_duplicates: if `True` duplicated rows in data will be removed before further processing
        :type drop_duplicates: bool, optional

        :param outliers_columns: replace outliers with NaN using IQR method for specified columns. By default,
            turned ON for regression learning_type and target column. To turn OFF empty list of columns must be passed
        :type outliers_columns: list, optional

        :param text_processing: if `True` text processing will be enabled, applicable only on Cloud
        :type text_processing: bool, optional

        :param word2vec_feature_number: number of features which will be generated from text column,
            will be applied only if `text_processing` is `True`, if `None` the default value will be taken
        :type word2vec_feature_number: int, optional

        :param daub_give_priority_to_runtime: the importance of run time over score for pipelines ranking,
            can take values between 0 and 5, if set to 0.0 only score is used,
            if set to 1 equally score and runtime are used, if set to value higher than 1
            the runtime gets higher importance over score
        :type daub_give_priority_to_runtime: float, optional

        :param fairness_info: dictionary that specifies metadata needed for measuring fairness,
            it contains three key values: `favorable_labels`, `unfavorable_labels` and `protected_attributes`,
            the `favorable_labels` attribute indicates that when the class column contains one of the value from list,
            that is considered a positive outcome, the `unfavorable_labels` is oposite to the `favorable_labels`
            and is obligatory for regression learning type, a protected attribute is a list of features that partition
            the population into groups whose outcome should have parity, if protected attribute is empty list
            then automatic detection of protected attributes will be run,
            if `fairness_info` is passed then fairness metric will be calculated
        :type fairness_info: fairness_info

        :param n_parallel_data_connections: number of maximum parallel connection to data source,
            supported only for IBM Cloud Pak® for Data 4.0.1 and above
        :type n_parallel_data_connections: int, optional

        :param categorical_imputation_strategy: missing values imputation strategy for categorical columns

            Possible values (only non-forecasting scenario):

            - ImputationStrategy.MEAN
            - ImputationStrategy.MEDIAN
            - ImputationStrategy.MOST_FREQUENT (default)

        :type categorical_imputation_strategy: ImputationStrategy, optional


        :param numerical_imputation_strategy: missing values imputation strategy for numerical columns

            Possible values (non-forecasting scenario):

            - ImputationStrategy.MEAN
            - ImputationStrategy.MEDIAN (default)
            - ImputationStrategy.MOST_FREQUENT

            Possible values (forecasting scenario):

            - ImputationStrategy.MEAN
            - ImputationStrategy.MEDIAN
            - ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS (default)
            - ImputationStrategy.VALUE
            - ImputationStrategy.FLATTEN_ITERATIVE
            - ImputationStrategy.LINEAR
            - ImputationStrategy.CUBIC
            - ImputationStrategy.PREVIOUS
            - ImputationStrategy.NEXT
            - ImputationStrategy.NO_IMPUTATION

        :param numerical_imputation_value: value for filling missing values if numerical_imputation_strategy
            is set to ImputationStrategy.VALUE, for forecasting only
        :type numerical_imputation_value: float, optional

        :param imputation_threshold: maximum threshold of missing values imputation, for forecasting only
        :type imputation_threshold: float, optional

        :param retrain_on_holdout: if True final pipelines will be train also on holdout data
        :type retrain_on_holdout: bool, optional

        :param categorical_columns: list of columns names that must be treated as categorical
        :type categorical_columns: list, optional

        :param numerical_columns: list of columns names that must be treated as numerical
        :type numerical_columns: list, optional

        :param sampling_type: type of sampling data for training, one of SamplingTypes enum values,
            default is SamplingTypes.FIRST_N_RECORDS, supported only for IBM Cloud Pak® for Data 4.0.1 and above
        :type sampling_type: str, optional

        :param sample_size_limit: the size of sample upper bound (in bytes). The default value is 1GB,
            supported only for IBM Cloud Pak® for Data 4.5 and above
        :type sample_size_limit: int, optional

        :param sample_rows_limit: the size of sample upper bound (in rows),
            supported only for IBM Cloud Pak® for Data 4.6 and above
        :type sample_rows_limit: int, optional

        :param sample_percentage_limit: the size of sample upper bound (as fraction of dataset size),
            supported only for IBM Cloud Pak® for Data 4.6 and above
        :type sample_percentage_limit: float, optional

        :param number_of_batch_rows: number of rows to read in each batch when reading from flight connection
        :type number_of_batch_rows: int, optional

        :param test_data_csv_separator: the separator, or list of separators to try for separating
            columns in a CSV user-defined holdout/test file, not used if the file_name is not a CSV file,
            default is ','
        :type test_data_csv_separator: list[str] or str, optional

        :param test_data_excel_sheet: name of the excel sheet to use for user-defined holdout/test data, 
            only use when xlsx file is an test, dataset file, by default first sheet is used
        :type test_data_excel_sheet: str or int, optional

        :param test_data_encoding: encoding type for CSV user-defined holdout/test file
        :type test_data_encoding: str, optional

        :param confidence_level: when the pipeline "PointwiseBoundedHoltWinters" or "PointwiseBoundedBATS" is used, 
            the prediction interval is calculated at a given confidence_level to decide if a data record 
            is an anomaly or not, optional for timeseries anomaly prediction
        :type confidence_level: float, optional

        :param incremental_learning: triggers incremental learning process for supported pipelines
        :type incremental_learning: bool, optional

        :param early_stop_enabled: enables early stop for incremental learning process
        :type early_stop_enabled: bool, optional

        :param early_stop_window_size: the number of iterations without score improvements before training stop
        :type early_stop_window_size: int, optional

        :return: RemoteAutoPipelines or LocalAutoPipelines, depends on how you initialize the AutoAI object
        :rtype: RemoteAutoPipelines or LocalAutoPipelines

        **Examples**

        .. code-block:: python

            from ibm_watson_machine_learning.experiment import AutoAI
            experiment = AutoAI(...)

            fairness_info = {
                       "protected_attributes": [
                           {"feature": "Sex", "reference_group": ['male'], "monitored_group": ['female']},
                           {"feature": "Age", "reference_group": [[50,60]], "monitored_group": [[18, 49]]}
                       ],
                       "favorable_labels": ["No Risk"],
                       "unfavorable_labels": ["Risk"],
                       }

            optimizer = experiment.optimizer(
                   name="name of the optimizer.",
                   prediction_type=AutoAI.PredictionType.BINARY,
                   prediction_column="y",
                   scoring=AutoAI.Metrics.ROC_AUC_SCORE,
                   desc="Some description.",
                   holdout_size=0.1,
                   max_num_daub_ensembles=1,
                   fairness_info= fairness_info,
                   cognito_transform_names=[AutoAI.Transformers.SUM,AutoAI.Transformers.MAX],
                   train_sample_rows_test_size=1,
                   include_only_estimators=[AutoAI.ClassificationAlgorithms.LGBM, AutoAI.ClassificationAlgorithms.XGB],
                   t_shirt_size=AutoAI.TShirtSize.L
               )

            optimizer = experiment.optimizer(
                   name="name of the optimizer.",
                   prediction_type=AutoAI.PredictionType.MULTICLASS,
                   prediction_column="y",
                   scoring=AutoAI.Metrics.ROC_AUC_SCORE,
                   desc="Some description.",
               )
        """
        # note: convert `timeseries` type to PredictionType.FORECASTING:
        if prediction_type == 'timeseries':
            prediction_type = PredictionType.FORECASTING


        if prediction_type != PredictionType.FORECASTING and retrain_on_holdout is None:
            retrain_on_holdout = True

        # Deprecation of excel_sheet as number:
        if isinstance(excel_sheet, int):
            warn(
                message="Support for excel sheet as number of the sheet (int) is deprecated! Please set excel sheet with name of the sheet.")

        if data_join_graph is not None:
            if self._workspace.wml_client.ICP_46 or self._workspace.wml_client.ICP_47:
                raise WMLClientError(Messages.get_message(message_id="obm_removal_message_cpd"))
            elif self._workspace.wml_client.ICP:
                print(Messages.get_message(message_id="obm_deprecation_message_cpd"))
            else:
                raise WMLClientError(Messages.get_message(message_id="obm_removal_message_cloud"))

        if prediction_type == PredictionType.FORECASTING and self._workspace.wml_client.ICP and \
                (self._workspace.wml_client.wml_credentials['version'].startswith('2.5') or \
                        self._workspace.wml_client.wml_credentials['version'].startswith('3.0') or \
                        self._workspace.wml_client.wml_credentials['version'].startswith('3.5')):
            raise TSNotSupported()

        if prediction_type == PredictionType.TIMESERIES_ANOMALY_PREDICTION and self._workspace.wml_client.ICP and \
                self._workspace.wml_client.wml_credentials['version'].startswith(('2.5', '3.0', '3.5', '4.0', '4.5', '4.6')):
            raise TSADNotSupported()

        if prediction_type in (PredictionType.FORECASTING, 'timeseries'):
            if not numerical_imputation_strategy and type(numerical_imputation_strategy) is not list:
                numerical_imputation_strategy = ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS
            elif not numerical_imputation_strategy and type(numerical_imputation_strategy) is list:
                numerical_imputation_strategy = ImputationStrategy.NO_IMPUTATION

            if prediction_column is not None or prediction_columns is None:
                raise ForecastPredictionColumnsMissing()
        elif prediction_type == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
            if feature_columns is None or prediction_column is not None or prediction_columns is not None:
                raise TimeseriesAnomalyPredictionFeatureColumnsMissing()
            if scoring is not None and scoring not in (
                    Metrics.F1_SCORE, Metrics.ROC_AUC_SCORE, Metrics.AVERAGE_PRECISION_SCORE, Metrics.PRECISION_SCORE,
                    Metrics.RECALL_SCORE):
                raise TimeseriesAnomalyPredictionUnsupportedMetric(scoring)
        else:
            if prediction_column is None or prediction_columns is not None:
                raise NonForecastPredictionColumnMissing(prediction_type)

        if test_size:
            print('Note: Using `test_size` is deprecated. Use `holdout_size` instead.')
            if not holdout_size:
                holdout_size = test_size
            test_size = None

        if daub_include_only_estimators:
            print('Note: Using `daub_include_only_estimators` is deprecated. Use `include_only_estimators` instead.')
            if not include_only_estimators:
                include_only_estimators = daub_include_only_estimators
            daub_include_only_estimators = None

        if train_sample_rows_test_size and (self._workspace.wml_client.ICP_46 or self._workspace.wml_client.ICP_47):
            print('Note: Using `train_sample_rows_test_size` is deprecated.'
                  'Use either `sample_rows_limit` or `sample_percentage_limit` instead.')
            if not sample_rows_limit and not sample_percentage_limit:
                if type(train_sample_rows_test_size) is float and train_sample_rows_test_size <= 1:
                    print('Value of `train_sample_rows_test_size` parameter'
                          'will be passed as `sample_percentage_limit`.')
                    sample_percentage_limit = train_sample_rows_test_size
                elif int(train_sample_rows_test_size) == train_sample_rows_test_size and train_sample_rows_test_size > 1:
                    print('Value of `train_sample_rows_test_size` parameter'
                          'will be passed as `sample_rows_limit`.')
                    sample_rows_limit = int(train_sample_rows_test_size)
                train_sample_rows_test_size = None
            elif sample_rows_limit or sample_percentage_limit:
                print('Parameter `train_sample_rows_test_size` will be ignored.')
                train_sample_rows_test_size = None

        def translate_str_imputation_param(x):
            if type(x) is list and prediction_type != PredictionType.FORECASTING:
                raise ImputationListNotSupported()

            if type(x) == str or (type(x) == list and type(x[0]) == str):
                return translate_imputation_string_strategy_to_enum(x, prediction_type)
            else:
                return x

        def translate_str_include_only_estimators_param(x):
            return [translate_estimator_string_to_enum(estimator) for estimator in x]

        def translate_str_include_batched_ensemble_estimators_param(x):
            return [translate_batched_estimator_string_to_enum(estimator) for estimator in x]

        def translate_str_pipeline_types_param(x):
            if prediction_type == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
                return [TimeseriesAnomalyPredictionPipelineTypes(pipeline_type) for pipeline_type in x]
            else:
                return [ForecastingPipelineTypes(pipeline_type) for pipeline_type in x]

        categorical_imputation_strategy = translate_str_imputation_param(categorical_imputation_strategy)
        numerical_imputation_strategy = translate_str_imputation_param(numerical_imputation_strategy)
        include_only_estimators = translate_str_include_only_estimators_param(include_only_estimators) if include_only_estimators else None
        include_batched_ensemble_estimators = translate_str_include_batched_ensemble_estimators_param(include_batched_ensemble_estimators) if include_batched_ensemble_estimators else None
        pipeline_types = translate_str_pipeline_types_param(pipeline_types) if pipeline_types != None else None

        if include_batched_ensemble_estimators:
            for batched_estimator in include_batched_ensemble_estimators:
                basic_estimator_str = batched_estimator.value.split("(")[1][:-1]
                basic_estimator = translate_estimator_string_to_enum(basic_estimator_str)
                if include_only_estimators is not None and basic_estimator not in include_only_estimators:
                    raise MissingEstimatorForExistingBatchedEstimator(batched_estimator, basic_estimator)

        validate_optimizer_enum_values(
            prediction_type=prediction_type,
            daub_include_only_estimators=daub_include_only_estimators,
            include_only_estimators=include_only_estimators,
            include_batched_ensemble_estimators=include_batched_ensemble_estimators,
            cognito_transform_names=cognito_transform_names,
            imputation_strategies=[x for y in list(filter(None, [categorical_imputation_strategy, numerical_imputation_strategy])) for x in (y if type(y) is list else [y])],
            scoring=scoring,
            t_shirt_size=kwargs.get("t_shirt_size", TShirtSize.M),
            is_cpd=self._workspace.wml_client.ICP
        )

        if daub_give_priority_to_runtime is not None:
            if daub_give_priority_to_runtime < 0.0 or daub_give_priority_to_runtime > 5.0:
                raise ParamOutOfRange('daub_give_priority_to_runtime', daub_give_priority_to_runtime, 0.0, 5.0)

        if data_join_graph:
            data_join_graph.problem_type = prediction_type
            data_join_graph.target_column = prediction_column

        if (prediction_type == PredictionType.BINARY and scoring in vars(PositiveLabelClass).values()
                and positive_label is None):
            raise MissingPositiveLabel(scoring, reason=f"\"{scoring}\" needs a \"positive_label\" "
                                                       f"parameter to be defined when used with binary classification.")

        if self._workspace is None and kwargs.get('t_shirt_size'):
            raise LocalInstanceButRemoteParameter(
                "t_shirt_size",
                reason="During LocalOptimizer initialization, \"t_shirt_size\" parameter was provided. "
                       "\"t_shirt_size\" parameter is only applicable to the RemoteOptimizer instance."
            )
        elif self._workspace is None:
            if prediction_type == PredictionType.FORECASTING:
                raise ForecastingCannotBeRunAsLocalScenario()
            if prediction_type == PredictionType.TIMESERIES_ANOMALY_PREDICTION:
                raise TimeseriesAnomalyPredictionCannotBeRunAsLocalScenario()

            reduced_kwargs = copy.copy(kwargs)

            for n in ['_force_local_scenario']:
                if n in reduced_kwargs:
                    del reduced_kwargs[n]

            validate_additional_params_for_optimizer(reduced_kwargs)

            return LocalAutoPipelines(
                name=name,
                prediction_type='classification' if prediction_type in ['binary', 'multiclass'] else prediction_type,
                prediction_column=prediction_column,
                scoring=scoring,
                desc=desc,
                holdout_size=holdout_size,
                max_num_daub_ensembles=max_number_of_estimators,
                train_sample_rows_test_size=train_sample_rows_test_size,
                include_only_estimators=include_only_estimators,
                include_batched_ensemble_estimators=include_batched_ensemble_estimators,
                cognito_transform_names=cognito_transform_names,
                positive_label=positive_label,
                _force_local_scenario=kwargs.get('_force_local_scenario', False),
                **reduced_kwargs
            )

        else:
            reduced_kwargs = copy.copy(kwargs)

            for n in ['t_shirt_size', 'notebooks', 'autoai_pod_version', 'obm_pod_version']:
                if n in reduced_kwargs:
                    del reduced_kwargs[n]

            validate_additional_params_for_optimizer(reduced_kwargs)

            engine = WMLEngine(self._workspace)

            if self._20_class_limit_removal_test:
                engine._20_class_limit_removal_test = True

            optimizer = RemoteAutoPipelines(
                name=name,
                prediction_type=prediction_type,
                prediction_column=prediction_column,
                prediction_columns=prediction_columns,
                timestamp_column_name=timestamp_column_name,
                scoring=scoring,
                desc=desc,
                holdout_size=holdout_size,
                max_num_daub_ensembles=max_number_of_estimators,
                t_shirt_size=self._workspace.restrict_pod_size(t_shirt_size=kwargs.get(
                    't_shirt_size', TShirtSize.M if self._workspace.wml_client.ICP else TShirtSize.L)
                ),
                train_sample_rows_test_size=train_sample_rows_test_size,
                include_only_estimators=include_only_estimators,
                include_batched_ensemble_estimators=include_batched_ensemble_estimators,
                backtest_num=backtest_num,
                lookback_window=lookback_window,
                forecast_window=forecast_window,
                backtest_gap_length=backtest_gap_length,
                cognito_transform_names=cognito_transform_names,
                data_join_graph=data_join_graph,
                drop_duplicates=drop_duplicates,
                outliers_columns=outliers_columns,
                text_processing=text_processing,
                word2vec_feature_number=word2vec_feature_number,
                csv_separator=csv_separator,
                excel_sheet=excel_sheet,
                encoding=encoding,
                positive_label=positive_label,
                data_join_only=data_join_only,
                engine=engine,
                daub_give_priority_to_runtime=daub_give_priority_to_runtime,
                notebooks=kwargs.get('notebooks', True),
                autoai_pod_version=kwargs.get('autoai_pod_version', None),
                obm_pod_version=kwargs.get('obm_pod_version', None),
                fairness_info=fairness_info,
                sampling_type=sampling_type,
                sample_size_limit=sample_size_limit,
                sample_rows_limit=sample_rows_limit,
                sample_percentage_limit=sample_percentage_limit,
                number_of_batch_rows=number_of_batch_rows,
                n_parallel_data_connections=n_parallel_data_connections,
                categorical_imputation_strategy=categorical_imputation_strategy,
                numerical_imputation_strategy=numerical_imputation_strategy,
                numerical_imputation_value=numerical_imputation_value,
                imputation_threshold=imputation_threshold,
                retrain_on_holdout=retrain_on_holdout,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                feature_columns=feature_columns,
                pipeline_types=pipeline_types,
                supporting_features_at_forecast=supporting_features_at_forecast,
                test_data_csv_separator=test_data_csv_separator,
                test_data_excel_sheet=test_data_excel_sheet,
                test_data_encoding=test_data_encoding,
                confidence_level=confidence_level,
                incremental_learning=incremental_learning,
                early_stop_enabled=early_stop_enabled,
                early_stop_window_size=early_stop_window_size,
                **reduced_kwargs
            )
            optimizer._workspace = self._workspace
            return optimizer

    def _init_estimator_enums(self):
        if self._workspace and self._workspace.wml_client.ICP:
            self.ClassificationAlgorithms = ClassificationAlgorithmsCP4D
            self.RegressionAlgorithms = RegressionAlgorithmsCP4D
            self.ForecastingAlgorithms = ForecastingAlgorithmsCP4D
        else:
            self.ClassificationAlgorithms = ClassificationAlgorithms
            self.RegressionAlgorithms = RegressionAlgorithms
            self.ForecastingAlgorithms = ForecastingAlgorithms

    def _block_autoai_on_git_based_project(self):
        """Raises ForbiddenActionForGitBasedProject error for AutoAI experiments on git based project.
        It can be disabled by setting environment variable ENABLE_AUTOAI to 'true'
        """
        from os import environ

        if self._workspace:
            if getattr(self._workspace.wml_client, 'project_type', None) == 'local_git_storage' \
                    and environ.get('ENABLE_AUTOAI', 'false').lower() == 'false':
                raise ForbiddenActionForGitBasedProject(reason="Creating AutoAI experiment is not supported for git based project.")


