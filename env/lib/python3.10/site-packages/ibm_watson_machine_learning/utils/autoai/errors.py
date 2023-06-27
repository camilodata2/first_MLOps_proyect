#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

__all__ = [
    "MissingPipeline",
    "FitNotCompleted",
    "FitNeeded",
    "AutoAIComputeError",
    "MissingAutoPipelinesParameters",
    "UseWMLClient",
    "PipelineNotLoaded",
    "MissingCOSStudioConnection",
    "MissingProjectLib",
    "LocalInstanceButRemoteParameter",
    "DataFormatNotSupported",
    "HoldoutSplitNotSupported",
    'LibraryNotCompatible',
    'CannotInstallLibrary',
    'InvalidCOSCredentials',
    'CannotDownloadTrainingDetails',
    'TShirtSizeNotSupported',
    'MissingPositiveLabel',
    'MissingDataPreprocessingStep',
    'CannotDownloadWMLPipelineDetails',
    'WrongDataJoinGraphNodeName',
    'NotInWatsonStudio',
    'WrongWMLServer',
    'CredentialsNotFound',
    'SetIDFailed',
    'MissingLocalAsset',
    'DataSourceSizeNotSupported',
    'TrainingDataSourceIsNotFile',
    'VisualizationFailed',
    'InvalidPredictionType',
    'InvalidIdType',
    'NoneDataConnection',
    'CannotReadSavedRemoteDataBeforeFit',
    'NotWSDEnvironment',
    'NotExistingCOSResource',
    'AdditionalParameterIsUnexpected',
    'OBMForNFSIsNotSupported',
    'ForecastPredictionColumnsMissing',
    'TimeseriesAnomalyPredictionFeatureColumnsMissing',
    'TimeseriesAnomalyPredictionUnsupportedMetric',
    'NonForecastPredictionColumnMissing',
    'ForecastingCannotBeRunAsLocalScenario',
    'TimeseriesAnomalyPredictionCannotBeRunAsLocalScenario',
    'ForecastingUnsupportedOperation',
    'OBMMainNodeAlreadySet',
    'InvalidSequenceValue',
    'NoAvailableMetrics',
    'WrongAssetType',
    'TSNotSupported',
    'TSADNotSupported',
    'InvalidDataAsset',
    'ParamOutOfRange',
    'CannotConnectToWebsocket',
    'TestDataNotPresent',
    'NoAutomatedHoldoutSplit',
    'DiscardedModel',
    'WrongModelName',
    'StrategyIsNotApplicable',
    'NumericalImputationStrategyValueMisused',
    'InvalidImputationParameterNonTS',
    'InvalidImputationParameterTS',
    'InconsistentImputationListElements',
    'ImputationListNotSupported',
    'MissingEstimatorForExistingBatchedEstimator',
    'FutureExogenousFeaturesNotSupported',
    'NoAvailableNotebookLocation',
    "ContainerTypeNotSupported",
    "InvalidSizeLimit",
    "CorruptedData"
]


from ibm_watson_machine_learning.utils import WMLClientError
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, ImputationStrategy


class MissingPipeline(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"There is no such a Pipeline like: {value_name}", reason)


class FitNotCompleted(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Fit run is not completed or the status is failed for run: {value_name}", reason)


class FitNeeded(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Fit run was not performed.", reason)


class AutoAIComputeError(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Fit run failed for run_id: {value_name}.", reason)


class MissingAutoPipelinesParameters(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"AutoPipelines parameters are {value_name}", reason)


class UseWMLClient(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Use WML v4 Client instead of {value_name}", reason)


class PipelineNotLoaded(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Pipeline model: {value_name} cannot load.", reason)


class MissingCOSStudioConnection(WMLClientError, ValueError):
    def __init__(self, reason=None):
        WMLClientError.__init__(self, f"Missing COS Studio connection.", reason)


class MissingProjectLib(WMLClientError, ValueError):
    def __init__(self, reason=None):
        WMLClientError.__init__(self,
                                f"project-lib package missing in the environment, please make sure you are on Watson Studio"
                                f" and want to automatically initialize your COS connection."
                                f" If you want to initialize COS connection manually, do not use from_studio() method.",
                                reason)


class LocalInstanceButRemoteParameter(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Provided {value_name} parameter to local optimizer instance.", reason)


class DataFormatNotSupported(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self,
                                f"This data format is not supported by SDK. (CSV and XLSX formats are supported.)",
                                reason)


class HoldoutSplitNotSupported(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Holdout split is not supported for xlsx data.", reason)


class LibraryNotCompatible(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Library not compatible or missing!", reason)


class CannotInstallLibrary(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Library cannot be installed! Error: {value_name}", reason)


class InvalidCOSCredentials(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Wrong COS credentials!", reason)


class CannotDownloadTrainingDetails(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Cannot download training details, training is not done yet. "
                                      f"Please try again after training is finished.", reason)


class TShirtSizeNotSupported(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"This t-shirt size: \"{value_name}\" is not supported on this environment.",
                                reason)


class MissingPositiveLabel(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Missing positive label for \"{value_name}\"", reason)


class MissingDataPreprocessingStep(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Data preprocessing step not performed.", reason)


class CannotDownloadWMLPipelineDetails(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Cannot download WML pipeline details ", reason)


class WrongDataJoinGraphNodeName(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"{value_name} is not a correct node name.", reason)


class NotInWatsonStudio(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Wrong environment.", reason)


class WrongWMLServer(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Wrong WML Server instance.", reason)


class CredentialsNotFound(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Cannot find WMLS credentials in WSD.", reason)


class SetIDFailed(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Cannot set {value_name}.", reason)


class MissingLocalAsset(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Local asset: {value_name} cannot be found.", reason)


class DataSourceSizeNotSupported(WMLClientError, ValueError):
    def __init__(self, reason=None):
        WMLClientError.__init__(self, f"The selected data source is too large for selected compute configuration "
                                      f"and might fail to run. Consider increasing the compute configuration", reason)


class TrainingDataSourceIsNotFile(WMLClientError, ValueError):
    def __init__(self, data_location=None, reason=None):
        WMLClientError.__init__(self, f"Training data location: {data_location} is a directory or does not exist."
                                      f"Please set training data location to dataset file location.", reason)


class VisualizationFailed(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Cannot perform visualization.", reason)


class InvalidPredictionType(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, f"Cannot recognize prediction type: {value_name}", reason)


class InvalidIdType(WMLClientError, ValueError):
    def __init__(self, typ):
        WMLClientError.__init__(self, f"Unexpected type of id: {typ}")


class NoneDataConnection(WMLClientError, ValueError):
    def __init__(self, resource_name):
        WMLClientError.__init__(self, f"Invalid DataConnection in {resource_name} list. DataConnection cannot be None")


class CannotReadSavedRemoteDataBeforeFit(WMLClientError, ValueError):
    def __init__(self):
        WMLClientError.__init__(self, f"Cannot read saved remote data before fit.")


class NotWSDEnvironment(WMLClientError, ValueError):
    def __init__(self):
        WMLClientError.__init__(self, f"Function cannot be used outside WSD environment.")


class NotExistingCOSResource(WMLClientError, ValueError):
    def __init__(self, bucket, path):
        WMLClientError.__init__(self, f"There is no COS resource: {path} in bucket: {bucket}.")


class AdditionalParameterIsUnexpected(WMLClientError):
    def __init__(self, param):
        WMLClientError.__init__(self, f"Additional parameter is not expected: {param}")


class OBMForNFSIsNotSupported(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, f"OBM for NFS is not supported.")


class ForecastPredictionColumnsMissing(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self,
                                "For prediction_type = `forecasting` passing `prediction_columns` is required, while `prediction_column` should be None.")


class TimeseriesAnomalyPredictionFeatureColumnsMissing(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self,
                                "For prediction_type = `timeseries_anomaly_prediction` passing `feature_columns` is required, while `prediction_columns` and `prediction_column` should be None.")


class TimeseriesAnomalyPredictionUnsupportedMetric(WMLClientError):
    def __init__(self, metric):
        WMLClientError.__init__(self, f"Metric {metric} is unsupported for timeseries anomaly prediction type.")


class NonForecastPredictionColumnMissing(WMLClientError):
    def __init__(self, prediction_type):
        WMLClientError.__init__(self,
                                f"For prediction_type = `{prediction_type}` passing `prediction_column` is required, while `prediction_columns` should be None.")


class ForecastingCannotBeRunAsLocalScenario(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, f"Forecasting cannot be run in local scenario.")


class TimeseriesAnomalyPredictionCannotBeRunAsLocalScenario(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, f"Timeseries anomaly prediction cannot be run in local scenario.")


class ForecastingUnsupportedOperation(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, f"Operation is unsupported for timeseries forecasting prediction type.")


class OBMMainNodeAlreadySet(WMLClientError):
    def __init__(self, param):
        WMLClientError.__init__(self, f"OBM Main node is already set to: {param}. If you want to set the new one,"
                                      f"please reinitialize entire object.")


class InvalidSequenceValue(WMLClientError, ValueError):
    def __init__(self, el, correct_values):
        WMLClientError.__init__(self, f"Invalid sequence element: '{el}' sequence must be composed from "
                                      f"given values: {correct_values}")


class NoAvailableMetrics(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, f"Currently there is no available metrics.")


class WrongAssetType(WMLClientError):
    def __init__(self, asset_type: str):
        WMLClientError.__init__(self, f"This asset type: '{asset_type}' is not supported.")


class TSNotSupported(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, f"Time series forecasting is not supported for CPD 2.5, 3.0 and 3.5.")


class TSADNotSupported(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self,
                                f"Time series anomaly prediction is not supported for CPD 2.5, 3.0, 3.5, 4.0, 4.5 and 4.6.")


class InvalidDataAsset(WMLClientError, ValueError):
    def __init__(self, reason='Wrong Data Asset'):
        WMLClientError.__init__(self, "Data Asset without 'connection' is not supported for result store action.",
                                reason)


class ParamOutOfRange(WMLClientError, ValueError):
    def __init__(self, param_name, value, min, max):
        WMLClientError.__init__(self,
                                f"Value of parameter `{param_name}`, {value}, is out of expected range - between {min} and {max}.")


class CannotConnectToWebsocket(WMLClientError, ValueError):
    def __init__(self, n):
        WMLClientError.__init__(self, f"{n} times connecting to websocket failed.")


class TestDataNotPresent(WMLClientError, ValueError):
    def __init__(self, reason):
        WMLClientError.__init__(self, "User defined (test / holdout) data is not present for this AutoAI experiment.",
                                reason)


class NoAutomatedHoldoutSplit(WMLClientError, ValueError):
    def __init__(self, reason):
        WMLClientError.__init__(self, "AutoAI experiment was performed with user defined holdout."
                                      "To recreate holdout split, please use test data connection."
                                      "You can call optimizer.get_test_data_connections().",
                                reason)


class DiscardedModel(WMLClientError, ValueError):
    def __init__(self, model_name):
        WMLClientError.__init__(self,
                                "You are trying to store a discarded forecasting model pipeline. Please use not discarded one."
                                "Look at the pipelines 'summary' method and choose the 'Winner' one.",
                                reason=f"Pipeline: '{model_name}' is discarded!")


class WrongModelName(WMLClientError, ValueError):
    def __init__(self, model_name):
        WMLClientError.__init__(self,
                                "This model name does not exist. Please provide correct one."
                                "Look at the pipelines 'summary' method and choose the 'Winner' one.",
                                reason=f"Pipeline: '{model_name}' does not exist!")


class StrategyIsNotApplicable(WMLClientError, ValueError):
    def __init__(self, strategy, prediction_type, valid_strategies):
        WMLClientError.__init__(self,
                                f'{strategy} is not valid for '
                                f'{"forecasting" if prediction_type == PredictionType.FORECASTING else "non-forecasting"} '
                                f'scenario. Valid strategies are: {", ".join(valid_strategies)}')


class NumericalImputationStrategyValueMisused(WMLClientError, ValueError):
    def __init__(self):
        WMLClientError.__init__(self, 'Parameter `numerical_imputation_value` can only be used when '
                                      '`numerical_imputation_strategy` is set to ImputationStrategy.VALUE.')


class InvalidImputationParameterNonTS(WMLClientError, ValueError):
    def __init__(self):
        WMLClientError.__init__(self, '`numerical_imputation_value` and `imputation_threshold` can be set only for forecasting experiments.')


class InvalidImputationParameterTS(WMLClientError, ValueError):
    def __init__(self):
        WMLClientError.__init__(self, '`categorical_imputation_strategy` does not apply to a forecasting experiment.')


class InconsistentImputationListElements(WMLClientError, ValueError):
    def __init__(self, strategies):
        l = ['ImputationStrategy.' + s.name for s in strategies]
        WMLClientError.__init__(self, f'`categorical_imputation_strategy` list elements are not compatible: {", ".join(l)}')


class ImputationListNotSupported(WMLClientError, ValueError):
    def __init__(self):
        WMLClientError.__init__(self, f'List passed as imputation strategy is not supported for non-forecasting scenarios.')


class InvalidSamplingType(WMLClientError, ValueError):
    def __init__(self, sampling_type: str, predcition_type: str):
        WMLClientError.__init__(self, f"Sampling type for {sampling_type} is invalid for prediction type {predcition_type}.")


class MissingEstimatorForExistingBatchedEstimator(WMLClientError, ValueError):
    def __init__(self, batched_estimator: str, missing_estimator: str):
        WMLClientError.__init__(self, f"There is no corresponding estimator in `include_only_estimators` list for " +
                                f"{batched_estimator} estimator in `include_batched_ensemble_estimators` list. " +
                                f"Add {missing_estimator} estimator to `include_only_estimators` list.")


class FutureExogenousFeaturesNotSupported(WMLClientError, ValueError):
    def __init__(self, prediction_type: str):
        WMLClientError.__init__(self, f"Future Exogenous features are only supported for forecasting."
                                      f"Current prediction type is: {prediction_type}")


class NoAvailableNotebookLocation(WMLClientError, ValueError):
    def __init__(self, pipeline_name):
        WMLClientError.__init__(self, f"Unable to find notebook location in training details for {pipeline_name}. " +
                                "Possible causes: pipeline failure or pipeline not classified as winner in " +
                                "forecasting scenario.")


class ContainerTypeNotSupported(WMLClientError, ValueError):
    def __init__(self):
        WMLClientError.__init__(self, f"The container data connection type is not supported for CP4D environment."
                                      f" Supported types are: data asset and connection asset. ")


class InvalidSizeLimit(WMLClientError, ValueError):
    def __init__(self, size_limit, max_size_limit):
        WMLClientError.__init__(self,
                                f"The value of sampling size limit (total_size_limit): {size_limit} is incorrect. "
                                f"The maximum value allowed for chosen environment configuration is: {max_size_limit}. "
                                f"Adjust the sampling size to the environment limitations or "
                                f"consider a change the compute resources allocated for running the experiment.")


class CorruptedData(WMLClientError, ValueError):
    def __init__(self, reason='Corrupted Data'):
        WMLClientError.__init__(self,
                                "Cannot read corrupted data. "
                                "Check the data for unexpected characters (e.g. non-printing, control character). "
                                "Clean up or use other data source and try again.",
                                reason)