#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from enum import Enum

__all__ = [
    "ClassificationAlgorithms",
    "ClassificationAlgorithmsCP4D",
    "RegressionAlgorithms",
    "RegressionAlgorithmsCP4D",
    "ForecastingAlgorithms",
    "ForecastingAlgorithmsCP4D",
    "PredictionType",
    "Metrics",
    "Transformers",
    "DataConnectionTypes",
    "RunStateTypes",
    "PipelineTypes",
    "Directions",
    "TShirtSize",
    "MetricsToDirections",
    'PositiveLabelClass',
    'VisualizationTypes',
    'SamplingTypes',
    'ImputationStrategy',
    'ForecastingPipelineTypes',
    'TimeseriesAnomalyPredictionPipelineTypes',
    'TimeseriesAnomalyPredictionAlgorithms'
]


class ClassificationAlgorithms(Enum):
    """Classification algorithms that AutoAI could use for IBM Cloud."""
    EX_TREES = "ExtraTreesClassifier"
    GB = "GradientBoostingClassifier"
    LGBM = "LGBMClassifier"
    LR = "LogisticRegression"
    RF = "RandomForestClassifier"
    XGB = "XGBClassifier"
    DT = "DecisionTreeClassifier"
    SnapDT = "SnapDecisionTreeClassifier"
    SnapRF = "SnapRandomForestClassifier"
    SnapSVM = "SnapSVMClassifier"
    SnapLR = "SnapLogisticRegression"
    SnapBM = "SnapBoostingMachineClassifier"


class ClassificationAlgorithmsCP4D(Enum):
    """
    Classification algorithms that AutoAI could use for IBM Cloud Pak® for Data(CP4D).
    The SnapML estimators (SnapDT, SnapRF, SnapSVM, SnapLR) are supported
    on IBM Cloud Pak® for Data version 4.0.2 and above.
    """
    EX_TREES = "ExtraTreesClassifierEstimator"
    GB = "GradientBoostingClassifierEstimator"
    LGBM = "LGBMClassifierEstimator"
    LR = "LogisticRegressionEstimator"
    RF = "RandomForestClassifierEstimator"
    XGB = "XGBClassifierEstimator"
    DT = "DecisionTreeClassifierEstimator"
    SnapDT = "SnapDecisionTreeClassifier"
    SnapRF = "SnapRandomForestClassifier"
    SnapSVM = "SnapSVMClassifier"
    SnapLR = "SnapLogisticRegression"
    SnapBM = "SnapBoostingMachineClassifier"


class BatchedClassificationAlgorithms(Enum):
    """Batched tree ensemble classification algorithms that AutoAI could use for IBM Cloud."""
    RF = "BatchedTreeEnsembleClassifier(RandomForestClassifier)"
    EX_TREES = "BatchedTreeEnsembleClassifier(ExtraTreesClassifier)"
    LGBM = "BatchedTreeEnsembleClassifier(LGBMClassifier)"
    XGB = "BatchedTreeEnsembleClassifier(XGBClassifier)"
    SnapRF = "BatchedTreeEnsembleClassifier(SnapRandomForestClassifier)"
    SnapBM = "BatchedTreeEnsembleClassifier(SnapBoostingMachineClassifier)"


class RegressionAlgorithms(Enum):
    """Regression algorithms that AutoAI could use for IBM Cloud."""
    RF = "RandomForestRegressor"
    RIDGE = "Ridge"
    EX_TREES = "ExtraTreesRegressor"
    GB = "GradientBoostingRegressor"
    LR = "LinearRegression"
    XGB = "XGBRegressor"
    LGBM = "LGBMRegressor"
    DT = "DecisionTreeRegressor"
    SnapDT = "SnapDecisionTreeRegressor"
    SnapRF = "SnapRandomForestRegressor"
    SnapBM = "SnapBoostingMachineRegressor"


class RegressionAlgorithmsCP4D(Enum):
    """
    Regression algorithms that AutoAI could use for IBM Cloud Pak® for Data(CP4D).
    The SnapML estimators (SnapDT, SnapRF, SnapBM) are supported
    on IBM Cloud Pak® for Data version 4.0.2 and above.
    """
    RF = "RandomForestRegressorEstimator"
    RIDGE = "RidgeEstimator"
    EX_TREES = "ExtraTreesRegressorEstimator"
    GB = "GradientBoostingRegressorEstimator"
    LR = "LinearRegressionEstimator"
    XGB = "XGBRegressorEstimator"
    LGBM = "LGBMRegressorEstimator"
    DT = "DecisionTreeRegressorEstimator"
    SnapDT = "SnapDecisionTreeRegressor"
    SnapRF = "SnapRandomForestRegressor"
    SnapBM = "SnapBoostingMachineRegressor"


class BatchedRegressionAlgorithms(Enum):
    """Batched tree ensemble regression algorithms that AutoAI could use for IBM Cloud."""
    RF = "BatchedTreeEnsembleRegressor(RandomForestRegressor)"
    EX_TREES = "BatchedTreeEnsembleRegressor(ExtraTreesRegressor)"
    LGBM = "BatchedTreeEnsembleRegressor(LGBMRegressor)"
    SnapRF = "BatchedTreeEnsembleRegressor(SnapRandomForestRegressor)"
    SnapBM = "BatchedTreeEnsembleRegressor(SnapBoostingMachineRegressor)"
    XGB = "BatchedTreeEnsembleRegressor(XGBRegressor)"


class ForecastingAlgorithmsCP4D(Enum):
    """Forecasting algorithms that AutoAI could use for IBM Cloud."""
    LR = "LinearRegression"
    ENSEMBLER = "Ensembler"
    ARIMA = "ARIMA"
    HW = "HoltWinters"
    BATS = "BATS"
    RF = "RandomForest"
    SVM = "SVM"


class ForecastingAlgorithms(Enum):
    """Forecasting algorithms that AutoAI could use for IBM Cloud Pak® for Data(CP4D)."""
    LR = "LinearRegression"
    ENSEMBLER = "Ensembler"
    ARIMA = "ARIMA"
    HW = "HoltWinters"
    BATS = "BATS"
    RF = "RandomForest"
    SVM = "SVM"


class ForecastingPipelineTypes(Enum):
    """Forecasting pipeline types that AutoAI could use for IBM Cloud Pak® for Data(CP4D)."""
    RandomForestRegressor = 'RandomForestRegressor'
    ExogenousRandomForestRegressor = 'ExogenousRandomForestRegressor'
    SVM = 'SVM'
    ExogenousSVM = 'ExogenousSVM'
    LocalizedFlattenEnsembler = 'LocalizedFlattenEnsembler'
    DifferenceFlattenEnsembler = 'DifferenceFlattenEnsembler'
    FlattenEnsembler = 'FlattenEnsembler'
    ExogenousLocalizedFlattenEnsembler = 'ExogenousLocalizedFlattenEnsembler'
    ExogenousDifferenceFlattenEnsembler = 'ExogenousDifferenceFlattenEnsembler'
    ExogenousFlattenEnsembler = 'ExogenousFlattenEnsembler'
    MT2RForecaster = 'MT2RForecaster'
    ExogenousMT2RForecaster = 'ExogenousMT2RForecaster'
    HoltWinterAdditive = 'HoltWinterAdditive'
    HoltWinterMultiplicative = 'HoltWinterMultiplicative'
    Bats = 'Bats'
    ARIMA = 'ARIMA'
    ARIMAX = 'ARIMAX'
    ARIMAX_RSAR = 'ARIMAX_RSAR'
    ARIMAX_PALR = 'ARIMAX_PALR'
    ARIMAX_RAR = 'ARIMAX_RAR'
    ARIMAX_DMLR = 'ARIMAX_DMLR'

    @staticmethod
    def get_exogenous():
        """Get list of pipelines that use supporting features (exogenous pipelines).

        :return: list of pipelines using supporting features
        :rtype: list[ForecastingPipelineTypes]
        """
        return [pipeline for pipeline in ForecastingPipelineTypes if pipeline.value.startswith('Exogenous')]

    @staticmethod
    def get_non_exogenous():
        """Get list of pipelines not using supporting features (non-exogenous pipelines).

        :return: list of pipelines that do not use supporting features
        :rtype: list[ForecastingPipelineTypes]
        """
        return [pipeline for pipeline in ForecastingPipelineTypes if not pipeline.value.startswith('Exogenous')]


class TimeseriesAnomalyPredictionAlgorithms(Enum):
    """Timeseries Anomaly Prediction algorithms that AutoAI could use for IBM Cloud."""
    Forecasting = "Forecasting"
    Window = "Window"
    Relationship = "Relationship"


class TimeseriesAnomalyPredictionPipelineTypes(Enum):
    """Timeseries Anomaly Prediction pipeline types that AutoAI could use for IBM Cloud."""
    PointwiseBoundedHoltWintersAdditive = 'PointwiseBoundedHoltWintersAdditive'
    PointwiseBoundedBATS = 'PointwiseBoundedBATS'
    PointwiseBoundedBATSForceUpdate = 'PointwiseBoundedBATSForceUpdate'
    WindowNN = 'WindowNN'
    WindowPCA = 'WindowPCA'
    WindowLOF = 'WindowLOF'


class PredictionType:
    """Supported types of learning."""
    CLASSIFICATION = 'classification'
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    TIMESERIES_ANOMALY_PREDICTION = "timeseries_anomaly_prediction"


class PositiveLabelClass:
    """Metrics that need positive label definition for binary classification."""
    AVERAGE_PRECISION_SCORE = "average_precision"
    F1_SCORE = "f1"
    PRECISION_SCORE = "precision"
    RECALL_SCORE = "recall"
    F1_SCORE_MICRO = "f1_micro"
    F1_SCORE_MACRO = "f1_macro"
    F1_SCORE_WEIGHTED = "f1_weighted"
    PRECISION_SCORE_MICRO = "precision_micro"
    PRECISION_SCORE_MACRO = "precision_macro"
    PRECISION_SCORE_WEIGHTED = "precision_weighted"
    RECALL_SCORE_MICRO = "recall_micro"
    RECALL_SCORE_MACRO = "recall_macro"
    RECALL_SCORE_WEIGHTED = "recall_weighted"


class Metrics:
    """Supported types of classification and regression metrics in autoai."""
    ACCURACY_SCORE = "accuracy"
    ACCURACY_AND_DISPARATE_IMPACT_SCORE = "accuracy_and_disparate_impact"
    AVERAGE_PRECISION_SCORE = "average_precision"
    F1_SCORE = "f1"
    LOG_LOSS = "neg_log_loss"
    PRECISION_SCORE = "precision"
    RECALL_SCORE = "recall"
    ROC_AUC_SCORE = "roc_auc"

    F1_SCORE_MICRO = "f1_micro"
    F1_SCORE_MACRO = "f1_macro"
    F1_SCORE_WEIGHTED = "f1_weighted"
    PRECISION_SCORE_MICRO = "precision_micro"
    PRECISION_SCORE_MACRO = "precision_macro"
    PRECISION_SCORE_WEIGHTED = "precision_weighted"
    RECALL_SCORE_MICRO = "recall_micro"
    RECALL_SCORE_MACRO = "recall_macro"
    RECALL_SCORE_WEIGHTED = "recall_weighted"

    EXPLAINED_VARIANCE_SCORE = "explained_variance"
    MEAN_ABSOLUTE_ERROR = "neg_mean_absolute_error"
    MEAN_SQUARED_ERROR = "neg_mean_squared_error"
    MEAN_SQUARED_LOG_ERROR = "neg_mean_squared_log_error"
    MEDIAN_ABSOLUTE_ERROR = "neg_median_absolute_error"
    ROOT_MEAN_SQUARED_ERROR = "neg_root_mean_squared_error"
    ROOT_MEAN_SQUARED_LOG_ERROR = "neg_root_mean_squared_log_error"
    R2_SCORE = "r2"
    R2_AND_DISPARATE_IMPACT_SCORE = "r2_and_disparate_impact"


class Transformers:
    """Supported types of congito transformers names in autoai."""
    SQRT = "sqrt"
    LOG = "log"
    ROUND = "round"
    SQUARE = "square"
    CBRT = "cbrt"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"

    ABS = "abs"
    SIGMOID = "sigmoid"
    PRODUCT = "product"
    MAX = "max"
    DIFF = "diff"
    SUM = "sum"
    DIVIDE = "divide"
    STDSCALER = "stdscaler"

    MINMAXSCALER = "minmaxscaler"
    PCA = "pca"
    NXOR = "nxor"
    CUBE = "cube"
    FEATUREAGGLOMERATION = "featureagglomeration"
    ISOFORESTANOMALY = "isoforestanomaly"


class DataConnectionTypes:
    """Supported types of DataConnection."""
    S3 = "s3"
    FS = 'fs'
    DS = 'data_asset'
    CA = 'connection_asset'
    CN = 'container'


class RunStateTypes:
    """Supported types of AutoAI fit/run."""
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineTypes:
    """Supported types of Pipelines."""
    LALE = "lale"
    SKLEARN = "sklearn"


class Directions:
    """Possible metrics directions"""
    ASCENDING = "ascending"
    DESCENDING = "descending"


class TShirtSize:
    """Possible sizes of the AutoAI POD.
    Depends on the POD size, AutoAI could support different data sets sizes.

    - S - small (2vCPUs and 8GB of RAM)
    - M - Medium (4vCPUs and 16GB of RAM)
    - L - Large (8vCPUs and 32GB of RAM))
    - XL - Extra Large (16vCPUs and 64GB of RAM)
    """
    S = 's'
    M = 'm'
    L = 'l'
    XL = 'xl'


class MetricsToDirections(Enum):
    """Map of metrics directions."""
    ROC_AUC = Directions.ASCENDING
    NORMALIZED_GINI_COEFFICIENT = Directions.ASCENDING
    PRECISION = Directions.ASCENDING
    AVERAGE_PRECISION = Directions.ASCENDING
    NEG_LOG_LOSS = Directions.DESCENDING
    RECALL = Directions.ASCENDING
    ACCURACY = Directions.ASCENDING
    F1 = Directions.ASCENDING

    PRECISION_MICRO = Directions.ASCENDING
    PRECISION_MACRO = Directions.ASCENDING
    PRECISION_WEIGHTED = Directions.ASCENDING
    F1_MICRO = Directions.ASCENDING
    F1_MACRO = Directions.ASCENDING
    F1_WEIGHTED = Directions.ASCENDING
    RECALL_MICRO = Directions.ASCENDING
    RECALL_MACRO = Directions.ASCENDING
    RECALL_WEIGHTED = Directions.ASCENDING

    NEG_ROOT_MEAN_SQUARED_ERROR = Directions.DESCENDING
    EXPLAINED_VARIANCE = Directions.ASCENDING
    NEG_MEAN_ABSOLUTE_ERROR = Directions.DESCENDING
    NEG_MEAN_SQUARED_ERROR = Directions.DESCENDING
    NEG_MEAN_SQUARED_LOG_ERROR = Directions.DESCENDING
    NEG_MEDIAN_ABSOLUTE_ERROR = Directions.DESCENDING
    NEG_ROOT_MEAN_SQUARED_LOG_ERROR = Directions.DESCENDING
    R2 = Directions.ASCENDING


class VisualizationTypes:
    """Types of visualization options."""
    PDF = 'pdf'
    INPLACE = 'inplace'


class SamplingTypes:
    """Types of training data sampling"""
    FIRST_VALUES = 'first_n_records'
    RANDOM = 'random'
    STRATIFIED = 'stratified'
    LAST_VALUES = 'truncate'


class ImputationStrategy(Enum):
    """Missing values imputation strategies."""
    MEAN = 'mean'
    MEDIAN = 'median'
    MOST_FREQUENT = 'most_frequent'
    BEST_OF_DEFAULT_IMPUTERS = 'best_of_default_imputers'
    VALUE = 'value'
    FLATTEN_ITERATIVE = 'flatten_iterative'
    LINEAR = 'linear'
    CUBIC = 'cubic'
    PREVIOUS = 'previous'
    NEXT = 'next'
    NO_IMPUTATION = 'no_imputation'
