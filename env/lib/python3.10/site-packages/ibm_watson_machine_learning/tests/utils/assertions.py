#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import traceback

from lale.operators import BasePipeline

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType

test_case = unittest.TestCase()


def is_lale_pipeline_type(pipeline: object) -> None:
    print("Testing if object is of lale pipeline type")
    test_case.assertIsInstance(pipeline, BasePipeline)


def validate_autoai_experiment(experiment_info, autoai_pod_version, data_join_graph = None):
    """
    Checks if all requiered parameters for AutoAI experiment was set.
    Also adds autoai pod version if missing.

    Returns
    -------
    Dictionary with checked experiment info.
    """

    required_params = ['name', 'prediction_type', 'prediction_column']

    for param in required_params:
        test_case.assertIn(param, experiment_info.keys(),
                           msg=f"{param} is missing in experiment parameters {experiment_info}")

    if "autoai_pod_version" not in experiment_info.keys():
        experiment_info['autoai_pod_version'] = autoai_pod_version

    if data_join_graph:
        experiment_info['data_join_graph'] = data_join_graph

    return experiment_info


def validate_autoai_timeseries_experiment(experiment_info, autoai_pod_version):
    """
    Checks if all requiered parameters for AutoAI FORECASTING experiment was set.
    Also adds autoai pod version if missing.

    Returns
    -------
    Dictionary with checked experiment info.
    """

    required_params = ['name', 'prediction_columns']

    for param in required_params:
        test_case.assertIn(param, experiment_info.keys(),
                           msg=f"{param} is missing in experiment parameters {experiment_info}")

    if "prediction_type" not in experiment_info.keys():
        experiment_info['prediction_type'] = PredictionType.FORECASTING

    test_case.assertEqual(experiment_info['prediction_type'], PredictionType.FORECASTING,
                          msg="Prediction type for forecasting experiment was set incorrectly ")

    if "autoai_pod_version" not in experiment_info.keys():
        experiment_info['autoai_pod_version'] = autoai_pod_version

    return experiment_info


def validate_autoai_tsad_experiment(experiment_info, autoai_pod_version):
    """
    Checks if all required parameters for AutoAI timeseries prediction experiment was set.
    Also adds autoai pod version if missing.

    Returns
    -------
    Dictionary with checked experiment info.
    """

    required_params = ['name', 'feature_columns']

    for param in required_params:
        test_case.assertIn(param, experiment_info.keys(),
                           msg=f"{param} is missing in experiment parameters {experiment_info}")

    if "prediction_type" not in experiment_info.keys():
        experiment_info['prediction_type'] = PredictionType.TIMESERIES_ANOMALY_PREDICTION

    test_case.assertEqual(experiment_info['prediction_type'], PredictionType.TIMESERIES_ANOMALY_PREDICTION,
                          msg="Prediction type for timeseries prediction experiment was set incorrectly ")

    if "autoai_pod_version" not in experiment_info.keys():
        experiment_info['autoai_pod_version'] = autoai_pod_version

    return experiment_info


def get_and_predict_all_pipelines_as_lale(remote_auto_pipelines: 'RemoteAutoPipelines', x_values):
    """
    Based on summary of the experiment the method is loading each pipeline to lale type and check if predict is working.
    """

    from lale.operators import TrainablePipeline

    summary = remote_auto_pipelines.summary()
    print(summary)

    failed_pipelines = []
    error_messages = []

    for pipeline_name in summary.reset_index()['Pipeline Name']:
        print(f"Getting pipeline: {pipeline_name}")
        try:
            lale_pipeline = remote_auto_pipelines.get_pipeline(pipeline_name=pipeline_name)
            test_case.assertIsInstance(lale_pipeline, TrainablePipeline)
            predictions = lale_pipeline.predict(X=x_values[:1])
            print(predictions)
            test_case.assertGreater(len(predictions), 0, msg=f"Returned prediction for {pipeline_name} are empty")
        except:
            print(f"Failure: {pipeline_name}")
            failed_pipelines.append(pipeline_name)
            error_message = traceback.format_exc()
            print(error_message)
            error_messages.append(error_message)

    test_case.assertEqual(len(failed_pipelines), 0, msg=f"Some pipelines failed. Full list: {failed_pipelines} \n "
                                                       f"Errors: {error_messages}")
