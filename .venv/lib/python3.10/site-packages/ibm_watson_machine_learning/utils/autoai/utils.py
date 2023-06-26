#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


__all__ = [
    'fetch_pipelines',
    'load_file_from_file_system',
    'load_file_from_file_system_nonautoai',
    'NextRunDetailsGenerator',
    'prepare_auto_ai_model_to_publish_normal_scenario',
    'prepare_auto_ai_model_to_publish_notebook_normal_scenario',
    'prepare_auto_ai_model_to_publish',
    'remove_file',
    'ProgressGenerator',
    'is_ipython',
    'try_import_lale',
    'try_load_dataset',
    'check_dependencies_versions',
    'try_import_autoai_libs',
    'try_import_autoai_ts_libs',
    'try_import_tqdm',
    'try_import_xlrd',
    'try_import_openpyxl',
    'try_import_graphviz',
    'prepare_cos_client',
    'create_model_download_link',
    'create_summary',
    'prepare_auto_ai_model_to_publish_notebook',
    'get_node_and_runtime_index',
    'download_experiment_details_from_file',
    'prepare_model_location_path',
    'download_wml_pipeline_details_from_file',
    'init_cos_client',
    'check_graphviz_binaries',
    'try_import_joblib',
    'get_sw_spec_and_type_based_on_sklearn',
    'validate_additional_params_for_optimizer',
    'is_list_composed_from_enum',
    'validate_optimizer_enum_values',
    'all_logging_disabled',
    'check_if_ts_pipeline_is_winner',
    'get_values_for_imputation_strategy',
    'translate_imputation_string_strategy_to_enum',
    'translate_estimator_string_to_enum',
    'translate_batched_estimator_string_to_enum',
    'convert_dataframe_to_fields_values_payload',
    'get_autoai_run_id_from_experiment_metadata'
]


import io
import json
import os
import enum
import gzip
import inspect
import re
import shutil
import sys
from contextlib import redirect_stdout
from functools import wraps
from subprocess import check_call
from sys import executable
from tarfile import open as open_tar
from typing import Dict, Union, Tuple, List, TYPE_CHECKING, Optional, Any
from warnings import warn
from zipfile import ZipFile
from contextlib import contextmanager
import pandas as pd
import logging
from collections.abc import Sequence

import pkg_resources
import ibm_watson_machine_learning._wrappers.requests as requests
from packaging import version

from .enums import (RegressionAlgorithms, RegressionAlgorithmsCP4D, ClassificationAlgorithms,
                    ClassificationAlgorithmsCP4D, ForecastingAlgorithms, ForecastingAlgorithmsCP4D, Transformers,
                    Metrics, TShirtSize, PredictionType, DataConnectionTypes, ImputationStrategy,
                    BatchedRegressionAlgorithms, BatchedClassificationAlgorithms)
from .errors import (MissingPipeline, DataFormatNotSupported, LibraryNotCompatible,
                     CannotInstallLibrary, CannotDownloadTrainingDetails, CannotDownloadWMLPipelineDetails,
                     VisualizationFailed, AdditionalParameterIsUnexpected, InvalidSequenceValue, NoAvailableMetrics,
                     DiscardedModel, WrongModelName, StrategyIsNotApplicable, InconsistentImputationListElements,
                     NoAvailableNotebookLocation, FitNotCompleted)
from ibm_watson_machine_learning.utils import get_module_version

if TYPE_CHECKING:
    from io import BytesIO, BufferedIOBase
    from pandas import DataFrame
    from collections import OrderedDict
    from sklearn.pipeline import Pipeline
    from ibm_watson_machine_learning import APIClient
    from ibm_watson_machine_learning.helpers import DataConnection, S3Connection
    from ibm_boto3 import resource, client

import logging
_logger = logging.getLogger(__name__)


def create_model_download_link(file_path: str):
    """Create download link and shows it in the jupyter notebook.

    :param file_path: path to model file
    :type file_path: str
    """
    if is_ipython():
        from IPython.display import display
        from ibm_watson_machine_learning.utils import create_download_link
        display(create_download_link(file_path))


def fetch_pipelines(run_params: dict,
                    path: str,
                    wml_client: 'APIClient',
                    pipeline_name: str = None,
                    load_pipelines: bool = False,
                    store: bool = False,
                    auto_pipelines_parameters: dict = None) -> Tuple[Union[None, Dict[str, 'Pipeline']], bool]:
    """Helper function to download and load computed AutoAI pipelines (sklearn pipelines).

    :param run_params: fetched details of the run/fit
    :type run_params: dict
    :param path: local system path indicates where to store downloaded pipelines
    :type path: str
    :param wml_client: WML API client
    :type wml_client: APIClient
    :param pipeline_name: name of the pipeline to download, if not specified, all pipelines are downloaded
    :type pipeline_name: str, optional
    :param load_pipelines: indicator if we load and return downloaded piepelines
    :type load_pipelines: bool, optional
    :param store: indicator to store pipelines in local filesystem
    :type store: bool, optional
    :param auto_pipelines_parameters: auto pipelines parameters
    :type auto_pipelines_parameters: dict, optional

    :return: list of sklearn Pipelines or None if load_pipelines is set to False, with additional information
        about pipeline nodes check
    :rtype: tuple[list[Pipeline], bool]
    """

    def check_pipeline_nodes(pipeline: dict, request_json: dict, wml_client) -> bool:
        """Automate check all pipeline nodes to find xgboost, lightgbm or snapml dependency."""
        snapml_estimators = [i.value for i in ClassificationAlgorithms if 'Snap' in i.value] + [
            i.value for i in RegressionAlgorithms if 'Snap' in i.value]
        xgboost_estimators = ['XGBClassifierEstimator', 'XGBRegressorEstimator', 'XGBClassifier', 'XGBRegressor']
        lightgbm_estimators = ['LGBMClassifierEstimator', 'LGBMRegressorEstimator', 'LGBMClassifier', 'LGBMRegressor']

        # note: check dependencies for estimators and other packages
        estimator_name = pipeline['context']['intermediate_model'].get('pipeline_nodes', [None])[-1]
        if estimator_name in xgboost_estimators:
            check_lale = check_dependencies_versions(request_json, wml_client, 'xgboost')

        elif estimator_name in lightgbm_estimators:
            check_lale = check_dependencies_versions(request_json, wml_client, 'lightgbm')

        elif estimator_name in snapml_estimators:
            check_lale = check_dependencies_versions(request_json, wml_client, 'snapml')
        else:
            check_lale = check_dependencies_versions(request_json, wml_client, None)

        # TODO: When another package estimators will be available update above!
        # --- end note

        return check_lale

    joblib = try_import_joblib()

    path = os.path.abspath(path)
    pipelines_names = []
    pipelines = {}
    check_lale = True
    is_ts_metrics = 'timeseries' in run_params['entity']['status'].get('metrics', [None])[0]['context']

    if wml_client.ICP:
        model_paths = []

        # note: iterate over all computed pipelines
        for pipeline in run_params['entity']['status'].get('metrics', []):
            model_number = pipeline['context']['intermediate_model']['name'].split('P')[-1]
            model_phase = chose_model_output(model_number=model_number, is_ts_metrics=is_ts_metrics, run_params=run_params)

            # note: populate available pipeline names
            if pipeline['context']['phase'] == model_phase:

                # note: fetch and create model paths from file system
                model_path = pipeline['context']['intermediate_model'].get('location', {}).get('model', None)
                if model_path is None: # model_path is missing for incremental learning pipelines that were not finished yet or failed
                    raise FitNotCompleted(run_params['metadata']['id'])
                # --- end note

                if pipeline_name is None:
                    model_paths.append(model_path)
                    pipelines_names.append(
                        f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}")

                    # note: check dependencies for estimators and other packages
                    request_json = download_request_json(run_params, pipelines_names[-1], wml_client,
                                                         auto_pipelines_parameters=auto_pipelines_parameters)
                    check_lale = check_pipeline_nodes(pipeline, request_json, wml_client)

                # checking only chosen pipeline
                elif pipeline_name == f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}":
                    model_paths.append(model_path)
                    pipelines_names = [f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}"]

                    # note: check dependencies for estimators and other packages
                    request_json = download_request_json(run_params, pipelines_names[-1], wml_client,
                                                         auto_pipelines_parameters=auto_pipelines_parameters)
                    check_lale = check_pipeline_nodes(pipeline, request_json, wml_client)

                    break
        # --- end note

        if load_pipelines:
            # Disable printing to suppress warning from ai4ml
            with redirect_stdout(open(os.devnull, "w")):
                for model_path, pipeline_name in zip(model_paths, pipelines_names):
                    pipelines[pipeline_name] = joblib.load(load_file_from_file_system(wml_client=wml_client,
                                                                                      file_path=model_path))

        if store:
            for name, pipeline in pipelines.items():
                local_model_path = os.path.join(path, name)
                joblib.dump(pipeline, local_model_path)
                print(f"Selected pipeline stored under: {local_model_path}")

                # note: display download link to the model
                create_model_download_link(local_model_path)
                # --- end note

    else:
        from ibm_boto3 import client
        cos_client = client(
            service_name='s3',
            endpoint_url=run_params['entity']['results_reference']['connection']['endpoint_url'],
            aws_access_key_id=run_params['entity']['results_reference']['connection']['access_key_id'],
            aws_secret_access_key=run_params['entity']['results_reference']['connection']['secret_access_key']
        )
        buckets = []
        filenames = []
        keys = []

        pipeline_suffix = "gz" if is_ts_metrics else "pickle"
        for pipeline in run_params['entity']['status'].get('metrics', []):
            model_number = pipeline['context']['intermediate_model']['name'].split('P')[-1]
            pipeline_phase = pipeline['context']['phase']
            model_phase = chose_model_output(model_number=model_number, is_ts_metrics=is_ts_metrics, run_params=run_params)

            if pipeline['context']['phase'] == model_phase:
                model_path = f"{pipeline['context']['intermediate_model']['location']['model']}"

                if pipeline_name is None:
                    buckets.append(run_params['entity']['results_reference']['location']['bucket'])
                    filenames.append(
                        f"{path}/Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}.{pipeline_suffix}")
                    keys.append(model_path)
                    pipelines_names.append(
                        f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}")

                    # note: check dependencies for estimators and other packages
                    request_json = download_request_json(run_params, pipelines_names[-1], wml_client,
                                                         auto_pipelines_parameters=auto_pipelines_parameters)
                    check_lale = check_pipeline_nodes(pipeline, request_json, wml_client)

                elif pipeline_name == f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}":
                    buckets = [run_params['entity']['results_reference']['location']['bucket']]
                    filenames = [
                        f"{path}/Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}.{pipeline_suffix}"]
                    keys = [model_path]
                    pipelines_names = [f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}"]

                    # note: check dependencies for estimators and other packages
                    request_json = download_request_json(run_params, pipelines_names[-1], wml_client,
                                                         auto_pipelines_parameters=auto_pipelines_parameters)
                    check_lale = check_pipeline_nodes(pipeline, request_json, wml_client)

                    break

        for bucket, filename, key, name in zip(buckets, filenames, keys, pipelines_names):
            cos_client.download_file(Bucket=bucket, Filename=filename, Key=key)
            if load_pipelines:

                # Disable printing to suppress warning from ai4ml
                with redirect_stdout(open(os.devnull, "w")):
                    if is_ts_metrics:
                        with gzip.open(filename, "rb") as f:
                            pipeline_content = f.read()
                            pipelines[name] = joblib.load(io.BytesIO(pipeline_content))
                    else:
                        pipelines[name] = joblib.load(filename)

                if not store:
                    if os.path.exists(filename):
                        os.remove(filename)

                else:
                    print(f"Selected pipeline stored under: {filename}")

                    # note: display download link to the model
                    create_model_download_link(filename)
                    # --- end note
    if load_pipelines and pipelines:
        return pipelines, check_lale

    elif load_pipelines:
        raise MissingPipeline(
            pipeline_name if pipeline_name is not None else "global_output pipeline",
            reason="The name of the pipeline is incorrect or there are no pipelines computed.")


def _download_notebook(run_params: dict,
                       path: str,
                       wml_client: 'APIClient',
                       pipeline_name: str,
                       filename: str = None) -> str:
    """Helper function to download and load computed AutoAI pipelines (sklearn pipelines).

    :param run_params: fetched details of the run/fit
    :type run_params: dict

    :param path: local system path indicates where to store downloaded pipelines
    :type path: str

    :param wml_client: WML API client
    :type wml_client: APIClient

    :param pipeline_name: name of the pipeline to download, if not specified, all pipelines are downloaded
    :type pipeline_name: str, optional

    :param filename: name of pipeline notebook file, if not specified,
        the file name will be created based on pipeline name
    :type filename: str, optional

    :return: path to saved pipeline notebook
    :rtype: str
    """
    path = os.path.abspath(path)
    is_ts_metrics = 'timeseries' in run_params['entity']['status'].get('metrics', [None])[0]['context']

    if wml_client.ICP:
        # note: iterate over all computed pipelines
        for pipeline in run_params['entity']['status'].get('metrics', []):
            model_number = pipeline['context']['intermediate_model']['name'].split('P')[-1]
            if pipeline_name == f"Pipeline_{model_number}":
                model_phase = chose_model_output(model_number=model_number, is_ts_metrics=is_ts_metrics, run_params=run_params)

                # note: populate available pipeline names
                if pipeline['context']['phase'] == model_phase:

                    # note: fetch and create model paths from file system
                    try:
                        notebook_path = f"{pipeline['context']['intermediate_model']['notebook_location']}"
                    except:
                        raise NoAvailableNotebookLocation(pipeline_name)
                    # --- end note

                    if not filename:
                        filename = f"{path}/Pipeline_{model_number}_notebook.ipynb"

                    with open(filename, 'wb') as file:
                        file.write(load_file_from_file_system(wml_client=wml_client, file_path=notebook_path).getvalue())

                    print(f"Selected pipeline notebook stored under: {filename}")

                    # note: display download link to the model
                    create_model_download_link(filename)
                    # --- end note

                    return filename

    else:
        from ibm_boto3 import client
        cos_client = client(
            service_name='s3',
            endpoint_url=run_params['entity']['results_reference']['connection']['endpoint_url'],
            aws_access_key_id=run_params['entity']['results_reference']['connection']['access_key_id'],
            aws_secret_access_key=run_params['entity']['results_reference']['connection']['secret_access_key']
        )

        for pipeline in run_params['entity']['status'].get('metrics', []):
            model_number = pipeline['context']['intermediate_model']['name'].split('P')[-1]
            if pipeline_name == f"Pipeline_{model_number}":
                model_phase = chose_model_output(model_number=model_number, is_ts_metrics=is_ts_metrics, run_params=run_params)


                if pipeline['context']['phase'] == model_phase:
                    try:
                        model_path = f"{pipeline['context']['intermediate_model']['notebook_location']}"
                    except:
                        raise NoAvailableNotebookLocation(pipeline_name)

                    bucket = run_params['entity']['results_reference']['location']['bucket']
                    if not filename:
                        filename = f"{path}/Pipeline_{model_number}_notebook.ipynb"

                    cos_client.download_file(Bucket=bucket, Filename=filename, Key=model_path)

                    print(f"Selected pipeline notebook stored under: {filename}")

                    # note: display download link to the model
                    create_model_download_link(filename)

                    return filename

    raise NoAvailableNotebookLocation(pipeline_name)


def load_file_from_file_system(wml_client: 'APIClient',
                               file_path: str,
                               stream: bool = True) -> 'io.BytesIO':
    """Load file into memory from the file system.

    :param wml_client: WML v4 client
    :type wml_client: APIClient
    :param file_path: path in the file system of the file
    :type file_path: str
    :param stream: indicator to stream data content
    :type stream: bool, optional

    :return: Sklearn Pipeline
    :rtype: io.BytesIO
    """
    # note: prepare the file path
    file_path = file_path.split('auto_ml/')[-1]
    # --- end note

    buffer = io.BytesIO()
    response_with_model = requests.get(
        url=f"{wml_client.service_instance._href_definitions.get_wsd_model_attachment_href()}auto_ml/{file_path}",
        headers=wml_client._get_headers(),
        params=wml_client._params(),
        stream=stream)

    wml_client.deployments._handle_response(200, u'get asset file', response_with_model, json_response=False)

    if stream:
        for data in response_with_model.iter_content():
            buffer.write(data)
    else:
        buffer.write(response_with_model.content)

    buffer.seek(0)

    return buffer


def load_file_from_file_system_nonautoai(wml_client: 'APIClient',
                                         file_path: str,
                                         stream: bool = True) -> 'io.BytesIO':
    """Load file into memory from the file system.

    :param wml_client: WML v4 client
    :type wml_client: APIClient
    :param file_path: path in the file system of the file
    :type file_path: str
    :param stream: indicator to stream data content
    :type stream: bool, optional

    :return: file content
    :rtype: io.BytesIO
    """
    # note: prepare the file path
    # --- end note

    buffer = io.BytesIO()

    response_with_model = requests.get(
        url=f"{wml_client.service_instance._href_definitions.get_wsd_model_attachment_href()}{file_path}",
        headers=wml_client._get_headers(),
        params=wml_client._params(),
        stream=stream)

    if stream:
        for data in response_with_model.iter_content():
            buffer.write(data)
    else:
        buffer.write(response_with_model.content)

    buffer.seek(0)

    return buffer


class NextRunDetailsGenerator:
    """Generator class to produce next list of run details."""

    def __init__(self, wml_client: 'APIClient', href: str) -> None:
        self.wml_client = wml_client
        self.next_href = href

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_href is not None:
            response = requests.get(
                url=f"{self.wml_client.wml_credentials['url']}{self.next_href}",
                headers=self.wml_client._get_headers())
            details = response.json()
            self.next_href = details.get('next', {'href': None})['href']
            return details.get('resources', [])

        else:
            raise StopIteration


def preprocess_request_json(request_json: Dict) -> Dict:
    """Removes S3 types from older trainings and replace them with container type."""
    if "content_location" in request_json:
        if "location" in request_json["content_location"]:
            if request_json["content_location"]["location"].get("type") == 's3':
                request_json["content_location"]["location"]["type"] = 'container'

        if request_json["content_location"].get("type") == 's3':
            request_json["content_location"]["type"] = 'container'

    if "training_data_references" in request_json:
        for ref in request_json["training_data_references"]:
            if ref.get("type") == 's3':
                ref['type'] = 'container'

            if "location" in ref:
                if ref['location'].get("type") == 's3':
                    ref['location']['type'] = 'container'

    return request_json


def chose_model_output(model_number: str, is_ml_metrics: bool = True, is_ts_metrics: bool = False,
                       run_params: dict = None) -> str:
    """Chose correct path for particular model number"""
    if is_ts_metrics:
        return 'after_final_pipelines_generation'
    elif is_ml_metrics:
        model_number = int(model_number)
        hpo_c_numbers = (4, 8, 12, 16)
        cognito_numbers = (3, 7, 11, 15)
        hpo_d_numbers = (2, 6, 10, 14)
        pre_hpo_d_numbers = (1, 5, 9, 13)

        if run_params is not None:
            model_name = 'P' + str(model_number)
            return [metric['context']['phase'] for metric in run_params['entity']['status']['metrics']
                    if metric['context']['intermediate_model']['name'] == model_name
                    and metric['context']['phase'] != 'global_output'][0]

        if model_number in pre_hpo_d_numbers:
            return 'pre_hpo_d_output'

        elif model_number in hpo_d_numbers:
            return 'hpo_d_output'

        elif model_number in cognito_numbers:
            return 'cognito_output'

        elif model_number in hpo_c_numbers:
            return 'hpo_c_output'

        else:
            return 'global_output'


def prepare_auto_ai_model_to_publish_notebook_normal_scenario(
        pipeline_model: Union['Pipeline', 'TrainablePipeline'],
        result_connection,
        cos_client,
        run_params: Dict,
        space_id: str,
        auto_pipelines_parameters=None) -> Tuple[str, Dict[str, dict]]:
    """Prepares autoai model to publish in Watson Studio via COS.
    Option only for auto-gen notebooks with correct result references on COS.

    :param pipeline_model: model object to publish
    :type pipeline_model: Pipeline or TrainablePipeline
    :param result_connection: connection object with COS credentials and all needed locations for jsons
    :type result_connection: DataConnection
    :param cos_client: initialized COS client
    :type cos_client: ibm_boto3.resource
    :param run_params: dictionary with training details
    :type run_params: dict
    :param space_id: UID of space
    :type space_id: str
    :param auto_pipelines_parameters: auto pipelines parameters
    :type auto_pipelines_parameters: dict, optional

    :return: path to the saved model and jsons in COS
    :rtype: tuple[str, dict[str, dict]]
    """
    path = result_connection.location._model_location
    model_number = pipeline_model.split('_')[-1]
    run_id = path.split('/data/')[0].split('/')[-1]
    is_ts_metrics = 'autoai-ts' in path
    request_path = f"{path.split('/data/')[0]}/assets/{run_id}_P{model_number}_{chose_model_output(model_number=model_number, is_ts_metrics=is_ts_metrics, run_params=run_params)}/resources/wml_model/request.json"
    bucket = result_connection.location.bucket
    cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=request_path)
    try:
        with open('request.json', 'r') as f:
            request_str = f.read()
    except Exception as e:
        if auto_pipelines_parameters:
            with open('request.json', 'r', encoding=auto_pipelines_parameters.get('encoding')) as f:
                request_str = f.read()
        else:
            raise e

    # note: only if there was 1 estimator during training
    if 'content_location' not in request_str:
        request_path = f"{path.split('/data/')[0]}/assets/{run_id}_P{model_number}_compose_model_type_output/resources/wml_model/request.json"
        cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=request_path)
        try:
            with open('request.json', 'r') as f:
                request_str = f.read()
        except Exception as e:
            if auto_pipelines_parameters:
                with open('request.json', 'r', encoding=auto_pipelines_parameters.get('encoding')) as f:
                    request_str = f.read()
            else:
                raise e

    request_json: Dict[str, dict] = json.loads(request_str)
    if 'connection' in run_params['entity']['results_reference']:
        if "content_location" in request_json:
            if "location" in request_json["content_location"]:
                if request_json["content_location"]["location"].get("type") != 's3':
                    request_json['content_location']['connection'] = run_params['entity']['results_reference']['connection']

    request_json = preprocess_request_json(request_json)

    artifact_name = f"autoai_sdk{os.path.sep}{pipeline_model}.pickle"

    return artifact_name, request_json


# TODO: remove this function
def prepare_auto_ai_model_to_publish_notebook(pipeline_model: Union['Pipeline', 'TrainablePipeline'],
                                              result_connection,
                                              cos_client,
                                              obm: Optional[bool] = False) -> Union[Tuple[Dict[str, dict], str], str]:
    """Prepares autoai model to publish in Watson Studio via COS.
    Option only for auto-gen notebooks with correct result references on COS.

    :param pipeline_model: model object to publish
    :type pipeline_model: Pipeline or TrainablePipeline
    :param result_connection: connection object with COS credentials and all needed locations for jsons
    :type result_connection: DataConnection
    :param cos_client: initialized COS client
    :type cos_client: ibm_boto3.resource
    :param obm: indicator if we need to extract OBM data
    :type obm: bool, optional

    :return: schema json as string and artifact name in tuple
    :rtype: tuple[str, str]
    """
    joblib = try_import_joblib()

    artifact_type = ".gzip"

    artifact_name = f"artifact_auto_ai_model{artifact_type}"
    model_artifact_name = f"model_.tar.gz"
    wml_pipeline_definition_name = "pipeline-model.json"
    obm_model_name = "obm_model.zip"
    temp_model_name = '__temp_model.pickle'

    # note: path to the json describing the autoai POD specification
    path_parts = result_connection.location._model_location.split('model.pickle')
    if len(path_parts) == 1:
        path = result_connection.location._model_location.split('model.zip')[0]
        path = f"{path}pre_hpo_d_output/Pipeline1/"

    else:
        path = result_connection.location._model_location.split('model.pickle')[0]

    pipeline_model_json_path = f"{path}pipeline-model.json"
    schema_path = f"{path}schema.json"

    bucket = result_connection.location.bucket

    # note: Check if we have OBM experiment and get paths for obm model and schema
    if obm:
        obm_model_path = f"{path.split('/data/')[0]}/data/obm/model.zip"
        schema_path = f"{path.split('/data/')[0]}/data/obm/schemas.json"
        cos_client.meta.client.download_file(Bucket=bucket, Filename=obm_model_name, Key=obm_model_path)

    # note: need to download model schema and wml pipeline definition json
    cos_client.meta.client.download_file(Bucket=bucket, Filename=wml_pipeline_definition_name,
                                         Key=pipeline_model_json_path)
    cos_client.meta.client.download_file(Bucket=bucket, Filename='schema.json', Key=schema_path)

    with open('schema.json', 'r') as f:
        schema_json = f.read()

    # note: update the schema, it has wrong field types
    schema_json = schema_json.replace('fieldType', 'type')
    # --- end note

    # note: saved passed model as pickle, for further tar.gz packaging
    joblib.dump(pipeline_model, temp_model_name)
    # --- end note

    # note: create a tar.gz file with model pickle, name it as 'model_run_id.tar.gz', model.pickle inside
    with open_tar(model_artifact_name, 'w:gz') as tar:
        tar.add(temp_model_name, arcname='model.pickle')

    remove_file(filename=temp_model_name)
    # --- end note

    # note: create final zip to publish on WML cloud v4 GA
    with ZipFile(artifact_name, 'w') as zip_file:
        if obm:
            # note: write order is important!
            zip_file.write(obm_model_name)
        zip_file.write(model_artifact_name)
        zip_file.write(wml_pipeline_definition_name)

    remove_file(filename=model_artifact_name)
    remove_file(filename=wml_pipeline_definition_name)
    if obm:
        remove_file(filename=obm_model_name)
    # --- end note

    return json.loads(schema_json), artifact_name


def prepare_auto_ai_model_to_publish_normal_scenario(
        pipeline_model: Union['Pipeline', 'TrainablePipeline'],
        run_params: dict,
        run_id: str,
        wml_client: 'APIClient',
        space_id: str = None,
        result_reference: 'DataConnection' = None,
        auto_pipelines_parameters: dict = None) -> Union[Tuple[str, Dict[str, dict]]]:
    """Helper function to specify `content_location` statement for AutoAI models to store in repository.

    :param pipeline_model: model that will be prepared for an upload
    :type pipeline_model: Pipeline or TrainablePipeline
    :param run_params: fetched details of the run/fit
    :type run_params: dict
    :param run_id: fit/run ID associated with the model
    :type run_id: str
    :param wml_client: WML API Client
    :type wml_client: APIClient
    :param space_id: UID of space
    :type space_id: str, optional
    :param result_reference: needed when we have something different than old S3 on a Cloud
    :type result_reference: DataConnection, optional
    :param auto_pipelines_parameters: auto pipelines parameters
    :type auto_pipelines_parameters: dict, optional

    :return:
        if cp4d: dictionary with model schema and artifact name to upload, stored temporally in the user local
            file system
        else: path name to the stored model in COS
    :rtype: dict or str
    """
    try:
        request_json: Dict[str, dict] = download_request_json(run_params, pipeline_model, wml_client, result_reference,
                                                              auto_pipelines_parameters=auto_pipelines_parameters)
    except:
        raise MissingPipeline(
            pipeline_model,
            reason="The name of the pipeline is incorrect or there are no pipelines computed."
        )
    # note: fill connection details
    if 'connection' in run_params['entity']['results_reference']:
        if "content_location" in request_json:
            if "location" in request_json["content_location"]:
                if request_json["content_location"]["location"].get("type") != 's3' and request_json["content_location"]["location"].get("type") != 'container':
                    request_json['content_location']['connection'] = run_params['entity']['results_reference']['connection']

    artifact_name = f"autoai_sdk{os.path.sep}{pipeline_model}.pickle"

    return artifact_name, request_json


# TODO: remove this function
def prepare_auto_ai_model_to_publish(
        pipeline_model: Union['Pipeline', 'TrainablePipeline'],
        run_params: dict,
        run_id: str,
        wml_client: 'APIClient') -> Union[Tuple[Dict[str, dict], str], str]:
    """Helper function to download and load computed AutoAI pipelines (sklearn pipelines).

    :param pipeline_model: model that will be prepared for an upload
    :type pipeline_model: Pipeline or TrainablePipeline
    :param run_params: fetched details of the run/fit
    :type run_params: dict
    :param run_id: fit/run ID associated with the model
    :type run_id: str
    :param wml_client: WML API Client
    :type wml_client: APIClient

    :return:
        if cp4d: Dictionary with model schema and artifact name to upload, stored temporally in the user local
            file system
        else: path name to the stored model in COS

    :rtype: tuple[dict, str] or str
    """

    joblib = try_import_joblib()

    is_ml_metrics = 'ml_metrics' in run_params['entity']['status'].get('metrics')[-1]
    is_ts_metrics = 'ts_metrics' in run_params['entity']['status'].get('metrics')[-1]
    is_tsad_metrics = 'tsad_metrics' in run_params['entity']['status'].get('metrics')[-1]

    if not is_ml_metrics and not is_ts_metrics and not is_tsad_metrics:
        raise NoAvailableMetrics()

    artifact_type = ".tar.gz" if wml_client.ICP else ".gzip"

    artifact_name = f"artifact_auto_ai_model{artifact_type}"
    model_artifact_name = f"model_{run_id}.tar.gz"
    wml_pipeline_definition_name = "pipeline-model.json"
    obm_model_name = "obm_model.zip"
    temp_model_name = '__temp_model.pickle'
    temp_ts_model_name = "model.gzip"

    # note: prepare file paths of pipeline-model and schema (COS / file system location)
    pipeline_info = run_params['entity']['status'].get('metrics')[-1]
    pipeline_model_path = f"{pipeline_info['context']['intermediate_model']['location']['pipeline_model']}"
    schema_path = f"{pipeline_info['context']['intermediate_model']['schema_location']}"
    obm_model_path = None
    # --- end note

    # note: Check if we have OBM experiment and get paths for obm model and schema
    if 'obm' in run_params['entity']['status'].get('feature_engineering_components', {}):
        obm_model_path = f"{pipeline_model_path.split('/data/')[0]}/data/obm/model.zip"
        schema_path = f"{pipeline_model_path.split('/data/')[0]}/data/obm/schemas.json"

    if wml_client.ICP:
        # note: downloading pipeline-model.json and schema.json from file system on CP4D
        schema_json = load_file_from_file_system(wml_client=wml_client, file_path=schema_path).read().decode()
        pipeline_model_json = load_file_from_file_system(wml_client=wml_client,
                                                         file_path=pipeline_model_path).read().decode()
        with open(wml_pipeline_definition_name, 'w') as f:
            f.write(pipeline_model_json)
        # --- end note

        # note: save obm model.zip locally
        if obm_model_path is not None:
            obm_model = load_file_from_file_system(wml_client=wml_client,
                                                   file_path=obm_model_path).read().decode()

            with open(obm_model_name, 'w') as f:
                f.write(obm_model)
        # --- end note

    else:
        from ibm_watson_machine_learning.helpers import DataConnection
        # Note: container type needs to be converted to S3 type to handle credentials!
        bucket = None

        results_reference = DataConnection._from_dict(run_params['entity']['results_reference'])
        results_reference.set_client(wml_client)
        results_reference._check_if_connection_asset_is_s3()
        bucket = results_reference.location.bucket  # bucket is removed when casting to dict for Container type
        run_params['entity']['results_reference'] = results_reference._to_dict() # update resolbved credentials in run params
        # --- end note

        if run_params['entity']['results_reference']['type'] == DataConnectionTypes.CA:
            connection_details = wml_client.connections.get_details(run_params['entity']['results_reference']['connection']['id'])['entity']
            cos_client = init_cos_client(connection_details)
        else:
            cos_client = init_cos_client(run_params['entity']['results_reference']['connection'])

        bucket = bucket if bucket is not None else run_params['entity']['results_reference']['location']['bucket']


        # note: need to download model schema and wml pipeline definition json
        cos_client.meta.client.download_file(Bucket=bucket, Filename=wml_pipeline_definition_name,
                                             Key=pipeline_model_path)
        cos_client.meta.client.download_file(Bucket=bucket, Filename='schema.json', Key=schema_path)

        with open('schema.json', 'r') as f:
            schema_json = f.read()

        # note: save obm model.zip locally
        if obm_model_path is not None:
            cos_client.meta.client.download_file(Bucket=bucket, Filename=obm_model_name, Key=obm_model_path)
        # --- end note

    # note: update the schema, it has wrong field types and missing id
    schema_json = schema_json.replace('fieldType', 'type')
    # --- end note


    if is_ml_metrics:
        # note: saved passed model as pickle, for further tar.gz packaging
        joblib.dump(pipeline_model, temp_model_name)
        # --- end note
        # note: create a tar.gz file with model pickle, name it as 'model_run_id.tar.gz', model.pickle inside
        with open_tar(model_artifact_name, 'w:gz') as tar:
            tar.add(temp_model_name, arcname='model.pickle')
        # --- end note
        remove_file(filename=temp_model_name)
    else: # ts & tsad scenario
        try_import_autoai_ts_libs()
        from autoai_ts_libs.utils.score import Score
        Score.save(pipeline_model, temp_ts_model_name)
        with open_tar(model_artifact_name, 'w:gz') as tar:
            tar.add(temp_ts_model_name, arcname='model.gzip')
        remove_file(filename=temp_model_name)

    with ZipFile(artifact_name, 'w') as zip_file:
        if obm_model_path is not None:
            # note: write order is important!
            zip_file.write(obm_model_name)
        zip_file.write(model_artifact_name)
        zip_file.write(wml_pipeline_definition_name)

    remove_file(filename=model_artifact_name)
    remove_file(filename=wml_pipeline_definition_name)
    if obm_model_path is not None:
        remove_file(filename=obm_model_name)
    # --- end note

    return json.loads(schema_json), artifact_name


def modify_pipeline_model_json(data_location: str, model_path: str) -> None:
    """Change the location of KB model in pipeline-model.json

    :param data_location: pipeline-model.json data local path
    :type data_location: str

    :param model_path: path to KB model stored in COS
    :type model_path: str
    """
    with open(data_location, 'r') as f:
        data = json.load(f)

    data['pipelines'][0]['nodes'][-1]['parameters']['output_model']['location'] = f"{model_path}model.pickle"

    with open(data_location, 'w') as f:
        f.write(json.dumps(data))


def init_cos_client(connection: dict) -> 'resource':
    """Initiate COS client for further usage."""
    from ibm_botocore.client import Config
    from ibm_boto3 import resource

    # note: In case of connection_asset we need too get COS credentials from connection details.
    if connection.get("properties") is not None:
        if "access_key" in connection["properties"] and "secret_key" in connection["properties"]:
            connection = {
                "endpoint_url": connection["properties"].get("url"),
                "access_key_id": connection["properties"].get("access_key"),
                "secret_access_key": connection["properties"].get("secret_key"),
            }
        else:
            connection = {
                "endpoint_url": connection["properties"].get("url"),
                "api_key": connection["properties"].get("api_key"),
                "auth_endpoint": connection["properties"].get("iam_url")
            }
    # --- end note

    if connection.get('auth_endpoint') is not None and connection.get('api_key') is not None:
        cos_client = resource(
            service_name='s3',
            ibm_api_key_id=connection['api_key'],
            ibm_auth_endpoint=connection['auth_endpoint'],
            config=Config(signature_version="oauth"),
            endpoint_url=connection['endpoint_url']
        )
    else:
        cos_client = resource(
            service_name='s3',
            endpoint_url=connection['endpoint_url'],
            aws_access_key_id=connection['access_key_id'],
            aws_secret_access_key=connection['secret_access_key']
        )
    return cos_client


def remove_file(filename: str):
    """Helper function to clean user local storage from temporary package files."""
    if os.path.exists(filename):
        os.remove(filename)


class ProgressGenerator:
    def __init__(self):
        self.progress_messages = {
            "pre_hpo_d_output": 15,
            "hpo_d_output": 30,
            "cognito_output": 50,
            "hpo_c_output": 70,
            "compose_model_type_output": 80,
            "fold_output": 90,
            "global_output": 99,
            # timeseries
            'after_pipeline_execution': 50,
            'after_holdout_execution': 80,
            'after_final_pipelines_generation': 99
        }
        self.total = 100
        self.position = 0
        self.max_position = 5

    def get_progress(self, text):
        for i, e in enumerate(self.progress_messages):
            if e in text:
                pos = self.max_position
                self.max_position = max(self.max_position, self.progress_messages[e])
                if pos < self.max_position:
                    progress = pos - self.position
                    self.position = pos
                    return progress

        if self.position + 1 >= self.max_position:
            return 0
        else:
            self.position += 1
            return 1

    def get_total(self):
        return self.total


def is_ipython():
    """Check if code is running in the notebook."""
    try:
        name = get_ipython().__class__.__name__
        if name != 'ZMQInteractiveShell':
            return False
        else:
            return True

    except Exception:
        return False


def try_import_lale():
    """Check if lale package is installed in local environment, if not, just download and install it."""
    lale_version = '0.4.15'
    try:
        installed_module_version = get_module_version('lale')

        if version.parse(installed_module_version) < version.parse(lale_version):
            warn(f"\"lale\" package version is too low than {lale_version}."
                 f"Installing version {lale_version}")

            try:
                check_call([executable, "-m", "pip", "install", f"lale=={lale_version}"])

            except Exception as e:
                raise CannotInstallLibrary(value_name=e,
                                           reason="lale failed to install. Please install it manually.")

    except pkg_resources.DistributionNotFound as e:
        warn(f"\"lale\" is not installed."
             f"Installing version {lale_version}")

        try:
            check_call([executable, "-m", "pip", "install", f"lale=={lale_version}"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="lale failed to install. Please install it manually.")


def try_import_autoai_libs(minimum_version: str = None):
    """Check if autoai_libs package is installed in local environment, if not, just download and install it."""
    if sys.version_info >= (3, 10):
        default_version = '1.14.0'
    else:
        default_version = '1.13.0'

    if minimum_version is not None:
        default_version = minimum_version

    def install_autoai_libs(version):
        """Try to install autoai-libs via pip."""
        try:
            check_call([executable, "-m", "pip", "install", f"autoai_libs>={version}"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason=f"autoai_libs>={version} failed to install. Please install it manually.")

    try:
        import autoai_libs

        if minimum_version is not None:
            installed_module_version = get_module_version('autoai-libs')

            if version.parse(installed_module_version) < version.parse(minimum_version):
                install_autoai_libs(version=minimum_version)

    except ImportError:
        warn(f"\"autoai_libs\" package is not installed. "
             f"This is the needed dependency for pipeline model refinery, we will try to install it now...")
        install_autoai_libs(version=default_version)


def try_import_autoai_ts_libs():
    """Check if autoai_ts_libs package is installed in local environment, if not, just download and install it."""
    try:
        import autoai_ts_libs

    except ImportError:
        warn(f"\"autoai_libs\" package is not installed. "
             f"This is the needed dependency for pipeline model refinery, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "autoai-ts-libs"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="autoai-ts-libs failed to install. Please install it manually.")


def try_import_tqdm():
    """Check if tqdm package is installed in local environment, if not, just download and install it."""
    try:
        import tqdm

    except ImportError:
        warn(f"\"tqdm\" package is not installed. "
             f"This is the needed dependency for pipeline training, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "tqdm==4.43.0"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="tqdm==4.43.0 failed to install. Please install it manually.")


def try_import_xlrd():
    """Check if xlrd package is installed in local environment, if not, just download and install it."""
    try:
        import xlrd

    except ImportError:
        warn(f"\"xlrd\" package is not installed. "
             f"This is the needed dependency for loading dataset from xls files, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "xlrd==1.2.0"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="xlrd==1.2.0 failed to install. Please install it manually.")


def try_import_graphviz():
    """Check if graphviz package is installed in local environment, if not, just download and install it."""
    try:
        import graphviz

    except ImportError:
        warn(f"\"graphviz\" package is not installed. "
             f"This is the needed dependency for visualizing data join graph, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "graphviz==0.14"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="graphviz==0.14 failed to install. Please install it manually.")


def try_import_joblib():
    """Check if joblib is available from scikit-learn or externally and change 'load' method to inform the user about
    compatibility issues.
    """

    try:
        # note only up to scikit version 0.20.3
        from sklearn.externals import joblib

    except ImportError:
        # only for scikit 0.23.*
        import joblib

    return joblib


ENCODING_LIST = [
# Main encodings to support
'utf-8', 'utf_16', 'gb18030', 'utf_32', 'iso_8859_1', 'latin-1',
# Less popular
'utf_8_sig', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_32_be', 'utf_32_le',
# Esoteric
'cp-424', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256', 'cp1257', 'cp1257', 'cp65001', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr', 'gb2312', 'gbk', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7', 'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_12', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213' ]


def try_load_dataset(
        buffer: Union['BytesIO', 'BufferedIOBase'],
        sheet_name: str = 0,
        separator: str = ',',
        encoding: Optional[str] = 'utf-8') -> Union['DataFrame', 'OrderedDict']:
    """Load data into a pandas DataFrame from BytesIO object.

    :param buffer: buffer with bytes data
    :type buffer: BytesIO or BufferedIOBase

    :param sheet_name: name of the xlsx sheet to read
    :type sheet_name: str, optional

    :param separator: csv separator
    :type separator: str, optional

    :param encoding: data encoding
    :type encoding: str, optional

    :return: loaded dataset
    :rtype: DataFrame or OrderedDict
    """
    from pandas import read_csv, read_excel
    data = None

    try:
        try:
            buffer.seek(0)
            data = read_csv(buffer, sep=separator, encoding=encoding)
        except Exception as e:
            msg = "File failed to load using separator '{}' in {} format".format(separator, encoding)
            _logger.warning(msg)
            encodings = ENCODING_LIST.copy()
            encodings.remove(encoding)

            for enc in encodings:
                try:
                    buffer.seek(0)
                    data = read_csv(buffer, sep=separator, encoding=enc)
                    break
                except UnicodeDecodeError:
                    msg = 'File is not in {} format'.format(enc)
                    _logger.warning(msg)
                except Exception:
                    msg = "File failed to load using separator '{}' in {} format".format(separator, enc)
                    _logger.warning(msg)

            if data is None:
                raise e

    except Exception as e1:
        try:
            try_import_xlrd()
            try_import_openpyxl()
            buffer.seek(0)
            data = read_excel(buffer, sheet_name=sheet_name)

        except Exception as e2:
            raise DataFormatNotSupported(None, reason=f"Error1: {e1} Error2: {e2}")

    return data


def try_import_openpyxl():
    """Check if openpyxl package is installed in local environment, if not, just download and install it."""
    try:
        import openpyxl

    except ImportError:
        warn(f"\"openpyxl\" package is not installed. "
             f"This is the needed dependency for loading dataset from xlsx files, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "openpyxl"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="openpyxl failed to install. Please install it manually.")


def try_load_tar_gz(
        buffer: Union['BytesIO', 'BufferedIOBase'],
        separator: str = ',',
        encoding: Optional[str] = 'utf-8') -> Union['DataFrame', 'OrderedDict']:
    """Load csv packed into tar.gz into a pandas DataFrame from BytesIO object.

    :param buffer: buffer with bytes data
    :type buffer: BytesIO or BufferedIOBase

    :param separator: csv separator
    :type separator: str, optional

    :param encoding: data encoding
    :type encoding: str, optional

    :return: loaded dataset
    :rtype: DataFrame or OrderedDict
    """
    from pandas import read_csv
    import uuid
    import tarfile

    tmp_file = f'tmp_file_{uuid.uuid4()}.tar.gz'

    try:
        with open(tmp_file, 'wb') as out:
            out.write(buffer.read())

        tar = tarfile.open(tmp_file, "r:gz")
        tar.extractall()
        tar.close()

        extracted_file_name = 'holdout_data_indices.csv'

        data = read_csv(extracted_file_name, sep=separator, encoding=encoding)
        return data
    finally:
        os.remove(tmp_file)
        os.remove('holdout_data_indices.csv')


def check_dependencies_versions(request_json: dict, wml_client, estimator_pkg: str) -> bool:
    """Check packages installed versions and inform the user about needed ones.

    :param request_json: dictionary with request from training saved on user COS or CP4D fs
    :type request_json: dict

    :param wml_client: internal WML client used for sw spec requests
    :type wml_client: APIClient

    :param estimator_pkg: name of the estimator package to check with
    :type estimator_pkg: str

    :return: check result
    :rtype: bool
    """
    sw_spec_name = request_json.get('hybrid_pipeline_software_specs', [{'name': None}])[-1]['name']
    sw_spec_id = wml_client.software_specifications.get_id_by_name(sw_spec_name)
    sw_spec = wml_client.software_specifications.get_details(sw_spec_id)

    packages = sw_spec['entity']['software_specification']['software_configuration']['included_packages']

    check_lale = True
    if 'lale' in str(packages):
        check_lale = False

    packages_to_check = ['numpy', 'scikit-learn', 'autoai-libs', 'gensim', 'lale']

    if estimator_pkg is not None:
        packages_to_check.append(estimator_pkg)
        packages_to_check.append(f'py-{estimator_pkg}')

    errored_packages = []

    for package in packages:
        if package['name'] in packages_to_check:
            try:
                installed_module_version = get_module_version(package['name'])

                # workaround for autoai-libs and lale and snapml: versions >= version in SW is accepted
                if package['name'] == 'autoai-libs' or package['name'] == 'lale' or package['name'] == 'snapml':
                    if version.parse(installed_module_version) < version.parse(package['version']):
                        # we should produce different error for lale and snap_ml as only 2 major numbers are significant
                        packages_less_strict_error_list = ['lale', 'snapml']
                        if package['name'] in packages_less_strict_error_list and len(
                                package['version'].split('.')) >= 3:
                            _version = package['version'].split('.')[:3]  # take only 3 major version number
                            package['version'] = '.'.join(_version)
                            package['version'] = f">={package['version']}"

                        errored_packages.append(package)

                # check up to minor version for numpy and xgboost (ex. 1.1.x, if SW spec version is 1.1.1)
                elif package['name'] == 'numpy' or package['name'] == 'xgboost':
                    installed_version = version.parse(installed_module_version)
                    sw_spec_version = version.parse(package['version'])

                    if installed_version.major != sw_spec_version.major:
                        errored_packages.append(package)

                    elif installed_version.minor != sw_spec_version.minor:
                        errored_packages.append(package)

                # additional workarounds for lightgbm
                elif package['name'] == 'lightgbm' or package['name'] == 'py-lightgbm':
                    if package['version'] == '3.1.1':
                        # Note: temporary check / workaround for lightgbm on WS
                        if installed_module_version == '3.1.1' or installed_module_version == '3.2.1':
                            pass
                        else:
                            errored_packages.append(package)
                    else:
                        lightgbm_installed_version = version.parse(installed_module_version)
                        lightgbm_sw_spec_version = version.parse(package['version'])

                        if lightgbm_installed_version.major != lightgbm_sw_spec_version.major:
                            errored_packages.append(package)

                        elif lightgbm_installed_version.minor != lightgbm_sw_spec_version.minor:
                            errored_packages.append(package)

                else:
                    if installed_module_version != package['version']:
                        errored_packages.append(package)

            except pkg_resources.DistributionNotFound as e:
                errored_packages.append(package)

        else:
            pass

    if errored_packages:
        raise LibraryNotCompatible(reason=f"Please check if you have installed correct versions "
                                          f"of the following packages: {errored_packages} "
                                          f"These packages are required to load ML model successfully "
                                          f"on your environment.")

    return check_lale


def prepare_cos_client(
        training_data_references: List['DataConnection'] = None,
        training_result_reference: 'DataConnection' = None) -> Tuple[Union[List[Tuple['DataConnection', 'resource']]],
                                                                     Union[Tuple['DataConnection', 'resource'], None]]:
    """Create COS clients for training data and results.

    :param training_data_references: references to training data
    :type training_data_references: list[DataConnection], optional

    :param training_result_reference: reference to training result
     :type training_result_reference: DataConnection, optional

    :return: list of COS clients for training data, client for results
    :rtype: tuple[list[tuple[DataConnection, resource]], tuple[DataConnection, resource]]
    """
    from ibm_watson_machine_learning.helpers import S3Connection
    from ibm_boto3 import resource
    from ibm_botocore.client import Config

    def differentiate_between_credentials(connection: 'S3Connection') -> 'resource':
        # note: we do not know which version of COS credentials user used during training
        if hasattr(connection, 'auth_endpoint') and hasattr(connection, 'api_key'):
            cos_client = resource(
                service_name='s3',
                ibm_api_key_id=connection.api_key,
                ibm_auth_endpoint=connection.auth_endpoint,
                config=Config(signature_version="oauth"),
                endpoint_url=connection.endpoint_url
            )

        else:
            cos_client = resource(
                service_name='s3',
                endpoint_url=connection.endpoint_url,
                aws_access_key_id=connection.access_key_id,
                aws_secret_access_key=connection.secret_access_key
            )
        # --- end note

        return cos_client

    cos_client_results = None
    data_cos_clients = []

    if training_result_reference is not None:
        if (isinstance(training_result_reference.connection, S3Connection) or
                training_result_reference._check_if_connection_asset_is_s3()):
            cos_client_results = (training_result_reference,
                                  differentiate_between_credentials(connection=training_result_reference.connection))

    if training_data_references is not None:
        for reference in training_data_references:
            if (isinstance(reference.connection, S3Connection) or reference._check_if_connection_asset_is_s3()):
                data_cos_clients.append((reference,
                                         differentiate_between_credentials(connection=reference.connection)))

    return data_cos_clients, cos_client_results


def create_summary(details: dict, scoring: str, sort_by_holdout_score: bool = True) -> 'DataFrame':
    """Create summary in a form of a pandas.DataFrame of computed pipelines (should be used in remote and local scenario
    with COS).

    :param details: dictionary with all training data
    :type details: type

    :param scoring: scoring method
    :type scoring: str

    :param sort_by_holdout_score: indicates if we want to sort pipelines by holdout metric (or by training one),
        by default use holdout metric
    :type sort_by_holdout_score: bool, optional

    :return: pipelines summary
    :rtype: pandas.DataFrame
    """
    from pandas import DataFrame

    is_ml_metrics = 'ml_metrics' in details['entity']['status'].get('metrics', [{}])[0]
    is_ts_metrics = 'ts_metrics' in details['entity']['status'].get('metrics', [{}])[0]
    is_tsad_metrics = 'tsad_metrics' in details['entity']['status'].get('metrics', [{}])[0]

    if not is_ml_metrics and not is_ts_metrics and not is_tsad_metrics:
        raise NoAvailableMetrics()

    def get_metrics_names():
        if is_ml_metrics:
            return details['entity']['status'].get('metrics', [{}])[0].get('ml_metrics', {}).keys()
        elif is_ts_metrics:
            names = list(['validation_' + x for x in
                          details['entity']['status'].get('metrics', [{}])[0].get('ts_metrics', {}).get('training',
                                                                                                        {}).keys()])
            try:
                holdout_names = list(['holdout_' + y for y in
                                      [x for x in details['entity']['status'].get('metrics', [{}]) if
                                       'holdout' in x['ts_metrics']][0]['ts_metrics']['holdout'].keys()])
            except IndexError:
                holdout_names = []
            try:
                backtest_names = list(['backtest_' + y for y in
                                       [x for x in details['entity']['status'].get('metrics', [{}]) if
                                        'backtest' in x['ts_metrics']][0]['ts_metrics']['backtest']['avg'].keys()])
            except IndexError:
                backtest_names = []

            return names + holdout_names + backtest_names
        elif is_tsad_metrics:
            tsad_metrics = [x['tsad_metrics'] for x in details['entity']['status'].get('metrics', []) if 'tsad_metrics' in x]
            training_metrics = [x['training'] for x in tsad_metrics if 'training' in x]
            holdout_metrics = [x['holdout']['agg'] for x in tsad_metrics if 'holdout' in x and 'agg' in x['holdout']]
            training_names, holdout_names = [], []
            if training_metrics:
                training_names = [x for x in training_metrics[0].keys()]
            if holdout_metrics:
                for x in holdout_metrics[0].keys():
                    for y in holdout_metrics[0][x].keys():
                        holdout_names.append(x + '_' + y)
            return training_names + holdout_names

    def is_winner(pipeline_name):  # ts only
        if is_ts_metrics or is_tsad_metrics:
            final_pipelines = [x for x in details['entity']['status'].get('metrics', [{}]) if
                               x['context']['phase'] == 'after_final_pipelines_generation']
            if len(final_pipelines) > 0:
                if 'winners' in final_pipelines[-1]['context'].get('timeseries', {}):
                    return pipeline_name in final_pipelines[-1]['context']['timeseries']['winners']

            pipelines_after_holdout_execution = [x for x in details['entity']['status'].get('metrics', [{}]) if
                                                 x['context']['phase'] == 'after_holdout_execution']
            if len(pipelines_after_holdout_execution) > 0:
                return pipeline_name in pipelines_after_holdout_execution[0]['context'].get('timeseries', {}).get(
                    'winners', [])
            else:
                return False
        elif is_ml_metrics:
            return True

    if is_ml_metrics:
        columns = ['Pipeline Name', 'Enhancements', 'Estimator']
    elif is_ts_metrics or is_tsad_metrics:
        columns = ['Pipeline Name', 'Enhancements', 'Estimator', 'Winner']

    columns = (columns +
               [metric_name for metric_name in
                get_metrics_names()])
    values = []

    for pipeline in details['entity']['status'].get('metrics', []):
        model_number = pipeline['context']['intermediate_model']['name'].split('P')[-1]
        pipeline_phase = pipeline['context']['phase']
        model_phase = chose_model_output(model_number, is_ml_metrics, (is_ts_metrics or is_tsad_metrics), run_params=details)

        # note: list only completed pipelines:
        if is_ml_metrics and pipeline['context']['intermediate_model'].get('location', {}).get('model', None) is None:
            continue  #don't list the not-finished pipeline in summary

        if pipeline_phase == model_phase:
            if is_ml_metrics:
                enhancements = []
                steps = pipeline['context']['intermediate_model']['composition_steps']
                hpo_count = len([s for s in steps if 'hpo' in s])

                if hpo_count > 0:
                    enhancements.append('HPO')
                if 'cognito' in steps:
                    enhancements.append('FE')
                if 'TextTransformer' in pipeline['context']['intermediate_model']['pipeline_nodes']:
                    enhancements.append('Text_FE')
                if hpo_count > 1:
                    enhancements.append('HPO')
                if pipeline_phase == 'batch_ensemble_output':
                    enhancements.append('Ensemble')
                if pipeline_phase == 'batch_ensemble_incremental_learning_output':
                    enhancements.append('Ensemble')
                    enhancements.append('Incremental_learning')

                enhancements = ', '.join(enhancements)
            elif is_ts_metrics:
                enhancements = 'HPO, FE'

                if pipeline['context'].get('timeseries', {}).get('autoai_ts_model_type','non-exogenous') == 'exogenous':
                    enhancements += ", SUP" # supporting features/exogenous
            elif is_tsad_metrics:
                enhancements = 'HPO, FE'

            def get_metrics_items():
                if is_ml_metrics:
                    return pipeline['ml_metrics'].items()
                elif is_ts_metrics:
                    def prepare_items(metrics, pipeline_name, metric_name, prefix):
                        try:
                            chosen_obj = [x for x in metrics if
                                          x['context']['intermediate_model']['name'] == pipeline_name and metric_name in
                                          x['ts_metrics']][0]
                        except Exception as e:
                            return []

                        if metric_name == 'backtest':
                            return list({prefix + x: chosen_obj['ts_metrics'][metric_name]['avg'][x] for x in
                                         chosen_obj['ts_metrics'][metric_name]['avg']}.items())
                        else:
                            return list({prefix + x: chosen_obj['ts_metrics'][metric_name][x] for x in
                                         chosen_obj['ts_metrics'][metric_name]}.items())

                    pipeline_name = pipeline['context']['intermediate_model']['name']
                    metrics = details['entity']['status'].get('metrics', [])
                    training_items = prepare_items(metrics, pipeline_name, 'training', 'validation_')
                    holdout_items = prepare_items(metrics, pipeline_name, 'holdout', 'holdout_')
                    backtest_items = prepare_items(metrics, pipeline_name, 'backtest', 'backtest_')

                    return training_items + holdout_items + backtest_items
                elif is_tsad_metrics:
                    def prepare_tsad_items(metrics, pipeline_name, metric_name):
                        try:
                            chosen_obj = [x for x in metrics if
                                          x['context']['intermediate_model']['name'] == pipeline_name and metric_name in
                                          x['tsad_metrics']][0]

                            if metric_name == 'holdout':
                                holdout_metrics = chosen_obj['tsad_metrics'][metric_name]['agg']
                                items = {}
                                for x in holdout_metrics.keys():
                                    for y in holdout_metrics[x].keys():
                                        items[x + '_' + y] = holdout_metrics[x][y].get('mean')
                                return list(items.items())
                            else:
                                items = {}
                                for x in chosen_obj['tsad_metrics'][metric_name].items():
                                    if isinstance(x[1], list):
                                        items[x[0]] = x[1][0][pipeline_name.lower()]
                                    else:
                                        items[x[0]] = x[1]
                                return list(items.items())
                        except Exception as e:
                            return []

                    pipeline_name = pipeline['context']['intermediate_model']['name']
                    metrics = details['entity']['status'].get('metrics', [])
                    training_items = prepare_tsad_items(metrics, pipeline_name, 'training')
                    holdout_items = prepare_tsad_items(metrics, pipeline_name, 'holdout')

                    return training_items + holdout_items

            # note: workaround when some pipelines have less or more metrics computed
            if is_ml_metrics:
                metrics = columns[3:]
            elif is_ts_metrics or is_tsad_metrics:
                metrics = columns[4:]
            pipeline_metrics = [None] * len(metrics)
            for metric, value in get_metrics_items():
                for i, metric_name in enumerate(metrics):
                    if metric_name == metric:
                        pipeline_metrics[i] = value
            # --- end note

            if is_ml_metrics:
                values.append(
                    ([f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}"] +
                     [enhancements] +
                     [pipeline['context']['intermediate_model']['pipeline_nodes'][-1].replace('base_estimator=', '')] +
                     pipeline_metrics
                     ))
            elif is_ts_metrics or is_tsad_metrics:
                values.append(
                    ([f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}"] +
                     [enhancements] +
                     [pipeline['context']['intermediate_model']['pipeline_nodes'][-1]] +
                     [is_winner(pipeline['context']['intermediate_model']['name'])] +
                     pipeline_metrics
                     ))

    pipelines = DataFrame(data=values, columns=columns)
    pipelines.drop_duplicates(subset="Pipeline Name", keep='first', inplace=True)
    pipelines.set_index('Pipeline Name', inplace=True)

    try:
        if is_ml_metrics:
            if sort_by_holdout_score:
                pipelines = pipelines.sort_values(
                    by=[f"holdout_{scoring}"], ascending=False)
            else:
                pipelines = pipelines.sort_values(
                    by=[f"training_{scoring}"], ascending=False)

            # the optimization score (metric) should be first one
            scoring_metric_column = pipelines.pop(f"training_{scoring}")
            pipelines.insert(2, f"training_{scoring}", scoring_metric_column)

        elif is_ts_metrics or is_tsad_metrics:
            metrics_names = get_metrics_names()
            pipelines = pipelines.sort_values(
                by=['Winner'] + ([metrics_names[0]] if len(metrics_names) > 0 else []), ascending=False)

    # note: sometimes backend will not return 'training_' prefix to the metric
    except KeyError:
        pass

    def neg(x):
        if x is None or (x.__class__.__name__ == 'Series' and len(x) == 0):
            return x

        if x.__class__.__name__ == 'Series' or type(x) is list:
            return [neg(a) for a in x]
        else:
            return -x

    # for columns with _neg_ inside name
    neg_columns = [col for col in pipelines if '_neg_' in col]
    pipelines[neg_columns] = pipelines[neg_columns].apply(lambda x: neg(x))
    pipelines = pipelines.rename(columns={col: col.replace('_neg_', '_') for col in neg_columns})

    # for columns with neg_ on beginning of name
    neg_columns = [col for col in pipelines if col.startswith('neg_')]
    pipelines[neg_columns] = pipelines[neg_columns].apply(lambda x: neg(x))
    pipelines = pipelines.rename(columns={col: col[4:] for col in neg_columns})

    return pipelines


def get_node_and_runtime_index(node_name: str, optimizer_config: dict) -> Tuple[int, int]:
    """Find node index from node name in experiment parameters."""
    node_number = None
    runtime_number = None

    for i, node in enumerate(optimizer_config['entity']['document']['pipelines'][0]['nodes']):
        if node_name == 'kb' and (node.get('id') == 'kb' or node.get('id') == 'automl'):
            node_number = i
            break

        elif node_name == 'obm' and node.get('id') == 'obm':
            node_number = i
            break

        elif node_name == 'ts' and (node.get('id') == 'ts' or node.get('id') == 'autoai-ts'):
            node_number = i
            break
        elif node_name == 'tsad' and (node.get('id') == 'tsad' or node.get('id') == 'autoai-tsad'):
            node_number = i
            break

    for i, runtime in enumerate(optimizer_config['entity']['document']['runtimes']):
        if node_name == 'kb' and (runtime.get('id') == 'kb' or runtime.get('id') == 'automl' or
                                  runtime.get('id') == 'autoai'):
            runtime_number = i
            break

        elif node_name == 'obm' and runtime.get('id') == 'obm':
            runtime_number = i
            break

        elif node_name == 'ts' and runtime.get('id') == 'autoai':
            runtime_number = i
            break

        elif node_name == 'tsad' and runtime.get('id') == 'autoai':
            runtime_number = i
            break

    return node_number, runtime_number


def download_experiment_details_from_file(result_client_and_connection: Tuple['DataConnection', 'resource']) -> dict:
    """Try to download training details from user COS."""

    try:
        file = result_client_and_connection[1].Object(
            result_client_and_connection[0].location.bucket,
            result_client_and_connection[0].location._training_status).get()

        details = json.loads(file['Body'].read())

    except Exception as e:
        raise CannotDownloadTrainingDetails('', reason=f"Error: {e}")

    return details


def download_wml_pipeline_details_from_file(result_client_and_connection: Tuple['DataConnection', 'resource']) -> dict:
    """Try to download wml pipeline details from user COS."""
    try:
        model_location = result_client_and_connection[0].location._model_location
        is_timeseries = 'autoai-ts' in model_location

        path_parts = model_location.split('model.pickle')
        if len(path_parts) == 1:
            path = model_location.split('model.zip')[0]
            if is_timeseries:
                path = f"{path}after_pipeline_execution/P1/"
            else:
                path = f"{path}pre_hpo_d_output/Pipeline1/"


        else:
            path = model_location.split('model.pickle')[0]

        path = f"{path}pipeline-model.json"

        file = result_client_and_connection[1].Object(
            result_client_and_connection[0].location.bucket,
            path).get()

        details = json.loads(file['Body'].read())

    except Exception as e:
        raise CannotDownloadWMLPipelineDetails('', reason=f"Error: {e}")

    return details


def prepare_model_location_path(model_path: str) -> str:
    """To be able to get best pipeline after computation we need to change model_location string to global_output."""

    if "data/automl/" in model_path:
        path = model_path.split('data/automl/')[0]
        path = f"{path}data/automl/global_output/"

    else:
        path = model_path.split('data/kb/')[0]
        path = f"{path}data/kb/global_output/"

    return path


def check_graphviz_binaries(f):
    @wraps(f)
    def _f(*method_args, **method_kwargs):
        from graphviz.backend import ExecutableNotFound
        try:
            output = f(*method_args, **method_kwargs)

        except ExecutableNotFound as e:
            raise VisualizationFailed(
                reason=f"Cannot perform visualization with graphviz. Please make sure that you have Graphviz binaries "
                       f"installed in your system. Please follow this guide: https://www.graphviz.org/download/")

        return output

    return _f


def get_sw_spec_and_type_based_on_sklearn(client: 'APIClient', spec: str) -> Tuple[str, str]:
    """Based on user environment and pipeline sw spec, check sklearn version and find appropriate sw spec.

    :param client: WML API Client
    :type client: APIClient

    :param spec: software specification name
    :type spec: str

    :return: model_type and sw_spec
    :rtype: tuple[str, str]
    """
    import sklearn

    if '0.20.' in sklearn.__version__ and spec in ['autoai-kb_3.0-py3.6', 'autoai-ts_1.0-py3.6']:
        sw_spec = client.software_specifications.get_id_by_name('autoai-kb_3.0-py3.6')
        model_type = 'scikit-learn_0.20'

    elif '0.20.' in sklearn.__version__ and spec not in ['autoai-kb_3.0-py3.6', 'autoai-ts_1.0-py3.6']:
        raise LibraryNotCompatible(reason="Your version of scikit-learn is different then trained pipeline. "
                                          "Trained pipeline version: 0.23.* "
                                          "Your version: " + sklearn.__version__)

    elif '0.23.' in sklearn.__version__ and spec in ['autoai-kb_3.1-py3.7', 'autoai-ts_1.0-py3.7']:
        sw_spec = client.software_specifications.get_id_by_name('autoai-kb_3.1-py3.7')
        model_type = 'scikit-learn_0.23'

    elif '0.23.' in sklearn.__version__ and spec not in ['autoai-kb_3.1-py3.7', 'autoai-ts_1.0-py3.7']:
        raise LibraryNotCompatible(reason="Your version of scikit-learn is different then trained pipeline. "
                                          "Trained pipeline version: 0.20.* "
                                          "Your version: " + sklearn.__version__)

    else:
        raise LibraryNotCompatible(reason="Your version of scikit-learn is not supported. Use one of [0.20.*, 0.23.*]")

    return model_type, sw_spec


def validate_additional_params_for_optimizer(params):
    expected_params = [
        'learning_type', 'positive_label', 'scorer_for_ranking', 'scorers', 'num_folds', 'random_state',
        'preprocessor_flag', 'preprocessor_hpo_flag', 'preprocessor_hpo_estimator', 'hpo_searcher', 'cv_num_folds',
        'hpo_d_iter_threshold', 'hpo_c_iter_threshold', 'max_initial_points', 'preprocess_transformer_chain',
        'daub_ensembles_flag', 'max_num_daub_ensembles', 'run_hpo_after_daub_flag', 'daub_include_only_estimators',
        'run_cognito_flag', 'cognito_ensembles_flag', 'max_num_cognito_ensembles', 'cognito_display_flag',
        'run_hpo_after_cognito_flag', 'cognito_kwargs', 'cognito_scorers', 'daub_adaptive_subsampling_max_mem_usage',
        'daub_adaptive_subsampling_used_mem_ratio_threshold', 'daub_kwargs', 'compute_feature_importances_flag',
        'compute_feature_importances_options', 'compute_feature_importances_pipeline_options', 'show_status_flag',
        'status_msg_handler', 'state_max_report_priority', 'msg_max_report_priority', 'cognito_pass_ptype',
        'hpo_timeout_in_seconds', 'cognito_use_feature_importances_flag', 'cognito_max_iterations',
        'cognito_max_search_level', 'cognito_transform_names', 'cognito_use_grasspile', 'cognito_subsample',
        'holdout_param', 'missing_values_reference_list', 'datetime_processing_flag',
        'datetime_delete_source_columns', 'datetime_processing_options', 'ensemble_pipelines_flag', 'ensemble_tags',
        'ensemble_comb_method', 'ensemble_selection_flag', 'ensemble_weighted_flag', 'ensemble_corr_sel_method',
        'ensemble_corr_termination_diff_threshold', 'ensemble_num_best_pipelines', 'ensemble_num_folds',
        'compute_pipeline_notebooks_flag', 'pipeline_ranking_metric', 'cpus_available', 'wml_status_msg_version',
        'float32_processing_flag', 'train_remove_missing_target_rows_flag', 'train_sample_rows_test_size',
        'train_sample_columns_index_list', 'train_sample_columns_list', 'preprocessor_cat_imp_strategy', 'preprocessor_cat_enc_encoding',
        'preprocessor_num_imp_strategy', 'preprocessor_num_scaler_use_scaler_flag', 'preprocessor_num_scaler_with_mean',
        'preprocessor_num_scaler_with_std', 'preprocessor_string_compress_type', 'FE_drop_unique_columns_flag',
        'FE_drop_constant_columns_flag', 'FE_add_frequency_columns_flag', 'FE_add_missing_indicator_columns_flag',
        'data_provenance', 'target_label_name', 'preprocessor_data_filename', 'cognito_data_filename',
        'holdout_roc_curve_max_size', 'holdout_reg_pred_obs_max_size', 'max_estimator_n_jobs',
        'enabled_feature_engineering_as_json',
        '_enable_snapml_estimators', 'return_holdout_indices', 'retrain_pipelines_on_holdout_data',
        'sample_size_limit', 'sample_rows_limit', 'sample_percentage_limit', 'sampling_type',
        'n_parallel_data_connections', 'calculate_data_metrics', 'number_of_batch_rows',
        'metrics_on_logical_batch', 'logical_batch_size', 'text_columns_names', 'enable_all_data_sources',
        'scorer_fairness_weight', 'use_flight'
    ]

    for k in params:
        if k not in expected_params:
            raise AdditionalParameterIsUnexpected(k)


def download_request_json(run_params: dict, model_name: str, wml_client, results_reference: 'DataConnection' = None,
                          auto_pipelines_parameters=None) -> dict:
    run_id = run_params['metadata']['id']
    is_ml_metrics = 'ml_metrics' in run_params['entity']['status'].get('metrics')[-1]
    is_ts_metrics = 'ts_metrics' in run_params['entity']['status'].get('metrics')[-1]
    is_tsad_metrics = 'tsad_metrics' in run_params['entity']['status'].get('metrics')[-1]

    if not is_ml_metrics and not is_ts_metrics and not is_tsad_metrics:
        raise NoAvailableMetrics()

    pipeline_info = run_params['entity']['status'].get('metrics')[0]
    schema_path = f"{pipeline_info['context']['intermediate_model']['schema_location']}"
    model_number = model_name.split('_')[-1]
    if is_ml_metrics:
        model_output = chose_model_output(model_number=model_number, run_params=run_params)
    elif is_ts_metrics or is_tsad_metrics:
        model_output = 'after_final_pipelines_generation'

    request_path = f"{schema_path.rsplit('/data/', 1)[0]}/assets/{run_id}_P{model_number}_{model_output}/resources/wml_model/request.json"

    if wml_client.ICP:
        request_bytes = load_file_from_file_system(wml_client=wml_client, file_path=request_path).read()
        try:
            request_str = request_bytes.decode()
        except Exception as e:
            if auto_pipelines_parameters:
                request_str = request_bytes.decode(encoding=auto_pipelines_parameters.get('encoding'))
            else:
                raise e
        # note: only if there was 1 estimator during training
        if 'content_location' not in request_str:
            request_path = f"{schema_path.rsplit('/data/', 1)[0]}/assets/{run_id}_P{model_number}_compose_model_type_output/resources/wml_model/request.json"
            request_bytes = load_file_from_file_system(wml_client=wml_client, file_path=request_path).read()
            try:
                request_str = request_bytes.decode()
            except Exception as e:
                if auto_pipelines_parameters:
                    request_str = request_bytes.decode(encoding=auto_pipelines_parameters.get('encoding'))
                else:
                    raise e

    else:
        # Note: container type needs to be converted to S3 type to handle credentials!
        bucket = None
        if results_reference is not None:
            results_reference._check_if_connection_asset_is_s3()
            bucket = results_reference.location.bucket  # bucket is removed when casting to dict for Container type
            results_reference = results_reference._to_dict()

        else:
            results_reference = run_params['entity']['results_reference']
        # --- end note

        if results_reference['type'] == DataConnectionTypes.CA:
            connection_details = wml_client.connections.get_details(results_reference['connection']['id'])['entity']
            cos_client = init_cos_client(connection_details)
        else:
            cos_client = init_cos_client(results_reference['connection'])

        bucket = bucket if bucket is not None else results_reference['location']['bucket']
        cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=request_path)
        try:
            with open('request.json', 'r') as f:
                request_str = f.read()
        except Exception as e:
            if auto_pipelines_parameters:
                with open('request.json', 'r', encoding=auto_pipelines_parameters.get('encoding')) as f:
                    request_str = f.read()
            else:
                raise e

        # note: only if there was 1 estimator during training
        if 'content_location' not in request_str:
            request_path = f"{schema_path.split('/data/')[0]}/assets/{run_id}_P{model_number}_compose_model_type_output/resources/wml_model/request.json"
            cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=request_path)
            try:
                with open('request.json', 'r') as f:
                    request_str = f.read()
            except Exception as e:
                if auto_pipelines_parameters:
                    with open('request.json', 'r', encoding=auto_pipelines_parameters.get('encoding')) as f:
                        request_str = f.read()
                else:
                    raise e

    request_json: Dict[str, dict] = json.loads(request_str)
    request_json = preprocess_request_json(request_json)

    return request_json


def is_list_composed_from_enum(sequence: List[Union[str, enum.Enum]],
                               enum_class: Union[object, enum.EnumMeta]
                               ) -> None:
    """Check if all the elements of a given sequence are values of a given enum class.

    :param sequence: sequence of elements
    :type sequence: list[str or enum.Enum]
    :param enum_class: class for which validation will be performed, it can be a class inheriting from enum.Enum or
        class which only contains attributes
    :type enum_class: object or enum.EnumMeta

    :raises InvalidSequenceValue: risen if element is not from enum class values
    """
    if sequence is not None:
        if isinstance(enum_class, enum.EnumMeta):
            enum_values = [e.value for e in enum_class]
        else:
            attributes = inspect.getmembers(enum_class, lambda attr: not (inspect.isroutine(attr)))
            enum_values = [attr_val for attr_name, attr_val in attributes
                           if not (attr_name.startswith('__') and attr_name.endswith('__'))]

        # TODO nicer solution
        for el in sequence:
            el_value = el.value if isinstance(el, enum.Enum) else el
            if el_value not in enum_values:
                if f"{el_value}Estimator" not in enum_values:  # ClassificationAlgorithms and RegressionAlgorithms only
                    raise InvalidSequenceValue(el, enum_values)


def validate_optimizer_enum_values(
        prediction_type: str,
        daub_include_only_estimators: List[Union[ClassificationAlgorithms,
                                                 RegressionAlgorithms,
                                                 ForecastingAlgorithms]],
        include_only_estimators: List[Union[ClassificationAlgorithms,
                                            RegressionAlgorithms,
                                            ForecastingAlgorithms]],
        include_batched_ensemble_estimators: List[Union[BatchedClassificationAlgorithms,
                                                        BatchedRegressionAlgorithms]],
        cognito_transform_names: List[Transformers],
        imputation_strategies: List[ImputationStrategy],
        scoring: str,
        t_shirt_size: str,
        is_cpd=False) -> None:
    """Validate if passed optimizer variables takes values from defined enums.

    :param prediction_type: type of the prediction
    :type prediction_type: str

    :param daub_include_only_estimators: list of estimators
    :type daub_include_only_estimators: list

    :param include_only_estimators: list of estimators
    :type include_only_estimators: list

    :param include_batched_ensemble_estimators: list of batched ensemble estimators
    :type include_batched_ensemble_estimators: list

    :param cognito_transform_names: list of transformers
    :type cognito_transform_names: list

    :param imputation_strategies: list of imputation strategies
    :type imputation_strategies: list

    :param scoring: type of the metric to optimize with
    :type scoring: str

    :param t_shirt_size: the size of the remote AutoAI POD instance
    :type t_shirt_size: str

    :param is_cpd: `True` if run on CP4D environment, CP4D estimators will be used to check
    :type is_cpd: bool

    :raises InvalidSequenceValue: risen if element is not from enum class values
    """
    if is_cpd:
        if prediction_type == PredictionType.REGRESSION:
            estimators_enum = RegressionAlgorithmsCP4D
        elif prediction_type == PredictionType.FORECASTING:
            estimators_enum = ForecastingAlgorithmsCP4D
        else:
            estimators_enum = ClassificationAlgorithmsCP4D
    else:
        if prediction_type == PredictionType.REGRESSION:
            estimators_enum = RegressionAlgorithms
        elif prediction_type == PredictionType.FORECASTING:
            estimators_enum = ForecastingAlgorithms
        else:
            estimators_enum = ClassificationAlgorithms

    batched_estimators_enum = BatchedRegressionAlgorithms \
        if prediction_type == PredictionType.REGRESSION \
        else BatchedClassificationAlgorithms

    sequence_enums_pairs = [
        ([prediction_type], PredictionType),
        (daub_include_only_estimators, estimators_enum),
        (include_only_estimators, estimators_enum),
        (include_batched_ensemble_estimators, batched_estimators_enum),
        (cognito_transform_names, Transformers),
        ([t_shirt_size], TShirtSize),
        (imputation_strategies, ImputationStrategy)
    ]
    if scoring is not None:
        sequence_enums_pairs.append(([scoring], Metrics))
    for (sequence, enum_class) in sequence_enums_pairs:
        is_list_composed_from_enum(sequence, enum_class)

@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """A context manager that will prevent any logging messages
    triggered during the body from being processed.

    :param highest_level: the maximum logging level in use
        This would only need to be changed if a custom level greater than CRITICAL
        is defined
    :type highest_level: int
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def check_if_ts_pipeline_is_winner(details: dict, model_name: str) -> None:
    """Check if ts pipeline is the winner one. It should be used before model store in the repo."""
    is_ts_metrics = 'ts_metrics' in details['entity']['status'].get('metrics', [{}])[0]
    if is_ts_metrics:
        summary = create_summary(details=details, scoring=None)
        summary.reset_index(inplace=True)
        vals = summary[summary['Pipeline Name'] == model_name]['Winner'].values
        if len(vals) == 0:
            raise WrongModelName(model_name)

        else:
            if not vals[0]:
                raise DiscardedModel(model_name)


def get_values_for_imputation_strategy(strategy, prediction_type, imputer_fill_value=None):
    values = {ImputationStrategy.MEAN: {'kb': 'mean',
                                        'ts': {'use_imputation': True,
                                               'imputer_list': ['Fill'],
                                               'imputer_fill_type': 'mean'}},
              ImputationStrategy.MEDIAN: {'kb': 'median',
                                          'ts': {'use_imputation': True,
                                                 'imputer_list': ['Fill'],
                                                 'imputer_fill_type': 'median'}},
              ImputationStrategy.MOST_FREQUENT: {'kb': 'most_frequent'},
              ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS: {'ts': {'use_imputation': True,
                                                                   'imputer_list': ['FlattenIterative', 'Linear', 'Cubic', 'Previous']}},
              ImputationStrategy.VALUE: {'ts': {'use_imputation': True,
                                                'imputer_list': ['Fill'],
                                                'imputer_fill_type': 'value',
                                                'imputer_fill_value': imputer_fill_value if imputer_fill_value else 0}},
              ImputationStrategy.FLATTEN_ITERATIVE: {'ts': {'use_imputation': True,
                                                            'imputer_list': ['FlattenIterative']}},
              ImputationStrategy.LINEAR: {'ts': {'use_imputation': True,
                                                 'imputer_list': ['Linear']}},
              ImputationStrategy.CUBIC: {'ts': {'use_imputation': True,
                                                'imputer_list': ['Cubic']}},
              ImputationStrategy.PREVIOUS: {'ts': {'use_imputation': True,
                                                   'imputer_list': ['Previous']}},
              ImputationStrategy.NEXT: {'ts': {'use_imputation': True,
                                               'imputer_list': ['Next']}},
              ImputationStrategy.NO_IMPUTATION: {'ts': {'use_imputation': False}}}

    if type(strategy) is list: # only for ts, checked earlier
        results = [get_values_for_imputation_strategy(x, prediction_type, imputer_fill_value) for x in strategy]

        def chose_strategies(res_elements):
            indexes = [results.index(el) for el in res_elements]
            return [strategy[i] for i in indexes]

        no_imputation_res = list(filter(lambda x: not x['use_imputation'], results))
        if any(no_imputation_res) and len(strategy) > len(no_imputation_res):
            raise InconsistentImputationListElements(chose_strategies(no_imputation_res))

        result = {'use_imputation': results[0]['use_imputation'],
                  'imputer_list': [x for r in results for x in r['imputer_list']]}

        fill_type_imputation_res = list(filter(lambda x: 'imputer_fill_type' in x, results))
        fill_value_imputation_res = list(filter(lambda x: 'imputer_fill_value' in x, results))

        if len(fill_type_imputation_res) > 1:
            raise InconsistentImputationListElements(chose_strategies(fill_type_imputation_res))
        elif len(fill_type_imputation_res) == 1:
            result['imputer_fill_type'] = fill_type_imputation_res[0]['imputer_fill_type']

        if len(fill_value_imputation_res) > 1:
            raise InconsistentImputationListElements(chose_strategies(fill_value_imputation_res))
        elif len(fill_value_imputation_res) == 1:
            result['imputer_fill_value'] = fill_type_imputation_res[0]['imputer_fill_value']

        return result

    v = values[strategy]

    if prediction_type == PredictionType.FORECASTING:
        if 'ts' not in v:
            l = ['ImputationStrategy.' + s.name for s in ImputationStrategy if values[s].get('ts')]
            raise StrategyIsNotApplicable(strategy, prediction_type, l)

        return v['ts']
    else:
        if 'kb' not in v:
            l = ['ImputationStrategy.' + s.name for s in ImputationStrategy if values[s].get('kb')]
            raise StrategyIsNotApplicable(strategy, prediction_type, l)

        return v['kb']


def translate_imputation_string_strategy_to_enum(strategy, prediction_type):
    if type(strategy) is list and prediction_type == PredictionType.FORECASTING:
        if 'Fill' in strategy:
            strategy.remove('Fill')
        return [translate_imputation_string_strategy_to_enum(x, prediction_type) for x in strategy]

    strategy = re.sub(r'(?<!^)(?=[A-Z])', '_', strategy).lower()

    return ImputationStrategy(strategy)


def translate_estimator_string_to_enum(estimator):
    algorithms = [ClassificationAlgorithms, RegressionAlgorithms, ForecastingAlgorithms]
    if type(estimator) is str:
        for alg in algorithms:
            alg_names = [a.value for a in alg]

            if estimator in alg_names:
                return alg(estimator)

            if not estimator.endswith('Estimator') and estimator + 'Estimator' in alg_names:
                return alg(estimator + 'Estimator')

            if estimator.endswith('Estimator') and estimator[:-9] in alg_names:
                return alg(estimator[:-9])

    return estimator


def translate_batched_estimator_string_to_enum(estimator):
    algorithms = [BatchedClassificationAlgorithms, BatchedRegressionAlgorithms]
    if type(estimator) is str:
        for alg in algorithms:
            alg_names = [a.value for a in alg]

            if estimator in alg_names:
                return alg(estimator)

            if estimator.endswith('Estimator)') and estimator[:-10] + ")" in alg_names:
                return alg(estimator[:-10] + ")")

    return estimator


def convert_dataframe_to_fields_values_payload(df: 'DataFrame', return_values_only=False) -> Dict[str, List]:
    if isinstance(df, pd.DataFrame):

        data = df.where(pd.notnull(df), None)
        data.fillna(json.dumps(float("nan")), inplace=True)
        values = data.values

        # note: scoring endpoint could not recognize NaN values, convert NaN to None
        try:
            values[pd.isnull(values)] = None

        # note: above code fails when there is no null values in a dataframe
        except TypeError:
            pass
        # --- end note

        values = values.tolist()
        # --- end note

        if return_values_only:
            return {'values': values}
        else:
            fields = df.columns.tolist()
            return {'fields': fields, 'values': values}
    else:
        raise TypeError('Passed parameter should be of type pandas.DataFrame.')


def get_autoai_run_id_from_experiment_metadata(experiment_metadata: dict) -> str:
    if experiment_metadata:
        training_result_reference = experiment_metadata.get('training_result_reference')

        from ibm_watson_machine_learning.helpers import S3Location, AssetLocation
        if isinstance(training_result_reference.location, (S3Location, AssetLocation)):  # cloud
            run_id = training_result_reference.location._training_status.split('/')[-2]
        else:
            run_id = training_result_reference.location.path.split('/')[-3]
    return run_id
