#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

__all__ = [
    'save_metadata_for_ui',
    'save_computed_pipelines_for_ui',
    'save_experiment_for_ui'
]


import json
import os
from datetime import datetime as dt
from typing import Dict, List, TYPE_CHECKING
from uuid import uuid4

import pandas as pd
import ibm_watson_machine_learning._wrappers.requests as requests

from .watson_studio import get_project

if TYPE_CHECKING:
    from ibm_watson_machine_learning import APIClient


def save_metadata_for_ui(wml_pipeline_config: Dict, training_details: Dict) -> None:
    """Prepare AutoAI experiment metadata file for UI.

    :param wml_pipeline_config: dictionary with WML pipeline configuration
    :type wml_pipeline_config: dict

    :param training_details: dictionary with training details
    :type training_details: dict
    """
    experiment_id = training_details['entity']['results_reference']['location']['path'].split('/')[-2].split(
        'auto_ml.')[-1]
    parameters = wml_pipeline_config['entity']['document']['pipelines'][0]['nodes'][0]['parameters']
    learning_type = parameters['optimization']['learning_type']
    distinct_values = training_details['entity']['status'].get(
        'metrics', [{'context': {'classes': []}}])[0]['context']['classes']

    # note: mapping classification types to UI types
    if learning_type == 'classification':
        if len(distinct_values) == 2:
            learning_type = 'binary'
        else:
            learning_type = 'multiclass'
    # --- end note

    project = get_project()
    project_id = project.get_metadata()["metadata"]["guid"]

    training_data_references = training_details['entity']['training_data_references']

    creation_time = training_details['metadata']['created_at']
    timestamp = int(f"{int(dt.timestamp(dt.strptime(creation_time[:-5].replace('T', ' '), '%Y-%m-%d %H:%M:%S')))}000")

    # note: read associated WMLS data with project
    with open(os.path.join(os.path.abspath('.'), project_id, 'project.json'), 'r') as f:
        data = json.load(f)
    # --- end note

    service_instance_id = data['compute'][0]['guid']

    template = {
        "metadata": {
            "name": f"{wml_pipeline_config['metadata']['name']} SDK",
            "asset_type": "auto_ml",
            "origin_country": "us",
            "created_at": creation_time,
            "created": timestamp,
            "project_id": project_id,
            "asset_id": f"auto_ml.{experiment_id}",
            "sandbox_id": project_id,
            "last_modified": creation_time,
            "asset_attributes": [
                "auto_ml"
            ],
            "usage": {
                "last_update_time": timestamp,
                "last_updated_at": creation_time
            }
        },
        "entity": {
            "auto_ml": {
                "wml": {
                    "service_instance_id": service_instance_id,
                    "pipeline": {
                        "metadata": {
                            "name": wml_pipeline_config['metadata']['name'],
                            "guid": wml_pipeline_config['metadata']['guid'],
                            "tags": [
                                f"dsx-project.{project_id}"
                            ],
                            "id": wml_pipeline_config['metadata']['id'],
                            "modified_at": wml_pipeline_config['metadata']['modified_at'],
                            "created_at": wml_pipeline_config['metadata']['created_at'],
                            "owner": wml_pipeline_config['metadata']['owner'],
                            "href": wml_pipeline_config['metadata']['href'],
                            "space_id": wml_pipeline_config['metadata']['space_id']
                        },
                        "entity": {
                            "tags": [
                                {
                                    "value": f"dsx-project.{project_id}",
                                    "description": "guid of associated DSX project"
                                }
                            ],
                            "space": wml_pipeline_config['entity']['space'],
                            "name": wml_pipeline_config['entity']['name']
                        }
                    },
                    "training_definition": {
                        "metadata": {
                            "created_at": training_details['metadata']['created_at'],
                            "guid": training_details['metadata']['guid'],
                            "href": training_details['metadata']['href'],
                            "id": training_details['metadata']['id'],
                            "space_id": training_details['entity']['space_id'],
                            "tags": [
                                f"dsx-project.{project_id}"
                            ]
                        },
                        "entity": {
                            "pipeline": training_details['entity']['pipeline'],
                            "results_reference": training_details['entity']['results_reference'],
                            "space": training_details['entity']['space'],
                            "space_id": training_details['entity']['space_id'],
                            "tags": [
                                {
                                    "description": "guid of associated DSX project",
                                    "value": f"dsx-project.{project_id}"
                                }
                            ],
                            "training_data_references": training_data_references
                        }
                    }
                },
                "status": "completed",
                "learning_type": learning_type
            }
        },
        "attachments": [
            {
                "asset_type": "auto_ml",
                "name": "config",
                "description": f"config attachment for auto_ml asset auto_ml.{experiment_id}",
                "object_key": f"auto_ml.{experiment_id}/attachments/config_auto_ml.{experiment_id}",
                "object_key_is_read_only": True,
                "created_at": creation_time,
                "id": f"{uuid4()}",
                "is_managed": False,
                "is_referenced": True,
                "is_remote": False
            },
            {
                "asset_type": "auto_ml",
                "name": "referenced",
                "description": f"referenced attachment for auto_ml asset auto_ml.{experiment_id}",
                "object_key": f"auto_ml.{experiment_id}/attachments/referenced_auto_ml.{experiment_id}",
                "object_key_is_read_only": True,
                "created_at": creation_time,
                "id": f"{uuid4()}",
                "is_managed": False,
                "is_referenced": True,
                "is_remote": False
            }
        ]
    }

    file_name = os.path.join(os.path.abspath('.'), project_id, 'assets', '.METADATA', f"auto_ml.{experiment_id}.json")

    with open(file_name, 'w') as f:
        f.write(json.dumps(template))


def save_computed_pipelines_for_ui(wml_client: 'APIClient', run_id: str) -> None:
    """Download all AutoAI experiments files into the local storage for UI usage.

    :param wml_client: WML API Client
    :type wml_client: APIClient

    :param run_id: ID of the training
    :type run_id: str
    """
    print('Saving all computed pipelines locally for WSD...')

    models_outputs = (
        'cognito_output',
        'hpo_c_output',
        'hpo_d_output',
        'pre_hpo_d_output'
    )

    project = get_project()
    project_id = project.get_metadata()["metadata"]["guid"]

    training_details = wml_client.training.get_details(run_id, _internal=True)

    # note: prepare needed information to fetch data
    path = training_details['entity']['results_reference']['location']['path'].split('assets/auto_ml/')[-1]
    experiment_id = path.split('/')[0].split('auto_ml.')[-1]
    remote_path = f"{path}/{run_id}/data/automl/"
    # --- end note

    # note: prepare local path within WSD
    local_path = os.path.join(os.path.abspath('.'), project_id, 'assets', 'auto_ml', f"auto_ml.{experiment_id}",
                              'wml_data', run_id, 'data', 'automl')
    # --- end note

    # note: download each model output (models, jsons etc.)
    for output in models_outputs:
        generated_path = f"{remote_path}{output}"

        response = requests.get(
            url=f"{wml_client.service_instance._href_definitions.get_wsd_model_attachment_href()}auto_ml/{generated_path}",
            headers=wml_client._get_headers(),
            params={'space_id': wml_client.default_space_id,
                    'flat': 'true'})

        response_with_paths = wml_client.data_assets._handle_response(200, u'getting data from WMLS', response)

        for resource in response_with_paths['resources']:
            pipeline_dir = resource['path'].split('/')[-2]
            file_name = resource['path'].split('/')[-1]

            generated_local_path = os.path.join(local_path, output, pipeline_dir)
            create_dir(generated_local_path)

            response_with_file = requests.get(
                url=f"{wml_client.service_instance._href_definitions.get_wsd_model_attachment_href()}auto_ml/{generated_path}/{pipeline_dir}/{file_name}",
                headers=wml_client._get_headers(),
                params={'space_id': wml_client.default_space_id})

            with open(os.path.join(generated_local_path, file_name), 'wb') as f:
                f.write(response_with_file.content)
    # --- end note
    print("Saved.")


def save_experiment_for_ui(wml_pipeline_config: Dict,
                           training_details: Dict,
                           training_data_file_path: str,
                           asset_id: str,
                           asset_name: str) -> None:
    """Prepare AutoAI experiment configuration from wml pipeline config for WSD UI reuse.

    :param wml_pipeline_config: dictionary with WML pipeline configuration
    :type wml_pipeline_config: dict

    :param training_details: dictionary with training details
    :type training_details: dict

    :param training_data_file_path: local path to the training data asset
    :type training_data_file_path: str

    :param asset_id: ID of the data asset on WSD
    :type asset_id: str

    :param asset_name: name of the data asset on WSD
    :type asset_name: str
    """
    print('Saving experiment metadata locally for WSD...')

    experiment_id = training_details['entity']['results_reference']['location']['path'].split('/')[-2].split(
        'auto_ml.')[-1]

    config = prepare_config_auto_ml(wml_pipeline_config, training_details, training_data_file_path,
                                    asset_id, asset_name)
    reference = prepare_reference_auto_ml(wml_pipeline_config, training_details)

    project = get_project()
    project_id = project.get_metadata()["metadata"]["guid"]

    config_path = os.path.join(os.path.abspath('.'), project_id, 'assets',
                               'auto_ml', f"auto_ml.{experiment_id}", 'attachments')
    config_file_name = f"config_auto_ml.{experiment_id}"
    reference_path = os.path.join(os.path.abspath('.'), project_id, 'assets',
                                  'auto_ml', f"auto_ml.{experiment_id}", 'attachments')
    reference_file_name = f"referenced_auto_ml.{experiment_id}"

    # note: make sure that directories are created
    create_dir(config_path)
    create_dir(reference_path)
    # --- end note

    with open(os.path.join(config_path, config_file_name), 'w') as f:
        f.write(decode_for_ui(config))

    with open(os.path.join(reference_path, reference_file_name), 'w') as f:
        f.write(decode_for_ui(reference))

    print("Saved.")


def create_dir(path: str) -> None:
    """Create entire path of directories if missing."""
    try:
        os.mkdir(path)

    except FileExistsError:
        pass

    except FileNotFoundError:
        create_dir(f'{os.sep}'.join(path.split(os.sep)[:-1]))
        create_dir(path)


def decode_for_ui(data: Dict) -> str:
    """Convert / decode dictionary with data for UI accessible format."""
    _dict = {
        "name": data['name'],
        "body": 'placeholder'
    }
    _dict = json.dumps(_dict)
    _body = json.dumps(data['body']).replace(r'"', r'\"')
    _dict = _dict.replace('placeholder', _body)

    return _dict


def prepare_config_auto_ml(wml_pipeline_config: Dict,
                           training_details: Dict,
                           training_data_file_path: str,
                           asset_id: str,
                           asset_name: str) -> Dict:
    """Prepare AutoAI experiment configuration from wml pipeline config for WSD UI reuse.

    :param wml_pipeline_config: dictionary with WML pipeline configuration
    :type wml_pipeline_config: dict

    :param training_details: dictionary with training details
    :type training_details: dict

    :param training_data_file_path: local path to the training data asset
    :type training_data_file_path: str

    :param asset_id: ID of the data asset on WSD
    :type asset_id: str

    :param asset_name: name of the data asset on WSD
    :type asset_name: str

    :return: dictionary with AutoAI experiment configuration for WSD UI
    :rtype: dict
    """
    experiment_id = training_details['entity']['results_reference']['location']['path'].split('/')[-2].split(
        'auto_ml.')[-1]
    parameters = wml_pipeline_config['entity']['document']['pipelines'][0]['nodes'][0]['parameters']

    try:
        spec = wml_pipeline_config['entity']['document']['runtimes'][0]['app_data']['wml_data']['hardware_spec']['name']

    except KeyError:
        spec = wml_pipeline_config['entity']['document']['runtimes'][0][
            'app_data']['wml_data']['runtime_spec_v4']['compute']['name']

    name = os.path.join('auto_ml', f"auto_ml.{experiment_id}", 'attachments', f"config_auto_ml.{experiment_id}")
    label_col = parameters['optimization']['label']
    learning_type = parameters['optimization']['learning_type']
    scorer_for_ranking = parameters['optimization']['scorer_for_ranking']
    compute_plan = spec.lower()
    holdout_param = parameters['optimization']['holdout_param']
    daub_include_only_estimators = parameters['optimization']['daub_include_only_estimators']
    cognito_transform_names = parameters['optimization']['cognito_transform_names']
    max_num_daub_ensembles = int(parameters['optimization']['max_num_daub_ensembles'])
    distinct_values = training_details['entity']['status'].get(
        'metrics', [{'context': {'classes': []}}])[0]['context']['classes']

    # note: mapping classification types to UI types
    if learning_type == 'classification':
        if len(distinct_values) == 2:
            learning_type = 'binary'
        else:
            learning_type = 'multiclass'
    # --- end note

    # note: asset information part
    data = pd.read_csv(training_data_file_path, nrows=2)
    label_col_idx = data.columns.to_list().index(label_col)
    total_columns = len(data.columns)
    asset_size = os.stat(training_data_file_path).st_size
    # --- end note

    template = {
        "name": name,
        "body": {
            "target": {
                "label_col": label_col,
                "label_col_idx": label_col_idx,
                "learning_type": learning_type,
                "metric": scorer_for_ranking,
                "main_table_id": asset_id,
                "distinct_values": distinct_values
            },
            "source_assets": [
                {
                    "asset_id": asset_id,
                    "asset_type": "data_asset",
                    "asset_name": asset_name,
                    "asset_size": asset_size,
                    "total_columns": total_columns
                }
            ],
            "obm": {},
            "kb": {
                "run_settings": {
                    "compute_plan": compute_plan,
                    "holdout_param": holdout_param,
                    "daub_include_only_estimators": daub_include_only_estimators,
                    "cognito_transform_names": cognito_transform_names,
                    "max_num_daub_ensembles": max_num_daub_ensembles
                }
            },
            "column_settings": {}
        }
    }

    return template


def prepare_reference_auto_ml(wml_pipeline_config: Dict, training_details: Dict) -> Dict:
    """Prepare AutoAI experiment configuration from wml pipeline config for WSD UI reuse.

    :param wml_pipeline_config: dictionary with WML pipeline configuration
    :type wml_pipeline_config: dict

    :param training_details: dictionary with training details
    :type training_details: dict

    :return: dictionary with AutoAI experiment configuration for WSD UI
    :rtype: dict
    """

    completed_at = training_details['entity']['status']['completed_at']
    last_modified = training_details['metadata']['modified_at']
    created_at = training_details['metadata']['created_at']
    guid = training_details['metadata']['guid']
    experiment_id = training_details['entity']['results_reference']['location']['path'].split('/')[-2].split(
        'auto_ml.')[-1]
    name = os.path.join('auto_ml', f"auto_ml.{experiment_id}", 'attachments', f"referenced_auto_ml.{experiment_id}")
    message_log = extract_message_log(training_details)
    estimators = extract_estimators(wml_pipeline_config, training_details)

    template = {
        "name": name,
        "body": {
            "version": 2,
            "trainingGuid": guid,
            "message": message_log[-1],
            "state": "completed",
            "pipelines": extract_pipelines(training_details),
            "estimators": estimators,
            "created_at": created_at,
            "preprocessing_step": "AUTO_AI_STEP_COMPLETED",
            "message_log": message_log,
            "completed_at": completed_at,
            "last_modified": last_modified
        }
    }

    return template


def extract_pipelines(training_details: Dict) -> Dict:
    """Extract information about computed pipelines from training details.

    :param training_details: dictionary with training details
    :type training_details: dict

    :return: dictionary with AutoAI pipelines details
    :rtype: dict
    """
    pipelines = {}
    stages = training_details['entity']['status'].get('metrics', [])

    current_pipeline_name = None
    for stage in stages[::-1]:
        name = stage['context']['intermediate_model']['name']

        if current_pipeline_name != name:
            current_pipeline_name = name

            pipelines[name] = stage

    sorted_pipelines = {}
    for key, value in sorted(pipelines.items()):
        sorted_pipelines[key] = value

    return sorted_pipelines


def extract_estimators(wml_pipeline_config: Dict, training_details: Dict) -> Dict:
    """Extract information about estimators from training details.

    :param wml_pipeline_config: dictionary with WML pipeline configuration
    :type wml_pipeline_config: dict

    :param training_details: dictionary with training details
    :type training_details: dict

    :return: dictionary with estimators and scores
    :rtype: dict
    """
    estimators = {}
    parameters = wml_pipeline_config['entity']['document']['pipelines'][0]['nodes'][0]['parameters']
    scorer_for_ranking = parameters['optimization']['scorer_for_ranking']
    stages = training_details['entity']['status'].get('metrics', [])

    current_pipeline_estimator = None
    for stage in stages[::-1]:
        estimator_name = stage['context']['intermediate_model']['pipeline_nodes'][-1]
        score = stage['ml_metrics'][f'training_{scorer_for_ranking}']

        if current_pipeline_estimator != estimator_name:
            current_pipeline_estimator = estimator_name

            estimators[estimator_name] = {'score': score}

    return estimators


def extract_message_log(training_details: Dict) -> List[Dict]:
    """Create message log for training.

    :param training_details: dictionary with training details
    :type training_details: dict

    :return: list of dictionaries with messages
    :rtype: list[dict]
    """
    steps = {'DAUB': 'feature_engineering',
             'hpo_d': 'model_optimization',
             'cognito': 'transforming_data',
             'hpo_c': 'optimizing_hyperparams'}

    message_log = []
    stages = training_details['entity']['status'].get('metrics', [])

    timestamp = 0
    model_name = 'P4'
    duration = 1

    for stage in stages:
        model_name = stage['context']['intermediate_model']['name']
        timestamp = int(f"{int(dt.timestamp(dt.strptime(training_details['metadata']['created_at'][:-5].replace('T', ' '), '%Y-%m-%d %H:%M:%S')))}000")
        step = steps[stage['context']['intermediate_model']['composition_steps'][-1]]
        duration = stage['context']['intermediate_model']['duration']

        message_start = {
            "timestamp": timestamp,
            "text": f"Train_log_msg_{step}_start",
            "running_pipeline": model_name,
            "level": "info"
        }

        message_end = {
            "timestamp": timestamp + duration,
            "text": f"Train_log_msg_{step}_complete",
            "running_pipeline": model_name,
            "level": "info"
        }

        message_log.append(message_start)
        message_log.append(message_end)

    message_log.append(
        {
            "timestamp": timestamp + duration,
            "text": "Train_log_msg_data_training_complete",
            "running_pipeline": model_name,
            "level": "info"
        }
    )
    message_log.append(
        {
            "timestamp": timestamp + duration,
            "text": "autoai execution completed",
            "running_pipeline": model_name,
            "level": "info"
        }
    )
    message_log.append(
        {
            "timestamp": timestamp + duration,
            "text": "Train_log_msg_data_training_complete",
            "running_pipeline": model_name,
            "level": "info"
        }
    )
    message_log.append(
        {
            "timestamp": timestamp + duration,
            "text": "Train_log_msg_pipeline_created",
            "running_pipeline": model_name,
            "level": "info"
        }
    )
    message_log.append(
        {
            "timestamp": timestamp + duration,
            "text": "Train_log_msg_training_completed",
            "running_pipeline": model_name,
            "level": "info"
        }
    )

    return message_log
