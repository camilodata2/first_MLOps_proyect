#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
import json
import re
import logging
from lomond import WebSocket
from ibm_watson_machine_learning.utils import print_text_header_h1, print_text_header_h2, TRAINING_RUN_DETAILS_TYPE, group_metrics, StatusLogger
import time
from ibm_watson_machine_learning.metanames import TrainingConfigurationMetaNames, TrainingConfigurationMetaNamesCp4d30
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
from ibm_watson_machine_learning.href_definitions import is_uid
from warnings import warn
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_boto3.exceptions import Boto3Error

from ibm_watson_machine_learning.messages.messages import Messages

logging.getLogger('lomond').setLevel(logging.CRITICAL)


class Training(WMLResource):
    """Train new models."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP
        self.ConfigurationMetaNames = TrainingConfigurationMetaNames()
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            self.ConfigurationMetaNames = TrainingConfigurationMetaNamesCp4d30()

    @staticmethod
    def _is_training_uid(s):
        res = re.match('p\-[a-zA-Z0-9\-\_]+', s)
        return res is not None

    @staticmethod
    def _is_training_url(s):
        res = re.match('\/v3\/models\/p\-[a-zA-Z0-9\-\_]+', s)
        return res is not None

    def _is_model_definition_url(self, s):
        res = re.match('\/v2\/assets\/p\-[a-zA-Z0-9\-\_]+', s)
        return res is not None

    def get_status(self, training_uid):
        """Get the status of a training created.

        :param training_uid: training UID
        :type training_uid: str

        :return: training_status
        :rtype: dict

        **Example**

        .. code-block:: python

            training_status = client.training.get_status(training_uid)
        """
        Training._validate_type(training_uid, 'training_uid', str, True)

        details = self.get_details(training_uid, _internal=True)

        if details is not None:
            return WMLResource._get_required_element_from_dict(details, u'details', [u'entity', u'status'])
        else:
            raise WMLClientError(u'Getting trained model status failed. Unable to get model details for training_uid: \'{}\'.'.format(training_uid))

    def get_details(self, training_uid=None, limit=None, asynchronous=False, get_all=False, training_type=None,
                    state=None, tag_value=None, training_definition_id=None, _internal=False):
        """Get metadata of training(s). If training_uid is not specified returns all model spaces metadata.

        :param training_uid: Unique Id of training
        :type training_uid: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all:  if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional
        :param training_type: filter the fetched list of trainings based on training type ["pipeline" or "experiment"]
        :type training_type: str, optional
        :param state: filter the fetched list of training based on their state:
            [`queued`, `running`, `completed`, `failed`]
        :type state: str, optional
        :param tag_value: filter the fetched list of training based on ther tag value
        :type tag_value: str, optional
        :param training_definition_id: filter the fetched trainings which are using the given training definition
        :type training_definition_id: str, optional


        :return: metadata of training(s)
        :rtype:
          - **dict** - if training_uid is not None
          - **{"resources": [dict]}** - if training_uid is None

        **Examples**

        .. code-block:: python

            training_run_details = client.training.get_details(training_uid)
            training_runs_details = client.training.get_details()
            training_runs_details = client.training.get_details(limit=100)
            training_runs_details = client.training.get_details(limit=100, get_all=True)
            training_runs_details = []
            for entry in client.training.get_details(limit=100, asynchronous=True, get_all=True):
                training_runs_details.extend(entry)
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(training_uid, 'training_uid', str, False)

        url = self._client.service_instance._href_definitions.get_trainings_href()

        if training_uid is None:
            query_params = {
                param_name: param_value
                for param_name, param_value in (
                    ('type', training_type),
                    ('state', state),
                    ('tag.value', tag_value),
                    ('training_definition_id', training_definition_id)
                ) if param_value is not None
            }
            # note: If query params is an empty dict convert it back to None value
            query_params = query_params if query_params != {} else None

            return self._get_artifact_details(url, training_uid, limit, 'trained models', _async=asynchronous,
                                              _all=get_all, query_params=query_params)
        else:
            details = self._get_artifact_details(url, training_uid, limit, 'trained models')

            if not _internal and details.get('entity', {}).get('status', {}).get('feature_engineering_components', {}).get('obm'):
                if self._client.ICP_46 or self._client.ICP_47:
                    raise WMLClientError(Messages.get_message(message_id="obm_removal_message_cpd"))
                elif self._client.ICP:
                    print(Messages.get_message(message_id="obm_deprecation_message_cpd"))
                else:
                    raise WMLClientError(Messages.get_message(message_id="obm_removal_message_cloud"))

            return details

    @staticmethod
    def get_href(training_details):
        """Get training href from training details.

        :param training_details:  metadata of the training created
        :type training_details: dict

        :return: training href
        :rtype: str

        **Example**

        .. code-block:: python

            training_details = client.training.get_details(training_uid)
            run_url = client.training.get_href(training_details)
        """

        Training._validate_type(training_details, u'training_details', object, True)
        if 'id' in training_details.get('metadata'):
            training_id = WMLResource._get_required_element_from_dict(training_details, u'training_details',
                                                                      [u'metadata', u'id'])
            return "/ml/v4/trainings/"+training_id
        else:
            Training._validate_type_of_details(training_details, TRAINING_RUN_DETAILS_TYPE)
            return WMLResource._get_required_element_from_dict(training_details, u'training_details', [u'metadata', u'href'])

    @staticmethod
    def get_uid(training_details):
        """This method is deprecated, please use ``get_id()`` instead."""

        warn("This method is deprecated, please use get_id()")
        print("This method is deprecated, please use get_id()")

        try:
            Training._validate_type(training_details, u'training_details', object, True)
            return WMLResource._get_required_element_from_dict(training_details, u'training_details',
                                                               [u'metadata', u'guid'])
        except Exception:
            return Training.get_id(training_details)

    @staticmethod
    def get_id(training_details):
        """Get training id from training details.

        :param training_details: metadata of the training created
        :type training_details: dict

        :return: Unique id of training
        :rtype: str

        **Example**

        .. code-block:: python

            training_details = client.training.get_details(training_id)
            training_id = client.training.get_id(training_details)

        """

        Training._validate_type(training_details, u'training_details', object, True)
        return WMLResource._get_required_element_from_dict(training_details, u'training_details',
                                                           [u'metadata', u'id'])

    def run(self, meta_props, asynchronous=True):
        """Create a new Machine Learning training.

        :param meta_props:  metadata of the training configuration. To see available meta names use:

            .. code-block:: python

                client.training.ConfigurationMetaNames.show()

        :type meta_props: str
        :param asynchronous:
            * `True` - training job is submitted and progress can be checked later
            * `False` - method will wait till job completion and print training stats
        :type asynchronous: bool, optional

        :return: metadata of the training created
        :rtype: dict

        .. note::

            You can provide one of the below values for training:
             * client.training.ConfigurationMetaNames.EXPERIMENT
             * client.training.ConfigurationMetaNames.PIPELINE
             * client.training.ConfigurationMetaNames.MODEL_DEFINITION

        **Examples**

        Example meta_props for Training run creation in IBM Cloud Pak® for Data version 3.0.1 or above:

        .. code-block:: python

            metadata = {
                client.training.ConfigurationMetaNames.NAME: 'Hand-written Digit Recognition',
                client.training.ConfigurationMetaNames.DESCRIPTION: 'Hand-written Digit Recognition Training',
                client.training.ConfigurationMetaNames.PIPELINE: {
                    "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
                    "rev": "12",
                    "model_type": "string",
                    "data_bindings": [
                        {
                            "data_reference_name": "string",
                            "node_id": "string"
                        }
                    ],
                    "nodes_parameters": [
                        {
                            "node_id": "string",
                            "parameters": {}
                        }
                    ],
                    "hardware_spec": {
                        "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
                        "rev": "12",
                        "name": "string",
                        "num_nodes": "2"
                    }
                },
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [{
                    'type': 's3',
                    'connection': {},
                    'location': {'href': 'v2/assets/asset1233456'},
                    'schema': { 'id': 't1', 'name': 'Tasks', 'fields': [ { 'name': 'duration', 'type': 'number' } ]}
                }],
                client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    'id' : 'string',
                    'connection': {
                        'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
                        'access_key_id': '***',
                        'secret_access_key': '***'
                    },
                    'location': {
                        'bucket': 'wml-dev-results',
                        'path' : "path"
                    }
                    'type': 's3'
                }
            }

        Example meta_prop values for training run creation in other version:

        .. code-block:: python

            metadata = {
                client.training.ConfigurationMetaNames.NAME: 'Hand-written Digit Recognition',
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [{
                    'connection': {
                        'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
                        'access_key_id': '***',
                        'secret_access_key': '***'
                    },
                    'source': {
                        'bucket': 'wml-dev',
                    }
                    'type': 's3'
                }],
                client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    'connection': {
                        'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
                        'access_key_id': '***',
                        'secret_access_key': '***'
                    },
                    'target': {
                        'bucket': 'wml-dev-results',
                    }
                    'type': 's3'
                },
                client.training.ConfigurationMetaNames.PIPELINE_UID : "/v4/pipelines/<PIPELINE-ID>"
            }
            training_details = client.training.run(definition_uid, meta_props=metadata)
            training_uid = client.training.get_id(training_details)

        Example of a Federated Learning training job:

        .. code-block:: python

            aggregator_metadata = {
                wml_client.training.ConfigurationMetaNames.NAME: 'Federated_Learning_Tensorflow_MNIST',
                wml_client.training.ConfigurationMetaNames.DESCRIPTION: 'MNIST digit recognition with Federated Learning using Tensorflow',
                wml_client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [],
                wml_client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    'type': results_type,
                    'name': 'outputData',
                    'connection': {},
                    'location': { 'path': '/projects/' + PROJECT_ID + '/assets/trainings/'}
                },
                wml_client.training.ConfigurationMetaNames.FEDERATED_LEARNING: {
                    'model': {
                        'type': 'tensorflow',
                        'spec': {
                        'id': untrained_model_id
                    },
                    'model_file': untrained_model_name
                },
                'fusion_type': 'iter_avg',
                'metrics': 'accuracy',
                'epochs': 3,
                'rounds': 10,
                'remote_training' : {
                    'quorum': 1.0,
                    'max_timeout': 3600,
                    'remote_training_systems': [ { 'id': prime_rts_id }, { 'id': nonprime_rts_id} ]
                },
                'hardware_spec': {
                    'name': 'S'
                },
                'software_spec': {
                    'name': 'runtime-22.1-py3.9'
                }
            }

            aggregator = wml_client.training.run(aggregator_metadata, asynchronous=True)
            aggregator_id = wml_client.training.get_id(aggregator)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(meta_props, 'meta_props', object, True)
        Training._validate_type(asynchronous, 'asynchronous', bool, True)

        self.ConfigurationMetaNames._validate(meta_props)
        training_configuration_metadata = {
            u'training_data_references': meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES],
            u'results_reference': meta_props[self.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE]
        }

        if self.ConfigurationMetaNames.TEST_DATA_REFERENCES in meta_props:
            training_configuration_metadata['test_data_references'] = meta_props[self.ConfigurationMetaNames.TEST_DATA_REFERENCES]

        if self.ConfigurationMetaNames.TEST_OUTPUT_DATA in meta_props:
            training_configuration_metadata['test_output_data'] = meta_props[self.ConfigurationMetaNames.TEST_OUTPUT_DATA]

        if self.ConfigurationMetaNames.TAGS in meta_props:
            training_configuration_metadata["tags"] = meta_props[self.ConfigurationMetaNames.TAGS]

        # TODO remove when training service starts copying such data on their own

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            training_configuration_metadata["name"] = meta_props[self.ConfigurationMetaNames.NAME]
            training_configuration_metadata["description"] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

            if self.ConfigurationMetaNames.PIPELINE in meta_props:
                training_configuration_metadata["pipeline"] = meta_props[self.ConfigurationMetaNames.PIPELINE]
            if self.ConfigurationMetaNames.EXPERIMENT in meta_props:
                training_configuration_metadata['experiment'] = meta_props[self.ConfigurationMetaNames.EXPERIMENT]
            if self.ConfigurationMetaNames.MODEL_DEFINITION in meta_props:
                training_configuration_metadata['model_definition'] = \
                    meta_props[self.ConfigurationMetaNames.MODEL_DEFINITION]
            if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                training_configuration_metadata["space_id"] = meta_props[self.ConfigurationMetaNames.SPACE_UID]

            if self._client.default_space_id is None and self._client.default_project_id is None:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")
            else:
                if self._client.default_space_id is not None:
                    training_configuration_metadata['space_id'] = self._client.default_space_id
                elif self._client.default_project_id is not None:
                    training_configuration_metadata['project_id'] = self._client.default_project_id

            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                if self.ConfigurationMetaNames.FEDERATED_LEARNING in meta_props:
                    training_configuration_metadata['federated_learning'] = \
                        meta_props[self.ConfigurationMetaNames.FEDERATED_LEARNING]
        else:
            if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
                training_configuration_metadata["pipeline"] = {
                    "href": "/v4/pipelines/"+meta_props[self.ConfigurationMetaNames.PIPELINE_UID]
                }
                if self.ConfigurationMetaNames.PIPELINE_DATA_BINDINGS in meta_props:
                    training_configuration_metadata["pipeline"]["data_bindings"] = meta_props[self.ConfigurationMetaNames.PIPELINE_DATA_BINDINGS]
                if self.ConfigurationMetaNames.PIPELINE_NODE_PARAMETERS in meta_props:
                    training_configuration_metadata["pipeline"]["nodes_parameters"] = meta_props[
                        self.ConfigurationMetaNames.PIPELINE_NODE_PARAMETERS]
                if self.ConfigurationMetaNames.PIPELINE_MODEL_TYPE in meta_props:
                    training_configuration_metadata["pipeline"]["model_type"] = meta_props[
                        self.ConfigurationMetaNames.PIPELINE_MODEL_TYPE]
            if self.ConfigurationMetaNames.EXPERIMENT_UID in meta_props:
                training_configuration_metadata["experiment"] = {
                    "href": "/v4/experiments/" + meta_props[self.ConfigurationMetaNames.EXPERIMENT_UID]
                }

            if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props:
                if self._client.CAMS:
                    type_uid = self._check_if_lib_or_def(meta_props[self.ConfigurationMetaNames.TRAINING_LIB_UID])
                    training_configuration_metadata["training_lib"] = {"href" : type_uid}
                else:
                    training_configuration_metadata["training_lib"]["href"] = {"href" : "/v4/libraries/" + meta_props[self.ConfigurationMetaNames.TRAINING_LIB_UID]}

                if self.ConfigurationMetaNames.COMMAND not in meta_props or self.ConfigurationMetaNames.TRAINING_LIB_RUNTIME_UID not in meta_props:
                    raise WMLClientError(u'Invalid input. command, runtime are mandatory parameter for training_lib')
                training_configuration_metadata["training_lib"].update({"command":meta_props[self.ConfigurationMetaNames.COMMAND]})
                training_configuration_metadata["training_lib"].update({"runtime": {"href" : "/v4/runtimes/"+meta_props[self.ConfigurationMetaNames.TRAINING_LIB_RUNTIME_UID]}})
                if self.ConfigurationMetaNames.TRAINING_LIB_MODEL_TYPE in meta_props:
                    training_configuration_metadata["training_lib"].update({"model_type":  meta_props[self.ConfigurationMetaNames.TRAINING_LIB_MODEL_TYPE]})
                if self.ConfigurationMetaNames.COMPUTE in meta_props:
                    training_configuration_metadata["training_lib"].update({"compute": meta_props[self.ConfigurationMetaNames.COMPUTE]})
                if self.ConfigurationMetaNames.TRAINING_LIB_PARAMETERS in meta_props:
                    training_configuration_metadata["training_lib"].update({"parameters": meta_props[self.ConfigurationMetaNames.TRAINING_LIB_PARAMETERS]})

            if self.ConfigurationMetaNames.TRAINING_LIB in meta_props:
                training_configuration_metadata["training_lib"] =  meta_props[self.ConfigurationMetaNames.TRAINING_LIB]
                # for model_definition asset - command, href and runtime are mandatory
                if self._is_model_definition_url(meta_props[self.ConfigurationMetaNames.TRAINING_LIB]['href']) is False:
                    if ('command' not in meta_props[self.ConfigurationMetaNames.TRAINING_LIB].keys() or
                            'runtime' not in meta_props[self.ConfigurationMetaNames.TRAINING_LIB].keys()):
                        raise WMLClientError(u'Invalid input. command, href, runtime are mandatory parameter for training_lib')

            if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                training_configuration_metadata["space"] = {
                    "href": "/v4/spaces/"+meta_props[self.ConfigurationMetaNames.SPACE_UID]
                }
            if self._client.CAMS:
                if self._client.default_space_id is not None:
                    training_configuration_metadata['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
                elif self._client.default_project_id is not None:
                    training_configuration_metadata['project'] = {'href': "/v2/projects/" + self._client.default_project_id}
                else:
                    raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        train_endpoint = self._client.service_instance._href_definitions.get_trainings_href()
        if not self._ICP:
            if self._client.CLOUD_PLATFORM_SPACES:
                params = self._client._params()
                if 'space_id' in params.keys():
                    params.pop('space_id')
                if 'project_id' in params.keys():
                    params.pop('project_id')
                response_train_post = requests.post(train_endpoint, json=training_configuration_metadata,
                                                    params= params,
                                                    headers=self._client._get_headers())
            else:
             response_train_post = requests.post(train_endpoint, json=training_configuration_metadata,
                                                headers=self._client._get_headers())

        else:
            if self._client.ICP_PLATFORM_SPACES:
                params = self._client._params()
                if 'space_id' in params.keys():
                    params.pop('space_id')
                if 'project_id' in params.keys():
                    params.pop('project_id')
                if 'userfs' in params.keys():
                    params.pop('userfs')

                response_train_post = requests.post(train_endpoint,
                                                    json=training_configuration_metadata,
                                                    params=params,
                                                    headers=self._client._get_headers())

            else:
                response_train_post = requests.post(train_endpoint, json=training_configuration_metadata,
                                                    headers=self._client._get_headers())

        run_details = self._handle_response(201, u'training', response_train_post)

        try:
            trained_model_guid = self.get_id(run_details)

        except Exception:
            trained_model_guid = self.get_uid(run_details)

        if asynchronous is True:
            return run_details
        else:
            print_text_header_h1(u'Running \'{}\''.format(trained_model_guid))

            status = self.get_status(trained_model_guid)
            state = status[u'state']

            with StatusLogger(state) as status_logger:
                while state not in ['error', 'completed', 'canceled', 'failed']:
                    time.sleep(5)
                    status = self.get_status(trained_model_guid)
                    state = status['state']
                    status_logger.log_state(state)

            if u'completed' in state:
                print(u'\nTraining of \'{}\' finished successfully.'.format(str(trained_model_guid)))
            else:
                print(u'\nTraining of \'{}\' failed with status: \'{}\'.'.format(trained_model_guid, str(status)))

            self._logger.debug(u'Response({}): {}'.format(state, run_details))
            return self.get_details(trained_model_guid, _internal=True)

    def list(self, limit=None, asynchronous=False, get_all=False, return_as_df=True):
        """Print stored trainings in a table format. If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records at once
        :type limit: int, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object when asynchronous is False, default: True
        :type return_as_df: bool, optional

        **Examples**

        .. code-block:: python

            client.training.list()
            training_runs_df = client.training.list(limit=100)
            training_runs_df = client.training.list(limit=100, get_all=True)
            training_runs_df = []
            for entry in client.training.list(limit=100, asynchronous=True, get_all=True):
                training_runs_df.extend(entry)
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        def preprocess_details(details: dict):
            resources = details[u'resources']
            values = [(m[u'metadata'].get('id', m[u'metadata'].get('guid')), m[u'entity'][u'status'][u'state'],
                       m[u'metadata'][u'created_at']) for m in resources]

            return self._list(values, [u'ID (training)', u'STATE', u'CREATED'], limit=None,
                              default_limit=100000, sort_by=None)

        if asynchronous:
            return (preprocess_details(details) for details in self.get_details(
                limit=limit, asynchronous=asynchronous, get_all=get_all, _internal=True))

        else:
            details = self.get_details(limit=limit, get_all=get_all, _internal=True)
            table = preprocess_details(details)
            if return_as_df:
                return table

    def list_subtrainings(self, training_uid):
        """Print the sub-trainings in a table format.

        :param training_uid: training ID
        :type training_uid: str

        **Example**

        .. code-block:: python

            client.training.list_subtrainings()

        """
        ##For CP4D, check if either spce or project ID is set
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Sub-trainings are no longer available for trainings, please use list_intermediate_models().")

        self._client._check_if_either_is_set()
        details = self.get_details(training_uid, _internal=True)
        if "experiment" not in details["entity"]:
            raise WMLClientError("Sub-trainings are available for training created via experiment only.")
        details_parent = requests.get(
            self._wml_credentials['url'] + '/v4/trainings?parent_id=' + training_uid,
            params=self._client._params(),
            headers=self._client._get_headers()
        )
        details_json = self._handle_response(200, "Get training details", details_parent)
        resources = details_json["resources"]
        values = [(m[u'metadata'].get('id', m[u'metadata'].get('guid')), m[u'entity'][u'status'][u'state'], m[u'metadata'][u'created_at']) for m in resources]

        self._list(values, [u'ID (sub_training)', u'STATE', u'CREATED'], None, 50)

    def list_intermediate_models(self, training_uid):
        """Print the intermediate_models in a table format.

        :param training_uid: training ID
        :type training_uid: str

        .. note::

            This method is not supported for IBM Cloud Pak® for Data.

        **Example**

        .. code-block:: python

            client.training.list_intermediate_models()

        """
        ##For CP4D, check if either spce or project ID is set
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("This method is not supported for IBM Cloud Pak® for Data. " )

        self._client._check_if_either_is_set()
        details = self.get_details(training_uid, _internal=True)
        #if status is completed then only lists only global_output else display message saying "state value"
        training_state = details[u'entity'][u'status'][u'state']
        if(training_state=='completed'):

            if 'metrics' in details[u'entity'][u'status'] and details[u'entity'][u'status'].get('metrics') is not None:
                metrics_list = details[u'entity'][u'status'][u'metrics']
                new_list=[]
                for ml in metrics_list:
                    # if(ml[u'context'][u'intermediate_model'][u'process']=='global_output'):
                    if 'context' in ml and 'intermediate_model' in ml[u'context']:
                        name = ml[u'context'][u'intermediate_model'].get('name', "")
                        if 'location' in ml[u'context'][u'intermediate_model']:
                            path = ml[u'context'][u'intermediate_model'][u'location'].get('model', "")
                        else:
                            path = ""
                    else:
                        name = ""
                        path = ""

                    accuracy=ml[u'ml_metrics'].get('training_accuracy', "")
                    F1Micro=round(ml[u'ml_metrics'].get('training_f1_micro', 0), 2)
                    F1Macro = round(ml[u'ml_metrics'].get('training_f1_macro', 0), 2)
                    F1Weighted = round(ml[u'ml_metrics'].get('training_f1_weighted', 0), 2)
                    logLoss=round(ml[u'ml_metrics'].get('training_neg_log_loss', 0), 2)
                    PrecisionMicro = round(ml[u'ml_metrics'].get('training_precision_micro', 0), 2)
                    PrecisionWeighted = round(ml[u'ml_metrics'].get('training_precision_weighted', 0), 2)
                    PrecisionMacro = round(ml[u'ml_metrics'].get('training_precision_macro', 0), 2)
                    RecallMacro = round(ml[u'ml_metrics'].get('training_recall_macro', 0), 2)
                    RecallMicro = round(ml[u'ml_metrics'].get('training_recall_micro', 0), 2)
                    RecallWeighted = round(ml[u'ml_metrics'].get('training_recall_weighted', 0), 2)
                    createdAt = details[u'metadata'][u'created_at']
                    new_list.append([name,path,accuracy,F1Micro,F1Macro,F1Weighted,logLoss,PrecisionMicro,PrecisionMacro,PrecisionWeighted,RecallMicro,RecallMacro,RecallWeighted,createdAt])
                    new_list.append([])

                from tabulate import tabulate
                header = [u'NAME', u'PATH', u'Accuracy', u'F1Micro', u'F1Macro', u'F1Weighted', u'LogLoss', u'PrecisionMicro' , u'PrecisionMacro',u'PrecisionWeighted', u'RecallMicro', u'RecallMacro', u'RecallWeighted', u'CreatedAt' ]
                table = tabulate([header] + new_list)

                print(table)
                #self._list(new_list, [u'NAME', u'PATH', u'Accuracy', u'F1Micro', u'F1Macro', u'F1Weighted', u'LogLoss', u'PrecisionMicro' , u'PrecisionMacro',u'PrecisionWeighted', u'RecallMicro', u'RecallMacro', u'RecallWeighted', u'CreatedAt' ], None, 50)
            else:
                print(" There is no intermediate model metrics are available for this training uid. ")
        else:
            self._logger.debug("state is not completed")

    def cancel(self, training_uid, hard_delete=False):
        """Cancel a training which is currently running and remove it. This method is also be used to delete metadata
        details of the completed or canceled training run when `hard_delete` parameter is set to `True`.

        :param training_uid: training UID
        :type training_uid: str
        :param hard_delete: specify `True` or `False`:

            * `True` - to delete the completed or canceled training run
            * `False` - to cancel the currently running training run
        :type hard_delete: bool, optional

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.training.cancel(training_uid)
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(training_uid, u'training_uid', str, True)
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            params = self._client._params()
        else:
            params = None

        if hard_delete is True:
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                params.update({'hard_delete': u'true'})
            else:
                params = {}
                params.update({'hard_delete': u'true'})

        if not self._ICP and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            response_delete = requests.delete(self._client.service_instance._href_definitions.get_training_href(training_uid),
                                          headers=self._client._get_headers(),params=params)
        else:
            response_delete = requests.delete(self._client.service_instance._href_definitions.get_training_href(training_uid),
                                              headers=self._client._get_headers(),params=params)

        if response_delete.status_code == 400 and \
           response_delete.text is not None and 'Job already completed with state' in response_delete.text:
            print("Job is not running currently. Please use 'hard_delete=True' parameter to force delete"
                  " completed or canceled training runs.")
            return "SUCCESS"
        else:
            return self._handle_response(204, u'trained model deletion', response_delete, False)

    def _COS_logs(self, run_uid,on_start=lambda: {}):
        on_start()
        run_details = self.get_details(run_uid, _internal=True)
        if 'connection' in run_details["entity"]["results_reference"] and run_details["entity"]["results_reference"].get("connection") is not None:
            endpoint_url = run_details["entity"]["results_reference"]["connection"]["endpoint_url"]
            aws_access_key = run_details["entity"]["results_reference"]["connection"]["access_key_id"]
            aws_secret = run_details["entity"]["results_reference"]["connection"]["secret_access_key"]
            bucket = run_details["entity"]["results_reference"]["location"]["bucket"]
            # try:
            #     run_details["entity"]["training_results_reference"]["location"]["model_location"]
            # except:
            #     raise WMLClientError("The training-run has not started. Error - " + run_details["entity"]["status"]["error"]["errors"][0]["message"])

            if (bucket == ""):
                bucket = run_details["entity"]["results_reference"]["target"]["bucket"]
            import ibm_boto3

            client_cos = ibm_boto3.client(service_name='s3', aws_access_key_id=aws_access_key,
                                          aws_secret_access_key=aws_secret,
                                          endpoint_url=endpoint_url)

            try:
                if self._client.CLOUD_PLATFORM_SPACES:
                    logs = run_details["entity"].get("results_reference").get("location").get("logs")
                    if logs is None:
                        print(" There is no logs details for this Training run, hence no logs.")
                        return

                    key = logs + "/learner-1/training-log.txt"

                else:
                    try:
                        key = "data/" + run_details["metadata"].get("id", run_details["metadata"].get('guid')) + "/pipeline-model.json"

                        obj = client_cos.get_object(Bucket=bucket, Key=key)
                        pipeline_model = json.loads((obj['Body'].read().decode('utf-8')))

                    except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:
                        if ex.response['Error']['Code'] == 'NoSuchKey':
                            print(" Error - There is no training logs are found for the given training run id")
                            return
                         #   print("ERROR - Cannot find pipeline_model.json in the bucket "+ run_uid)
                        else:
                            print(ex)
                            return
                    if pipeline_model is not None:
                        key = pipeline_model["pipelines"][0]["nodes"][0]["parameters"]["model_id"] + "/learner-1/training-log.txt"
                    else:
                        print(" Error - Cannot find the any logs for the given training run id")
                obj = client_cos.get_object(Bucket=bucket, Key=key)
                print(obj['Body'].read().decode('utf-8'))
            except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:

                if ex.response['Error']['Code'] == 'NoSuchKey':
                    print("ERROR - Cannot find training-log.txt in the bucket")
                else:
                    print(ex)
                    print("ERROR - Cannot get the training run log in the bucket")
        else:
            print(" There is no connection details for this Training run, hence no logs.")


    def _COS_metrics(self, run_uid,on_start=lambda: {}):
        on_start()
        run_details = self.get_details(run_uid, _internal=True)
        endpoint_url = run_details["entity"]["results_reference"]["connection"]["endpoint_url"]
        aws_access_key = run_details["entity"]["results_reference"]["connection"]["access_key_id"]
        aws_secret = run_details["entity"]["results_reference"]["connection"]["secret_access_key"]
        bucket = run_details["entity"]["results_reference"]["location"]["bucket"]
        # try:
        #     run_details["entity"]["training_results_reference"]["location"]["model_location"]
        # except:
        #     raise WMLClientError("The training-run has not started. Error - " + run_details["entity"]["status"]["error"]["errors"][0]["message"])

        if (bucket == ""):
            bucket = run_details["entity"]["results_reference"]["target"]["bucket"]
        import ibm_boto3

        client_cos = ibm_boto3.client(service_name='s3', aws_access_key_id=aws_access_key,
                                      aws_secret_access_key=aws_secret,
                                      endpoint_url=endpoint_url)

        try:
            if self._client.CLOUD_PLATFORM_SPACES:
                logs = run_details["entity"].get("results_reference").get("location").get("logs")
                if logs is None:
                    print(" Metric log location details for this Training run is not available.")
                    return
                key = logs + "/learner-1/evaluation-metrics.txt"
            else:
                try:
                    key = run_details["metadata"].get("id", run_details["metadata"].get('guid')) + "/pipeline-model.json"

                    obj = client_cos.get_object(Bucket=bucket, Key=key)

                    pipeline_model = json.loads((obj['Body'].read().decode('utf-8')))
                except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:

                    if ex.response['Error']['Code'] == 'NoSuchKey':
                        print("ERROR - Cannot find pipeline_model.json in the bucket for training id "+ run_uid)
                        print("There is no training logs are found for the given training run id")
                        return
                    else:
                        print(ex)
                        return
                key = pipeline_model["pipelines"][0]["nodes"][0]["parameters"].get["model_id"] + "/learner-1/evaluation-metrics.txt"

            obj = client_cos.get_object(Bucket=bucket, Key=key)
            print(obj['Body'].read().decode('utf-8'))

        except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                print("ERROR - Cannot find evaluation-metrics.txt in the bucket")
            else:
                print(ex)
                print("ERROR - Cannot get the location of evaluation-metrics.txt details in the bucket")

    def monitor_logs(self, training_uid):
        """Print the logs of a training created.

        :param training_uid: training UID
        :type training_uid: str

        .. note::

            This method is not supported for IBM Cloud Pak® for Data.

        **Example**

        .. code-block:: python

            client.training.monitor_logs(training_uid)

        """
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Metrics logs are not supported. This method is not supported for IBM Cloud Pak® for Data. ")

        Training._validate_type(training_uid, u'training_uid', str, True)

        self._simple_monitor_logs(training_uid, lambda: print_text_header_h1(u'Log monitor started for training run: ' + str(training_uid)))

        print_text_header_h2('Log monitor done.')

    def _simple_monitor_logs(self, training_uid, on_start=lambda: {}):
        try:
            run_details = self.get_details(training_uid, _internal=True)
        except ApiRequestFailure as ex:
            if "404" in str(ex.args[1]):
                print("Could not find the training run details for the given training run id. ")
                return
            else:
                raise ex

        status = run_details["entity"]["status"]["state"]

        if (status == "completed" or status == "error" or status == "failed" or status == "canceled"):
            self._COS_logs(training_uid,
                           lambda: print_text_header_h1(u'Log monitor started for training run: ' + str(training_uid)))
        else:
            if not self._ICP:
                if self._client.CLOUD_PLATFORM_SPACES:
                    ws_param = self._client._params()
                    if 'project_id' in ws_param.keys():
                        proj_id = ws_param.get('project_id')
                        monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                                 u'wss') + u'/ml/v4/trainings/' + training_uid + "?project_id=" + proj_id
                    else:
                        space_id = ws_param.get('space_id')
                        monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                                 u'wss') + u'/ml/v4/trainings/' + training_uid + "?space_id=" + space_id
                else:
                    monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                     u'wss') + u'/v4/trainings/' + training_uid
            else:
                monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                         u'wss') + u'/v4/trainings/' + training_uid
            websocket = WebSocket(monitor_endpoint)
            try:
                websocket.add_header(bytes("Authorization", "utf-8"), bytes("Bearer " + self._client.service_instance._get_token(), "utf-8"))
            except:
                websocket.add_header(bytes("Authorization"), bytes("bearer " + self._client.service_instance._get_token()))

            on_start()

            for event in websocket:

                if event.name == u'text':
                    text = json.loads(event.text)
                    entity = text[u'entity']
                    if 'status' in entity:
                      if 'message' in entity['status']:
                        message = entity['status']['message']
                        if len(message) > 0:
                          print(message)

            websocket.close()

    def monitor_metrics(self, training_uid):
        """Print the metrics of a training created.

        :param training_uid: training UID
        :type training_uid: str

        .. note::

            This method is not supported for IBM Cloud Pak® for Data.

        **Example**

        .. code-block:: python

            client.training.monitor_metrics(training_uid)
        """
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Metrics monitoring is not supported for IBM Cloud Pak® for Data")

        Training._validate_type(training_uid, u'training_uid', str, True)
        try:
            run_details = self.get_details(training_uid, _internal=True)
        except ApiRequestFailure as ex:
            if "404" in str(ex.args[1]):
                print("Could not find the training run details for the given training run id. ")
                return
            else:
                raise ex
        status = run_details["entity"]["status"]["state"]

        if (status == "completed" or status == "error" or status == "failed" or status == "canceled"):
            self._COS_metrics(training_uid,
                           lambda: print_text_header_h1(u'Log monitor started for training run: ' + str(training_uid)))
        else:
            if not self._ICP:
                if self._client.CLOUD_PLATFORM_SPACES:
                    ws_param = self._client._params()
                    if 'project_id' in ws_param.keys():
                        proj_id = ws_param.get('project_id')
                        monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                                 u'wss') + u'/ml/v4/trainings/' + training_uid + "?project_id=" + proj_id
                    else:
                        space_id = ws_param.get('space_id')
                        monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                                 u'wss') + u'/ml/v4/trainings/' + training_uid + "?space_id=" + space_id

                else:
                    monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                     u'wss') + u'/v4/trainings/' + training_uid
            else:
                monitor_endpoint = self._wml_credentials[u'url'].replace(u'https',
                                                                         u'wss') + u'/v4/trainings/' + training_uid
            websocket = WebSocket(monitor_endpoint)
            try:
                websocket.add_header(bytes("Authorization", "utf-8"), bytes("Bearer " + self._client.service_instance._get_token(), "utf-8"))
            except:
                websocket.add_header(bytes("Authorization"), bytes("bearer " + self._client.service_instance._get_token()))

            print_text_header_h1('Metric monitor started for training run: ' + str(training_uid))

            for event in websocket:
                if event.name == u'text':
                    text = json.loads(event.text)
                    entity = text[u'entity']
                    if 'status' in entity:
                        status = entity[u'status']
                        if u'metrics' in status:
                            metrics = status[u'metrics']
                            if len(metrics) > 0:
                             metric = metrics[0]
                             print(metric)

            websocket.close()

            print_text_header_h2('Metric monitor done.')

    def get_metrics(self, training_uid):
        """Get metrics.

        :param training_uid: training UID
        :type training_uid: str

        :return: metrics of a training run
        :rtype: list of dict

        **Example**

        .. code-block:: python

            training_status = client.training.get_metrics(training_uid)

        """
        Training._validate_type(training_uid, u'training_uid', str, True)
        status = self.get_status(training_uid)
        if 'metrics' in status:
            return status['metrics']
        else:
            details = self.get_details(training_uid, _internal=True)
            if 'metrics' in details:
                return details['metrics']
            else:
                raise WMLClientError("No metrics details are available for the given training_uid")

    def _get_latest_metrics(self, training_uid):
        """Get latest metrics values.

        :param training_uid: ID of trained model
        :type training_uid: str

        :return: metric values
        :rtype: list of dicts

        **Example**

        .. code-block:: python

            client.training.get_latest_metrics(training_uid)
        """
        Training._validate_type(training_uid, u'training_uid', str, True)

        status = self.get_status(training_uid)
        metrics = status.get('metrics', [])
        latest_metrics = []

        if len(metrics) > 0:
            grouped_metrics = group_metrics(metrics)

            for key, value in grouped_metrics.items():
                sorted_value = sorted(value, key=lambda k: k['iteration'])

            latest_metrics.append(sorted_value[-1])

        return latest_metrics
