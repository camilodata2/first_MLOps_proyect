#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
import json
import re
import time
import copy
from ibm_watson_machine_learning.wml_client_error import MissingValue, WMLClientError, MissingMetaProp
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from multiprocessing import Pool
from ibm_watson_machine_learning.utils import EXPERIMENT_DETAILS_TYPE
from ibm_watson_machine_learning.hpo import HPOParameter, HPOMethodParam
from ibm_watson_machine_learning.metanames import ExperimentMetaNames
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES, PIPELINES, LIBRARIES, EXPERIMENTS, RUNTIMES, DEPLOYMENTS

_DEFAULT_LIST_LENGTH = 50

class Experiments(WMLResource):
    """Run new experiment."""

    ConfigurationMetaNames = ExperimentMetaNames()
    """MetaNames for experiments creation."""

    @staticmethod
    def _HPOParameter(name, values=None, max=None, min=None, step=None):
        return HPOParameter(name, values, max, min, step)

    @staticmethod
    def _HPOMethodParam(name=None, value=None):
        return HPOMethodParam(name, value)

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._experiments_uids_cache = {}

    def store(self, meta_props):
        """Create an experiment.

        :param meta_props: meta data of the experiment configuration. To see available meta names use:

            .. code-block:: python

                client.experiments.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: stored experiment metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.experiments.ConfigurationMetaNames.NAME: 'my_experiment',
                client.experiments.ConfigurationMetaNames.EVALUATION_METRICS: ['accuracy'],
                client.experiments.ConfigurationMetaNames.TRAINING_REFERENCES: [
                    {'pipeline': {'href': pipeline_href_1}},
                    {'pipeline': {'href':pipeline_href_2}}
                ]
            }
            experiment_details = client.experiments.store(meta_props=metadata)
            experiment_href = client.experiments.get_href(experiment_details)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        metaProps = self.ConfigurationMetaNames._generate_resource_metadata(meta_props)
        ##Check if default space is set
        if self._client.CAMS and not self._client.ICP_PLATFORM_SPACES:
            if self._client.default_space_id is not None:
                metaProps['space'] = {'href': "/v4/spaces/"+self._client.default_space_id}
            elif self._client.default_project_id is not None:
                metaProps['project'] = {'href': "/v2/projects/"+self._client.default_project_id}
            else:
                raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            if self._client.default_space_id is not None:
                metaProps['space_id'] = self._client.default_space_id
            elif self._client.default_project_id is not None:
                metaProps['project_id'] = self._client.default_project_id
            else:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, str, True)
        else:
            self.ConfigurationMetaNames._validate(meta_props)

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            response_experiment_post = requests.post(
                self._client.service_instance._href_definitions.get_experiments_href(),
                params=self._client._params(skip_for_create=True),
                json=metaProps,
                headers=self._client._get_headers()
            )
        else:
            response_experiment_post = requests.post(
                self._client.service_instance._href_definitions.get_experiments_href(),
                json=metaProps,
                headers=self._client._get_headers()
            )

        return self._handle_response(201, u'saving experiment', response_experiment_post)

    def update(self, experiment_uid, changes):
        """Updates existing experiment metadata.

        :param experiment_uid: UID of experiment which definition should be updated
        :type experiment_uid: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated experiment
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.experiments.ConfigurationMetaNames.NAME: "updated_exp"
            }
            exp_details = client.experiments.update(experiment_uid, changes=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        self._validate_type(experiment_uid, u'experiment_uid', str, True)
        self._validate_type(changes, u'changes', dict, True)

        details = self._client.repository.get_details(experiment_uid)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes, with_validation=True)

        url = self._client.service_instance._href_definitions.get_experiment_href(experiment_uid)
        response = requests.patch(url, json=patch_payload, params = self._client._params(),headers=self._client._get_headers())
        updated_details = self._handle_response(200, u'experiment patch', response)

        return updated_details

    def get_details(self, experiment_uid=None, limit=None, asynchronous=False, get_all=False):
        """Get metadata of experiment(s). If no experiment UID is specified all experiments metadata is returned.

        :param experiment_uid:  UID of experiment
        :type experiment_uid: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: experiment(s) metadata
        :rtype: dict (if UID is not None) or {"resources": [dict]} (if UID is None)

        **Example**

        .. code-block:: python

            experiment_details = client.experiments.get_details(experiment_uid)
            experiment_details = client.experiments.get_details()
            experiment_details = client.experiments.get_details(limit=100)
            experiment_details = client.experiments.get_details(limit=100, get_all=True)
            experiment_details = []
            for entry in client.experiments.get_details(limit=100, asynchronous=True, get_all=True):
                experiment_details.extend(entry)

         """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        url = self._client.service_instance._href_definitions.get_experiments_href()

        if experiment_uid is None:
            return self._get_artifact_details(url, experiment_uid, limit, 'experiment',
                                              _async=asynchronous, _all=get_all)

        else:
            return self._get_artifact_details(url, experiment_uid, limit, 'experiment')

    @staticmethod
    def get_uid(experiment_details):
        """Get Unique Id of stored experiment.

        :param experiment_details: metadata of the stored experiment
        :type experiment_details: dict

        :return: Unique Id of stored experiment
        :rtype: str

        **Example**

        .. code-block:: python

            experiment_details = client.experiments.get_details(experiment_uid)
            experiment_uid = client.experiments.get_uid(experiment_details)

        """
        Experiments._validate_type(experiment_details, u'experiment_details', object, True)
        if 'id' not in experiment_details[u'metadata']:
            Experiments._validate_type_of_details(experiment_details, EXPERIMENT_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(experiment_details, u'experiment_details',
                                                           [u'metadata', u'id'])

    @staticmethod
    def get_id(experiment_details):
        """Get Unique Id of stored experiment.

        :param experiment_details: metadata of the stored experiment
        :type experiment_details: dict

        :return: Unique Id of stored experiment
        :rtype: str

        **Example**

        .. code-block:: python

            experiment_details = client.experiments.get_details(experiment_id)
            experiment_uid = client.experiments.get_id(experiment_details)

        """
        Experiments._validate_type(experiment_details, u'experiment_details', object, True)
        if 'id' not in experiment_details[u'metadata']:
            Experiments._validate_type_of_details(experiment_details, EXPERIMENT_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(experiment_details, u'experiment_details',
                                                           [u'metadata', u'id'])

    @staticmethod
    def get_href(experiment_details):
        """Get href of stored experiment.

        :param experiment_details: metadata of the stored experiment
        :type experiment_details: dict

        :return: href of stored experiment
        :rtype: str

        **Example**

        .. code-block:: python

            experiment_details = client.experiments.get_details(experiment_uid)
            experiment_href = client.experiments.get_href(experiment_details)

        """
        Experiments._validate_type(experiment_details, u'experiment_details', object, True)
        if 'href' in experiment_details['metadata']:
            Experiments._validate_type_of_details(experiment_details, EXPERIMENT_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(experiment_details, u'experiment_details',
                                                           [u'metadata', u'href'])
        else:
            experiment_id = WMLResource._get_required_element_from_dict(experiment_details, u'experiment_details',
                                                                     [u'metadata', u'id'])
            return "/ml/v4/experiments/" + experiment_id

    def delete(self, experiment_uid):
        """Delete a stored experiment.

        :param experiment_uid: Unique Id of the stored experiment
        :type experiment_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.experiments.delete(experiment_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Experiments._validate_type(experiment_uid, u'experiment_uid', str, True)

        url = self._client.service_instance._href_definitions.get_experiment_href(experiment_uid)
        response = requests.delete(url, params = self._client._params(),headers=self._client._get_headers())

        return self._handle_response(204, u'experiment deletion', response, False)

    def list(self, limit=None, return_as_df=True):
        """Print stored experiments in a table format.
        If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed experiments or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.experiments.list()

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        experiment_resources = self.get_details(limit=limit)[u'resources']
        header_list = [u'GUID', u'NAME', u'CREATED']
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            experiment_values = [(m[u'metadata'][u'id'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at']) for m in
                                 experiment_resources]
            header_list = [u'ID', u'NAME', u'CREATED']
        else:
            experiment_values = [(m[u'metadata'][u'id'], m[u'entity'][u'name'], m[u'metadata'][u'created_at']) for m in
                                 experiment_resources]

        table = self._list(experiment_values, header_list, limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def create_revision(self, experiment_id):
        """Create a new experiment revision.

        :param experiment_id: Unique Id of the stored experiment
        :type experiment_id: str

        :return: stored experiment new revision details
        :rtype: dict

        **Example**

        .. code-block:: python

            experiment_revision_artifact = client.experiments.create_revision(experiment_id)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Experiments._validate_type(experiment_id, u'experiment_id', str, True)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revision support is not there in this WML server. It is supported only from 3.1.0 onwards.')
        else:
            url =self._client.service_instance._href_definitions.get_experiments_href()
            return self._create_revision_artifact(url, experiment_id, 'experiments')

    def get_revision_details(self, experiment_uid, rev_uid):
        """Get metadata of stored experiments revisions.

        :param experiment_uid: stored experiment UID
        :type experiment_uid: str

        :param rev_uid: rev_id number of experiment
        :type rev_uid: int

        :return: stored experiment revision metadata
        :rtype: dict

        Example:

        .. code-block:: python

            experiment_details = client.experiments.get_revision_details(experiment_uid, rev_id)

         """
        self._client._check_if_either_is_set()
        Experiments._validate_type(experiment_uid, u'experiment_uid', str, True)
        Experiments._validate_type(rev_uid, u'rev_uid', int, True)
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                'Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data 3.0 and above.')
        else:
            url = self._client.service_instance._href_definitions.get_experiment_href(experiment_uid)
            return self._get_with_or_without_limit(url, limit=None, op_name="experiments",
                                                   summary=None, pre_defined=None, revision=rev_uid)

    def list_revisions(self, experiment_uid, limit=None, return_as_df=True):
        """Print all revision for the given experiment uid in a table format.

        :param experiment_uid: Unique id of stored experiment
        :type experiment_uid: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed revisions or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.experiments.list_revisions(experiment_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Experiments._validate_type(experiment_uid, u'experiment_uid', str, True)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revision support is not there in this WML server. It is supported only from 3.1.0 onwards.')
        else:
            url = self._client.service_instance._href_definitions.get_experiment_href(experiment_uid)
            experiment_resources = self._get_artifact_details(url, "revisions", limit, 'model revisions')[u'resources']
            experiment_values = [
                (m[u'metadata'][u'rev'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at']) for m in
                experiment_resources]

            table = self._list(experiment_values, [u'GUID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
            if return_as_df:
                return table

    def clone(self, experiment_uid, space_id=None, action="copy", rev_id=None):
        """Create a new experiment identical with the given experiment either in the same space or in a new space.
        All dependent assets will be cloned too.

        :param experiment_uid:  Unique Id of the experiment to be cloned
        :type experiment_uid: str

        :param space_id: Unique Id of the space to which the experiment needs to be cloned
        :type space_id: str, optional

        :param action: action specifying "copy" or "move"
        :type action: str, optional

        :param rev_id: revision ID of the experiment
        :type rev_id: str, optional

        :return: metadata of the experiment cloned
        :rtype: dict

        **Example**

        .. code-block:: python

            client.experiments.clone(experiment_uid=artifact_id,space_id=space_uid,action="copy")

        .. note::

            * If revision id is not specified, all revisions of the artifact are cloned.

            * Space id is mandatory for move action.

        """
        Experiments._validate_type(experiment_uid, 'experiment_uid', str, True)
        clone_meta = {}
        if space_id is not None:
            clone_meta["space"] = {"href": API_VERSION + SPACES + "/" + space_id}
        if action is not None:
            clone_meta["action"] = action
        if rev_id is not None:
            clone_meta["rev"] = rev_id

        url = self._client.service_instance._href_definitions.get_experiment_href(experiment_uid)
        response_post = requests.post(url, json=clone_meta,
                                      headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=200, operationName=u'cloning experiment',
                                        response=response_post)

        return details


