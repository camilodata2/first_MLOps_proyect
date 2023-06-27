#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging, re, os

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.runtimes_artifact import RuntimesArtifact
from ibm_watson_machine_learning.libs.repo.swagger_client.api_client import ApiException
import json
from .runtimes_adapter import WmlRuntimesAdapter
from ibm_watson_machine_learning.libs.repo.swagger_client.models import RuntimeSpecDefinitionInput, RuntimeSpecDefinitionInputCustomLibraries,\
    PatchOperationRuntimeSpec, RuntimeSpecDefinitionInputPlatform



logger = logging.getLogger('RuntimesCollection')


class RuntimesCollection:
    """
    Client operating on runtimes in repository service.

    :param str base_path: base url to Watson Machine Learning instance
    :param MLRepositoryApi repository_api: client connecting to repository rest api
    :param MLRepositoryClient client: high level client used for simplification and argument for constructors
    """
    def __init__(self, base_path, repository_api, client):

        from ibm_watson_machine_learning.libs.repo.mlrepositoryclient import MLRepositoryClient
        from ibm_watson_machine_learning.libs.repo.mlrepositoryclient import MLRepositoryApi

        if not isinstance(base_path, str) and not isinstance(base_path, unicode):
            raise ValueError('Invalid type for base_path: {}'.format(base_path.__class__.__name__))

        if not isinstance(repository_api, MLRepositoryApi):
            raise ValueError('Invalid type for repository_api: {}'.format(repository_api.__class__.__name__))

        if not isinstance(client, MLRepositoryClient):
            raise ValueError('Invalid type for client: {}'.format(client.__class__.__name__))

        self.base_path = base_path
        self.repository_api = repository_api
        self.client = client

    def all(self, library_name=None, library_id=None):
        """
        Gets info about all runtimes which belong to this user.

        Not complete information is provided by all(). To get detailed information about experiment use get().

        :return: info about runtimes
        :rtype: list[ExperimentsArtifact]
        """

        all_runtimes = self.repository_api.v3_runtime_spec_list(library_name, library_id)
        list_runtimes_artifact = []
        if all_runtimes is not None:
            resr = all_runtimes.resources
            for iter1 in resr:
                list_runtimes_artifact.append(WmlRuntimesAdapter(iter1, self.client).artifact())
            return list_runtimes_artifact
        else:
            return []

    def get(self, runtimes_id):

        """
        Gets detailed information about runtimes.

        :param str runtimes_id: uid used to identify experiment
        :return: returned object has all attributes of SparkPipelineArtifact but its class name is PipelineArtifact
        :rtype: PipelineArtifact(SparkPipelineLoader)
        """

        if not isinstance(runtimes_id, str) and not isinstance(runtimes_id, unicode):
            raise ValueError('Invalid type for experiment_id: {}'.format(runtimes_id.__class__.__name__))
        if(runtimes_id.__contains__("/v4/runtimes")):
            matched = re.search('.*/v4/runtimes/([A-Za-z0-9\-]+)', runtimes_id)
            if matched is not None:
                runtimes_id = matched.group(1)
                return self.get(runtimes_id)
            else:
                raise ValueError('Unexpected artifact href: {} format'.format(runtimes_id))
        else:
            runtimes_output = self.repository_api.v3_runtime_spec_get(runtimes_id)
            if runtimes_output is not None:
                return WmlRuntimesAdapter(runtimes_output, self.client).artifact()
            else:
                raise Exception('Library not found'.format(runtimes))

    def remove(self, runtimes_id):
        """
        Removes runtimes with given runtimes_id.

        :param str runtimes_id: uid used to identify experiment
        """

        if not isinstance(runtimes_id, str) and not isinstance(runtimes_id, unicode):
            raise ValueError('Invalid type for runtimes_id: {}'.format(runtimes_id.__class__.__name__))

        if(runtimes_id.__contains__("/v4/runtimes")):
            matched = re.search('.*/v4/runtimes/([A-Za-z0-9\-]+)', runtimes_id)
            if matched is not None:
                runtimes_id_value = matched.group(1)
                self.remove(runtimes_id_value)
            else:
                raise ValueError('Unexpected experiment artifact href: {} format'.format(runtimes_id))
        else:
            return self.repository_api.v3_runtime_spec_delete(runtimes_id)

    def patch(self, runtimes_id, artifact):
        runtimes_patch_input = self.prepare_runtimes_patch_input(artifact)
        runtimes_patch_output = self.repository_api.v3_runtime_spec_patch_with_http_info(runtimes_patch_input, runtimes_id)
        statuscode = runtimes_patch_output[1]

        if statuscode != 200:
            logger.info('Error while patching runtimes: no location header')
            raise ApiException(statuscode, "Error while patching runtimes")

        if runtimes_patch_output is not None:
            new_artifact = WmlRuntimesAdapter(runtimes_patch_output[0], self.client).artifact()
        return new_artifact

    def save(self, artifact, runtimespec_path  = None):
        """
        Saves runtimes in repository service.

        :param artifact : RuntimesArtifact to be saved in the repository service
        :param runtimes : runtimes file path to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: RuntimesArtifact
        """
        logger.debug('Creating a new WML Runtimes : {}'.format(artifact.name))

        if not issubclass(type(artifact), RuntimesArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        runtimes_input = self._prepare_wml_runtimes_input(artifact)
        runtimes_output = self.repository_api.v3_runtime_spec_create_with_http_info(runtimes_input)

        runtimes_id = runtimes_output[0].metadata.guid

        if runtimespec_path is not None:
            if not os.path.exists(runtimespec_path):
                raise IOError('The runtimes specified ( {} ) does not exist.'.format(runtimespec_path))
            artifact.runtimespec_path = runtimespec_path

        if runtimespec_path is None and artifact.runtimespec_path is not None:
            if not os.path.exists(artifact.runtimespec_path):
                raise IOError('The artifact specified ( {} ) does not exist.'.format(artifact.runtimespec_path))

        statuscode = runtimes_output[1]
        if statuscode != 201:
            logger.info('Error while creating runtimes: no location header')
            raise ApiException(statuscode, 'No artifact location')

        if runtimes_output is not None:
            new_artifact = WmlRuntimesAdapter(runtimes_output[0], self.client).artifact()
            new_artifact.runtimespec_path = artifact.runtimespec_path
            if artifact.runtimespec_path is not None:
                self._upload_runtimes_content(new_artifact, artifact.runtimespec_path)
        return new_artifact


    @staticmethod
    def _prepare_wml_runtimes_input(artifact):
        name = None
        version = None
        platform =None
        platform_version = None
        description = None

        name = artifact.meta.prop(MetaNames.RUNTIMES.NAME)
        description = artifact.meta.prop(MetaNames.RUNTIMES.DESCRIPTION)
        platform = json.loads(artifact.meta.prop(MetaNames.RUNTIMES.PLATFORM))
        platform_input = RuntimeSpecDefinitionInputPlatform(platform.get('name'), platform.get('version'))
        custom_libs = json.loads(artifact.meta.prop(MetaNames.RUNTIMES.CUSTOM_LIBRARIES_URLS))
        custom_library_param_list = []
        if custom_libs is not None:
            custom_library_param_list = RuntimeSpecDefinitionInputCustomLibraries(custom_libs.get('urls', None))
        if description is not None and not isinstance(description, str):
            raise ValueError('Invalid data format for description.')
        runtimes_input = RuntimeSpecDefinitionInput(
            name, description, platform_input, custom_library_param_list)
        return runtimes_input

    @staticmethod
    def prepare_runtimes_patch_input(artifact):
        patch_list = []
        patch_input = artifact.meta.prop(MetaNames.RUNTIMES.PATCH_INPUT)
        if isinstance(patch_input, str):
            patch_input_list = json.loads(artifact.meta.prop(MetaNames.RUNTIMES.PATCH_INPUT))
            if isinstance(patch_input_list, list):
                for iter1 in patch_input_list:
                    runtimes_patch = PatchOperationRuntimeSpec(
                        op = iter1.get('op'),
                        path= iter1.get('path'),
                        value = iter1.get('value', None),
                        _from =iter1.get('from', None),
                    )
                    patch_list.append(runtimes_patch)

                return patch_list

    def _upload_runtimes_content(self, runtimes_artifact, artifact_path, query_param=None):
        runtimes_id = runtimes_artifact.uid
        content_stream = runtimes_artifact.reader().read()
        self.repository_api.upload_runtimes(runtimes_id, content_stream)
        content_stream.close()
        runtimes_artifact.reader().close()
        logger.debug('Content uploaded for runtimes created at: {}'.format(runtimes_id))

