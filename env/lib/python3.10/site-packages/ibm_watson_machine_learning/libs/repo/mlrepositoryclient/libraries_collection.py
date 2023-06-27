#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging, re, os

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.libraries_artifact import LibrariesArtifact
from ibm_watson_machine_learning.libs.repo.swagger_client.api_client import ApiException
import json
from .libraries_adapter import WmlLibrariesAdapter
from ibm_watson_machine_learning.libs.repo.swagger_client.models import LibrariesDefinitionInput, LibrariesDefinitionInputPlatform, PatchOperationLibraries


logger = logging.getLogger('LibrariesCollection')


class LibrariesCollection:
    """
    Client operating on libraries in repository service.

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

    def all(self, queryMap=None):
        """
        Gets info about all libraries which belong to this user.

        Not complete information is provided by all(). To get detailed information about experiment use get().

        :return: info about libraries
        :rtype: list[ExperimentsArtifact]
        """
        all_libraries = self.repository_api.v3_libraries_list(queryMap)
        list_libraries_artifact = []
        if all_libraries is not None:
            resr = all_libraries.resources

            for iter1 in resr:
                list_libraries_artifact.append(WmlLibrariesAdapter(iter1, self.client).artifact())
            return list_libraries_artifact
        else:
            return []

    def get(self, libraries_id):

        """
        Gets detailed information about libraries.

        :param str libraries_id: uid used to identify experiment
        :return: returned object has all attributes of SparkPipelineArtifact but its class name is PipelineArtifact
        :rtype: PipelineArtifact(SparkPipelineLoader)
        """

        if not isinstance(libraries_id, str) and not isinstance(libraries_id, unicode):
            raise ValueError('Invalid type for experiment_id: {}'.format(libraries_id.__class__.__name__))
        if(libraries_id.__contains__("/v4/libraries")):
            matched = re.search('.*/v4/libraries/([A-Za-z0-9\-]+)', libraries_id)
            if matched is not None:
                library_id = matched.group(1)
                return self.get(library_id)
            else:
                raise ValueError('Unexpected artifact href: {} format'.format(libraries_id))
        else:
            libraries_output = self.repository_api.v3_libraries_get(libraries_id)
            if libraries_output is not None:
                return WmlLibrariesAdapter(libraries_output, self.client).artifact()
            else:
                raise Exception('Library not found'.format(libraries))

    def remove(self, libraries_id):
        """
        Removes libraries with given libraries_id.

        :param str libraries_id: uid used to identify experiment
        """

        if not isinstance(libraries_id, str) and not isinstance(libraries_id, unicode):
            raise ValueError('Invalid type for libraries_id: {}'.format(libraries_id.__class__.__name__))

        if(libraries_id.__contains__("/v4/libraries")):
            matched = re.search('.*/v4/libraries/([A-Za-z0-9\-]+)', libraries_id)
            if matched is not None:
                libraries_id_value = matched.group(1)
                self.remove(libraries_id_value)
            else:
                raise ValueError('Unexpected experiment artifact href: {} format'.format(libraries_id))
        else:
            return self.repository_api.v3_libraries_delete_id(libraries_id)

    def patch(self, libraries_id, artifact):
        libraries_patch_input = self.prepare_libraries_patch_input(artifact)
        libraries_patch_output = self.repository_api.v3_libraries_patch_with_http_info(libraries_patch_input, libraries_id)
        statuscode = libraries_patch_output[1]

        if statuscode != 200:
            logger.info('Error while patching libraries: no location header')
            raise ApiException(statuscode,"Error while patching libraries")

        if libraries_patch_output is not None:
            new_artifact = WmlLibrariesAdapter(libraries_patch_output[0], self.client).artifact()
        return new_artifact

    def save(self, artifact, library = None):
        """
        Saves libraries in repository service.

        :param artifact : LibrariesArtifact to be saved in the repository service
        :param library : library file path to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: LibrariesArtifact
        """
        logger.debug('Creating a new WML Libraries : {}'.format(artifact.name))

        if not issubclass(type(artifact), LibrariesArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        library_input = self._prepare_wml_library_input(artifact)
        library_output = self.repository_api.v3_libraries_create_with_http_info(library_input)

        library_id = library_output[0].metadata.guid

        if library is not None:
            if not os.path.exists(library):
                raise IOError('The library specified ( {} ) does not exist.'.format(library))
            artifact.library = library

        if library is None and artifact.library is not None:
            if not os.path.exists(artifact.library):
                raise IOError('The artifact specified ( {} ) does not exist.'.format(artifact.library))

        statuscode = library_output[1]
        if statuscode != 201:
            logger.info('Error while creating libraries: no location header')
            raise ApiException(statuscode, 'No artifact location')

        if library_output is not None:
            new_artifact = WmlLibrariesAdapter(library_output[0], self.client).artifact()
            #new_artifact = WmlLibrariesAdapter(library_output, self.client).artifact()
            new_artifact.library = artifact.library
            self._upload_libraries_content(new_artifact, artifact.library)
        return new_artifact


    @staticmethod
    def _prepare_wml_library_input(artifact):
        name = None
        version = None
        platform =None
        description = None

        name = artifact.meta.prop(MetaNames.LIBRARIES.NAME)
        version = artifact.meta.prop(MetaNames.LIBRARIES.VERSION)
        description = artifact.meta.prop(MetaNames.LIBRARIES.DESCRIPTION)
        if description is not None and not isinstance(description, str):
            raise ValueError('Invalid data format for description.')

        platform = json.loads(artifact.meta.prop(MetaNames.LIBRARIES.PLATFORM))
        platform_version = platform.get('versions')
        if platform_version is not None:
            if not issubclass (type(platform_version), list):
                raise ValueError('Invalid data format for platform.version.')
        platform_input = LibrariesDefinitionInputPlatform(platform.get('name'), platform.get('versions'))

        library_input = LibrariesDefinitionInput(
            name, description, version, platform_input)
        return library_input

    @staticmethod
    def prepare_libraries_patch_input(artifact):
        patch_list = []
        patch_input = artifact.meta.prop(MetaNames.LIBRARIES.PATCH_INPUT)
        if isinstance(patch_input, str):
            patch_input_list = json.loads(artifact.meta.prop(MetaNames.LIBRARIES.PATCH_INPUT))
            if isinstance(patch_input_list, list):
                for iter1 in patch_input_list:
                    libraries_patch = PatchOperationLibraries(
                        op = iter1.get('op'),
                        path= iter1.get('path'),
                        value = iter1.get('value', None),
                        _from =iter1.get('from', None),
                    )
                    patch_list.append(libraries_patch)

                return patch_list

    def _upload_libraries_content(self, libraries_artifact, artifact_path, query_param=None):
        lib_id = libraries_artifact.uid
        content_stream = libraries_artifact.reader().read()
        self.repository_api.upload_libraries(lib_id, content_stream)
        content_stream.close()
        libraries_artifact.reader().close()
        logger.debug('Content uploaded for libraries created at: {}'.format(lib_id))

