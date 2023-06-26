#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging
import re
import json

from .function_adapter import FunctionAdapter
from ibm_watson_machine_learning.libs.repo.swagger_client.rest import ApiException
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.function_artifact import FunctionArtifact
from ibm_watson_machine_learning.libs.repo.swagger_client.models.ml_assets_create_function_input import MlAssetsCreateFunctionInput
from ibm_watson_machine_learning.libs.repo.swagger_client.models import TagRepository
from ibm_watson_machine_learning.libs.repo.swagger_client.models import PatchOperationFunctions

logger = logging.getLogger('FunctionCollection')


class FunctionCollection:
    """
    Client operating on functions in repository service.

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

    def all(self, expand=None, runtime_name=None, runtime_id=None):
        """
        Gets info about all functions which belong to this user.

        Not complete information is provided by all(). To get detailed information about function use get().

        :return: info about functions
        :rtype: list[FunctionArtifact]
        """
        all_functions = self.repository_api.repository_functions_list(self._get_expand_query_param_val(expand), runtime_name, runtime_id)
        list_functions_artifact = []
        if all_functions is not None:
            resr = all_functions.resources
            for iter1 in resr:
                list_functions_artifact.append(FunctionAdapter(iter1, self.client).artifact())
            return list_functions_artifact
        else:
            return []

    def get(self, artifact_id, expand = None):
        """
        Gets detailed information about function.

        :param str artifact_id: uid used to identify function
        :return: returned object has all attributes of FunctionArtifact
        :rtype: FunctionArtifact
        """
        logger.debug('Fetching information about function: {}'.format(artifact_id))

        if not isinstance(artifact_id, str) and not isinstance(artifact_id, unicode):
            raise ValueError('Invalid type for artifact_id: {}'.format(artifact_id.__class__.__name__))

        function_output = self.repository_api.repository_function_get(artifact_id, self._get_expand_query_param_val(expand))

        if function_output is not None:

            return FunctionAdapter(function_output, self.client).artifact()
        else:
            logger.debug('Function with guid={} not found'.format(artifact_id))
            raise ApiException('Function with guid={} not found'.format(artifact_id))

    def revisions(self, artifact_id, expand = None):
        """
        Gets all available revisions.

        :param str artifact_id: uid used to identify function
        :return: ???
        :rtype: list[str]
        """

        if not isinstance(artifact_id, str) and not isinstance(artifact_id, unicode):
            raise ValueError('Invalid type for artifact_id: {}'.format(artifact_id.__class__.__name__))

        logger.debug('Fetching information about function: {}'.format(artifact_id))

        function_output = self.repository_api.repository_list_function_revisions(artifact_id, self._get_expand_query_param_val(expand))

        list_function_revision_artifact = [FunctionArtifact]
        if function_output is not None:
            resr = function_output.resources
            for iter1 in resr:
                list_function_revision_artifact.append(FunctionAdapter(iter1, self.client).artifact())
            return list_function_revision_artifact
        else:
            logger.debug('Function with guid={} not found'.format(artifact_id))
            raise ApiException('Function with guid={} not found'.format(artifact_id))

    def revision(self, artifact_id, rev, expand):
        """
        Gets Function revision with given artifact_id and ver
        :param str artifact_id: uid used to identify function
        :param str rev: uid used to identify revision of function
        :return: FunctionArtifact(FunctionLoader) -- returned object has all attributes of FunctionArtifact
        """
        logger.debug('Fetching information about function revision: {}, {}'.format(artifact_id, rev))

        if not isinstance(artifact_id, str) and not isinstance(artifact_id, unicode):
            raise ValueError('Invalid type for artifact_id: {}'.format(artifact_id.__class__.__name__))

        if not isinstance(rev, str) and not isinstance(rev, unicode):
            raise ValueError('Invalid type for rev: {}'.format(rev.__class__.__name__))

        function_revision_output = self.repository_api.repository_get_function_revision(artifact_id, rev, self._get_expand_query_param_val(expand))
        if function_revision_output is not None:
            if function_revision_output is not None:
                return FunctionAdapter(function_revision_output[0], self.client).artifact()
            else:
                raise Exception('Function with guid={} not found'.format(artifact_id))
        else:
            raise Exception('Function with guid={} not found'.format(artifact_id))

    def revision_from_href(self, artifact_revision_href, expand = None):
        """
        Gets function revision from given href

        :param str artifact_revision_href: href identifying artifact and revision
        :return: returned object has all attributes of FunctionArtifact
        :rtype: FunctionArtifact(FunctionLoader)
        """

        if not isinstance(artifact_revision_href, str) and not isinstance(artifact_revision_href, unicode):
            raise ValueError('Invalid type for artifact_revision_href: {}'.format(artifact_revision_href.__class__.__name__))

        matched = re.search('.*/v4/functions/([A-Za-z0-9\-]+)/revisions/([A-Za-z0-9\-]+)',
                            artifact_revision_href)
        if matched is not None:
            artifact_id = matched.group(1)
            revision_id = matched.group(2)
            return self.revision(artifact_id, revision_id, expand)
        else:
            raise ValueError('Unexpected artifact revision href: {} format'.format(artifact_revision_href))

    def save(self, artifact):
        """
        Saves function in repository service.

        :param FunctionArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: FunctionArtifact
        """

        logger.debug('Creating a new function: {}'.format(artifact.name))

        if not issubclass(type(artifact), FunctionArtifact) :
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        if artifact.meta.prop(MetaNames.FUNCTIONS.REVISION_URL) is not None:
            raise ApiException(400, 'Invalid operation: save the same function artifact twice')

        try:
            function_id = artifact.uid
            if function_id is None:
                function_input = self._prepare_function_input(artifact)
                r = self.repository_api.repository_new_function(function_input)
                statuscode = r[1]
                if statuscode != 201:
                    logger.info('Error while creating function: no location header')
                    raise ApiException(404, 'No artifact location')

                function_artifact = self._extract_function_from_output(r)
                location = r[2].get('Location')
                location_match = re.search('.*/v4/functions/([A-Za-z0-9\\-]+)', location)

                if location_match is not None:
                    function_id = location_match.group(1)
                else:
                    logger.info('Error while creating function: no location header')
                    raise ApiException(404, 'No artifact location')
                artifact_with_guid = artifact._copy(function_id)
                revision_location = function_artifact.meta.prop(MetaNames.FUNCTIONS.REVISION_URL)
                revision_id = function_artifact.meta.prop(MetaNames.FUNCTIONS.REVISION)

                if revision_location is not None:
                    content_stream = artifact_with_guid.reader().read()
                    self.repository_api.upload_function_revision(function_id, revision_id, content_stream)
                    content_stream.close()
                    artifact_with_guid.reader().close()
                    return function_artifact
                else:
                    logger.info('Error while creating function revision: no location header')
                    raise ApiException(404, 'No artifact location')
            else:
                raise ApiException(404, 'Function not found')

        except Exception as e:
            logger.info('Error in function creation')
            import traceback
            print(traceback.format_exc())
            raise e

    def _extract_function_from_output(self, service_output):
        return FunctionAdapter(service_output[0], self.client).artifact()

    def patch(self, patch_input, latest_rev_url):
        matched = re.search('.*/v4/functions/([A-Za-z0-9\-]+)/revisions/([A-Za-z0-9\-]+)',
                            latest_rev_url)
        if matched is not None:
          function_id = matched.group(1)
          revision_id = matched.group(2)

        function_patch_input = self.prepare_function_patch_input(patch_input)
        function_patch_output = self.repository_api.repository_patch_functions_with_http_info(function_id, function_patch_input, revision_id)
        statuscode = function_patch_output[1]

        if statuscode != 200:
            logger.info('Error while patching function: no location header')
            raise ApiException(statuscode,"Error while patching function")

        if function_patch_output is not None:
            new_artifact = FunctionAdapter(function_patch_output[0], self.client).artifact()
        return new_artifact


    def remove(self, function_id):
        """
        Removes function with given id.

        :param str function_id: uid used to identify function
        """

        if not isinstance(function_id, str) and not isinstance(function_id, unicode):
            raise ValueError('Invalid type for function_id: {}'.format(function_id.__class__.__name__))

        if(function_id.__contains__("/v4/functions")):
            matched = re.search('.*/v4/functions/([A-Za-z0-9\-]+)', function_id)
            if matched is not None:
                function_id_value = matched.group(1)
                self.remove(function_id_value)
            else:
                raise ValueError('Unexpected function artifact href: {} format'.format(function_id))
        else:
            return self.repository_api.repository_delete_function(function_id)

    @staticmethod
    def prepare_function_patch_input(patch_input):
        patch_list = []
        if isinstance(patch_input, str):
            patch_input_list = json.loads(patch_input)
            if isinstance(patch_input_list, list):
                for iter1 in patch_input_list:
                    function_patch = PatchOperationFunctions(
                        op = iter1.get('op'),
                        path= iter1.get('path'),
                        value = iter1.get('value', None),
                        _from =iter1.get('from', None),
                    )
                    patch_list.append(function_patch)

                return patch_list

    @staticmethod
    def _prepare_function_input(artifact):
        function_input = MlAssetsCreateFunctionInput()
        function_input.name = artifact.name
        function_input.description = artifact.meta.prop(MetaNames.FUNCTIONS.DESCRIPTION)
        function_input.type = artifact.meta.prop(MetaNames.FUNCTIONS.TYPE)
        function_input.runtime_url = artifact.meta.prop(MetaNames.RUNTIMES.URL)

        input_data_schema = artifact.meta.prop(MetaNames.FUNCTIONS.INPUT_DATA_SCHEMA)
        output_data_schema = artifact.meta.prop(MetaNames.FUNCTIONS.OUTPUT_DATA_SCHEMA)
        sample_scoring_input = artifact.meta.prop(MetaNames.FUNCTIONS.SAMPLE_SCORING_INPUT)


        function_input.input_data_schema = input_data_schema
        function_input.output_data_schema = output_data_schema
        function_input.sample_scoring_input = sample_scoring_input
        if artifact.meta.prop(MetaNames.FUNCTIONS.TAGS) is not None:
            tags=artifact.meta.prop(MetaNames.FUNCTIONS.TAGS)
            tags_data_list = []
            if isinstance(tags, str):
                tags_list = json.loads(artifact.meta.prop(MetaNames.FUNCTIONS.TAGS))
                if isinstance(tags_list, list):
                    for iter1 in tags_list:
                        tags_data = TagRepository()
                        for key in iter1:
                            if key == 'value':
                                tags_data.value= iter1['value']
                            if key == 'description':
                                tags_data.description = iter1['description']
                        tags_data_list.append(tags_data)
                else:
                    raise ValueError("Invalid tag Input")
                function_input.tags =  tags_data_list

        return function_input

    @staticmethod
    def _get_expand_query_param_val(expand):
       if expand is True:
        expand_val = 'runtime'
       else:
        expand_val = None
       return expand_val