#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json,ast
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames, MetaProps
from ibm_watson_machine_learning.libs.repo.swagger_client.models.ml_assets_create_functions_output import MlAssetsCreateFunctionsOutput
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.function_loader import FunctionLoader
from ibm_watson_machine_learning.libs.repo.mlrepository.wml_function_artifact import WmlFunctionArtifact
lib_checker = LibraryChecker()

class FunctionAdapter(object):
    """
    Adapter creating function artifact using output from service.
    """
    @staticmethod
    def __strip_output(output):
        return ast.literal_eval(json.dumps(output))

    def __init__(self, function_output, client):
        self.function_output = function_output
        self.client = client
        self.function_entity = self.function_output.entity
        if function_output is not None and not isinstance(function_output, MlAssetsCreateFunctionsOutput):
            raise ValueError('Invalid type for pipeline: {}'.format(function_output.__class__.__name__))


    def artifact(self):

        function_artifact_builder = type(
                "WmlFunctionArtifact",
                (FunctionLoader, WmlFunctionArtifact, object),
                {}
            )

        prop_map = {
            MetaNames.CREATION_TIME: self.function_output.metadata.created_at,
            MetaNames.LAST_UPDATED: self.function_output.metadata.modified_at
        }

        if self.function_entity.get('description', None) is not None:
            prop_map[MetaNames.FUNCTIONS.DESCRIPTION] = self.function_entity['description']

        if self.function_entity.get('tags', None) is not None:
            prop_map[MetaNames.FUNCTIONS.TAGS] = self.function_entity.get('tags')

        if self.function_entity.get('type', None) is not None:
            prop_map[MetaNames.FUNCTIONS.TYPE] = self.function_entity.get('type')

        if self.function_entity.get('runtime', None) is not None:
            if self.function_entity.get('runtime').get('url') is not None:
              prop_map[MetaNames.RUNTIMES.URL] = self.function_entity.get('runtime')['url']

            if self.function_entity.get('runtime').get('content_url') is not None:
              prop_map[MetaNames.RUNTIMES.CONTENT_URL] = self.function_entity.get('runtime')['content_url']

            if self.function_entity.get('runtime').get('custom_libraries') is not None:
              prop_map[MetaNames.RUNTIMES.CUSTOM_LIBRARIES_URLS] = self.function_entity.get('runtime')['custom_libraries']

            if self.function_entity.get('runtime').get('description') is not None:
              prop_map[MetaNames.RUNTIMES.DESCRIPTION] = self.function_entity.get('runtime')['description']

            if self.function_entity.get('runtime').get('platform') is not None:
             prop_map[MetaNames.RUNTIMES.PLATFORM] = self.function_entity.get('runtime')['platform']

            if self.function_entity.get('runtime').get('name') is not None:
             prop_map[MetaNames.RUNTIMES.NAME] = self.function_entity.get('runtime')['name']

        if self.function_entity.get('input_data_schema', None) is not None:
           prop_map[MetaNames.FUNCTIONS.INPUT_DATA_SCHEMA] = self.function_entity.get('input_data_schema')

        if self.function_entity.get('output_data_schema', None) is not None:
           prop_map[MetaNames.FUNCTIONS.OUTPUT_DATA_SCHEMA] = self.function_entity.get('output_data_schema')

        if self.function_entity.get('sample_scoring_input', None) is not None:
            prop_map[MetaNames.FUNCTIONS.SAMPLE_SCORING_INPUT] = self.function_entity.get('sample_scoring_input')


        name = self.function_entity.get('name', None)

        function_url = self.function_output.metadata.url
        function_id = function_url.split("/functions/")[1].split("/")[0]

        function_artifact = function_artifact_builder(
            function_id,
            name,
            MetaProps(prop_map))

        function_artifact.client = self.client

        if self.function_output.entity.get('function_revision') is not None:
            if self.function_output.entity['function_revision'].get('content_url') is not None:
              prop_map[MetaNames.FUNCTIONS.CONTENT_URL] = self.function_output.entity['function_revision']['content_url']
        else:
            prop_map[MetaNames.FUNCTIONS.CONTENT_URL] =   self.function_output.entity['content_url']

        if self.function_output.entity.get('function_revision') is not None:
            self.revision_output = self.function_output.entity['function_revision']
            revision_url = self.revision_output['url']
            function_url = revision_url.split("/revisions")[0]

            revision_id = revision_url.split("/revisions/")[1].split("/")[0]

            version_prop_map = {
                MetaNames.FUNCTIONS.REVISION: revision_id,
                MetaNames.FUNCTIONS.REVISION_URL:revision_url,
                MetaNames.FUNCTIONS.URL:function_url
            }

        else:
            revision_url = self.function_output.metadata.url
            revision_id = revision_url.split("/revisions/")[1].split("/")[0]
            function_url = revision_url.split("/revisions")[0]

            version_prop_map = {
                MetaNames.FUNCTIONS.REVISION: revision_id,
                MetaNames.FUNCTIONS.REVISION_URL:revision_url,
                MetaNames.FUNCTIONS.URL:function_url
            }


        function_artifact.meta.merge(MetaProps(version_prop_map))
        function_artifact._content_href = prop_map[MetaNames.FUNCTIONS.CONTENT_URL]
   
        return function_artifact
