#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging
import re

from ibm_watson_machine_learning.libs.repo.mlrepository import  ModelArtifact
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames, MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepository.scikit_model_artifact import ScikitModelArtifact
from ibm_watson_machine_learning.libs.repo.mlrepository.tensorflow_model_artifact import TensorflowModelArtifact
from ibm_watson_machine_learning.libs.repo.mlrepository.generic_archive_model_artifact import GenericArchiveModelArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact import SparkPipelineModelLoader, SparkPipelineModelContentLoader,\
    IBMSparkPipelineModelContentLoader, MLPipelineModelContentLoader
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.content_loaders import ScikitPipelineModelContentLoader,TensorflowPipelineModelContentLoader
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.scikit_pipeline_model_loader import ScikitPipelineModelLoader
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.tensorflow_pipeline_model_loader import TensorflowPipelineModelLoader
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.generic_file_pipeline_model_loader import GenericFilePipelineModelLoader
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.hybrid_pipeline_model_artifact import HybridPipelineModelArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.hybrid_pipeline_model_loader import  HybridPipelineModelLoader
from ibm_watson_machine_learning.libs.repo.swagger_client.models import AuthorRepository
from ibm_watson_machine_learning.libs.repo.util.generic_archive_file_check import GenericArchiveFrameworkCheck
import ast,json

from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

logger = logging.getLogger('ModelAdapter')


class ModelAdapter(object):
    """
    Adapter creating pipeline model artifact using output from service.
    """

    @staticmethod
    def __strip_output(output):
        return ast.literal_eval(json.dumps(output))

    def __init__(self, model_output, model_version_output, client):
        self.model_output = model_output
        self.model_version_output = model_version_output
        self.client = client
        self.model_type = model_output.entity['type']
        self.model_entity = model_output.entity
        self.model_metadata = model_output.metadata

    def artifact(self):
        if re.match('mllib', self.model_type) is not None or 'mllib' in self.model_type:
            lib_checker.check_lib(PYSPARK)
            model_artifact_builder = type(
                "ModelArtifact",
                (SparkPipelineModelContentLoader, SparkPipelineModelLoader, ModelArtifact, object),
                {}
            )
        elif re.match('wml_1.', self.model_type) is not None or 'wml_1.' in self.model_type:
            lib_checker.check_lib(PYSPARK)
            lib_checker.check_lib(MLPIPELINE)
            model_artifact_builder = type(
                "WMLModelArtifact",
                (MLPipelineModelContentLoader, SparkPipelineModelLoader, ModelArtifact, object),
                {}
            )
        elif re.match('scikit-learn', self.model_type) is not None or 'scikit-learn' in self.model_type:
            lib_checker.check_lib(SCIKIT)
            model_artifact_builder = type(
                "ScikitModelArtifact",
                (ScikitPipelineModelContentLoader, ScikitPipelineModelLoader, ScikitModelArtifact, object),
                {}
            )
        elif re.match('tensorflow', self.model_type) is not None or 'tensorflow' in self.model_type:
            lib_checker.check_lib(TENSORFLOW)
            model_artifact_builder = type(
                "TensorflowModelArtifact",
                (TensorflowPipelineModelContentLoader, TensorflowPipelineModelLoader, TensorflowModelArtifact, object),
                {}
            )

        elif re.match('xgboost', self.model_type) is not None or 'xgboost' in self.model_type:
            lib_checker.check_lib(SCIKIT)
            lib_checker.check_lib(XGBOOST)
            model_artifact_builder = type(
                "ScikitModelArtifact",
                (ScikitPipelineModelContentLoader, ScikitPipelineModelLoader, ScikitModelArtifact, object),
                {}
            )
        elif re.match('hybrid', self.model_type) is not None or \
             re.match('wml-hybrid', self.model_type) is not None or \
             'hybrid' in  self.model_type or 'wml-hybrid' in self.model_type:
            model_artifact_builder = type(
                "HybridModelArtifact",
                (HybridPipelineModelLoader, HybridPipelineModelArtifact,object),
                {}
            )
        elif GenericArchiveFrameworkCheck.is_archive_framework(self.model_type):
            model_artifact_builder = type(
                "GenericModelArchiveArtifact",
                (GenericFilePipelineModelLoader, GenericArchiveModelArtifact, object),
                {}
            )
        else:
            if isinstance(self.model_type, str):
                raise ValueError('Invalid model_type: {}'.format(self.model_type))
            else:
                raise ValueError('Invalid model_type: {}'.format(self.model_type.get('name')))

        prop_map = {
            MetaNames.TYPE: self.model_type
        }

        if self.model_metadata.created_at is not None:
            prop_map[MetaNames.CREATION_TIME] = self.model_output.metadata.created_at

        if self.model_output.metadata.modified_at is not None:
            prop_map[MetaNames.LAST_UPDATED] = self.model_output.metadata.modified_at

        if self.model_entity.get('domain', None) is not None:
            prop_map[MetaNames.DOMAIN] = self.model_entity.get('domain')

        if self.model_entity.get('custom', None) is not None:
            prop_map[MetaNames.CUSTOM] = self.model_entity.get('custom')

        if self.model_entity.get('label_column', None) is not None:
            prop_map[MetaNames.LABEL_FIELD] = self.model_entity.get('label_column')

        if self.model_entity.get('description', None) is not None:
            prop_map[MetaNames.DESCRIPTION] = self.model_entity.get('description')

        if self.model_entity.get('transformed_label', None) is not None:
            prop_map[MetaNames.TRANSFORMED_LABEL_FIELD] = self.model_entity.get('transformed_label')

        if self.model_entity.get('tags', None) is not None:
            prop_map[MetaNames.TAGS] = self.model_entity.get('tags')

        if self.model_entity.get('size', None) is not None:
            prop_map[MetaNames.SIZE] = self.model_entity.get('size')

        if "training_data_references" in self.model_entity and \
                self.model_entity.get('training_data_references', None) is not None:
            prop_map[MetaNames.TRAINING_DATA_REFERENCES] = self.model_entity.get('training_data_references')


        #v4 cloud specific model artifact:

        href_tmp = getattr(self.model_metadata, 'href', 'None')

        if self.model_metadata.id is not None and href_tmp is None:
            if self.model_entity.get('pipeline') is not None:
                runtimeval = self.model_entity.get('pipeline')
                if runtimeval['id'] is not None:
                    prop_map[MetaNames.PIPELINE_UID] = self.model_entity.get('pipeline')['id']

            if self.model_entity.get('model_definition') is not None:
                runtimeval = self.model_entity.get('model_definition')
                if runtimeval['id'] is not None:
                    prop_map[MetaNames.model_definition] = self.model_entity.get('model_definition')['id']

            if self.model_metadata.space_id is not None:
                prop_map[MetaNames.SPACE_ID] = self.model_metadata.space_id

            if self.model_metadata.project_id is not None:
                prop_map[MetaNames.PROJECT_ID] = self.model_metadata.project_id

            if self.model_entity.get('hyper_parameters') is not None:
                prop_map.add(MetaNames.HYPER_PARAMETERS, self.model_entity['hyper_parameters'])

            if self.model_entity.get('software_spec') is not None:
                runtimeval = self.model_entity.get('software_spec')
                if runtimeval['id'] is not None:
                    prop_map[MetaNames.SOFTWARE_SPEC] = self.model_entity.get('software_spec')['id']
            if self.model_entity.get('schemas') is not None and self.model_entity['schemas']['input'] is not None:
                prop_map[MetaNames.INPUT_DATA_SCHEMA] = self.model_entity['schemas']['input']

            if self.model_entity.get('schemas') is not None and self.model_entity['schemas']['output'] is not None:
                prop_map[MetaNames.OUTPUT_DATA_SCHEMA] = self.model_entity['schemas']['output']

            if 'metrics' in self.model_entity and self.model_entity['metrics'] is not None:
                prop_map[MetaNames.METRICS] = self.model_entity['metrics']

            if 'content_status' in self.model_entity:
                if self.model_entity['content_status'] is not None:
                    prop_map[MetaNames.CONTENT_STATUS] = self.model_entity['content_status']

            #TODO : Enable this, when we support import
            # if 'import' in self.model_entity:
            #     if self.model_entity['import'] is not None:
            #         prop_map.add(MetaNames.IMPORT, self.model_entity['import'])

            name = self.model_metadata.name
            model_id = self.model_metadata.id

            model_artifact = model_artifact_builder(
                uid=model_id,
                name=name,
                meta_props=MetaProps(prop_map)
            )

            model_artifact.client = self.client
            model_artifact._content_href = "/ml/v4/models/" + self.model_metadata.id + "/content"
            model_artifact._download_href = "/ml/v4/models/" + self.model_metadata.id + \
                                            "/download?version=2021-06-24"
            # TODO Don't use hardcoded version.
            # ( self.client.version_param <- this is not APIClient it is MLRepositoryClient)
        else:
            if self.model_entity.get('runtime') is not None:
                runtimeval = self.model_entity.get('runtime')
                if runtimeval['href'] is not None:
                    prop_map[MetaNames.RUNTIMES] = self.model_entity.get('runtime')['href']

            if self.model_entity.get('pipeline') is not None:
                runtimeval = self.model_entity.get('pipeline')
                if runtimeval['href'] is not None:
                    prop_map[MetaNames.PIPELINE_UID] = self.model_entity.get('pipeline')['href']

            if self.model_entity.get('training_lib') is not None:
                runtimeval = self.model_entity.get('training_lib')
                if runtimeval['href'] is not None:
                    prop_map[MetaNames.TRAINING_LIB_UID] = self.model_entity.get('training_lib')['href']
            if self.model_entity.get('space') is not None:
                runtimeval = self.model_entity.get('space')
                if runtimeval['href'] is not None:
                    prop_map[MetaNames.SPACE_UID] = self.model_entity.get('space')['href']

            if self.model_entity.get('project') is not None:
                projectval = self.model_entity.get('project')
                if projectval['href'] is not None:
                    prop_map[MetaNames.PROJECT_UID] = self.model_entity.get('project')['href']

               # TODO Fix Evaluation
        #         try:
        #             evaluation_data = self.model_entity.get('evaluation', None)
        #             if evaluation_data is not None:
        #                 prop_map[MetaNames.EVALUATION_METHOD] = evaluation_data.get('method', None)
        #                 if evaluation_data.get('metrics', None) is not None:
        #                     prop_map[MetaNames.EVALUATION_METRICS] = evaluation_data.get('metrics')
        #         #    evaluation_data = self.model_entity['evaluation']
        #         #    if evaluation_data is not None:t
        #         #        prop_map[MetaNames.EVALUATION_METHOD] = evaluation_data['method']
        #         #        if evaluation_data['metrics'] is not None:
        #         #           metrics = evaluation_data['metrics']
        #         #            prop_map[MetaNames.EVALUATION_METRICS] = metrics
        #         except KeyError:
        #             print("No Evlauation method given")

        #name = ''.join(map(lambda x: chr(ord(x)), self.model_output.entity['name']))
            name = self.model_entity.get('name')

            model_id = self.model_metadata.guid

            model_artifact = model_artifact_builder(
                uid=model_id,
                name=name,
                meta_props=MetaProps(prop_map)
            )

            model_artifact.client = self.client

            if model_artifact.meta.prop(MetaNames.TYPE).startswith('mllib'):
                model_artifact._pipeline_url = self.model_entity.get('pipeline')['href']

            if model_artifact.meta.prop(MetaNames.TYPE).startswith('wml_1.'):
                model_artifact._pipeline_url = self.model_entity.get('pipeline')['href']

            if self.model_entity is not None:
                version_props = MetaProps({})
                # content_status, content_location = None, None
                if 'content_status' in self.model_entity:
                    if self.model_entity['content_status'] is not None:
                        version_props.add(MetaNames.CONTENT_STATUS, self.model_entity['content_status'])

                if 'import' in self.model_entity:
                    if self.model_entity['import'] is not None:
                        version_props.add(MetaNames.IMPORT, self.model_entity['import'])

                if 'hyper_parameters' in self.model_entity:
                    if self.model_entity['hyper_parameters'] is not None:
                        version_props.add(MetaNames.HYPER_PARAMETERS, self.model_entity['hyper_parameters'])
                        version_props.add(MetaNames.HYPER_PARAMETERS, self.model_entitya['hyper_parameters'])

                model_artifact.meta.merge(version_props)
                model_artifact._content_href = "/v4/models/"+self.model_metadata.guid+"/content"
            else:
                model_artifact._content_href = None

        return model_artifact
