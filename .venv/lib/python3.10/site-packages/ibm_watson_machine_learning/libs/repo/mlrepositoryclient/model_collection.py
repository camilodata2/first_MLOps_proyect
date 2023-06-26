#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging
import re
import json
import os
import copy
import tarfile, shutil

from .model_adapter import ModelAdapter
from ibm_watson_machine_learning.libs.repo.swagger_client.rest import ApiException
from ibm_watson_machine_learning.libs.repo.mlrepository import  ModelArtifact
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.swagger_client.models import ModelInput, ModelVersionInput, ModelTrainingDataRef, ModelVersionOutput, ModelDefinitionModels, \
    MetaObjectMetadata, ModelVersionOutputEntity, ModelVersionOutputEntityModel, ArtifactAuthor, EvaluationDefinition, TagRepository, ModelContentLocation, TrainingModels,SpaceModels,ModelSchemas,PipelinesModels,ModelContentLocation,RuntimeModels,SoftwareSpecModels,ModelsCustom,ModelsMetrics,ModelSchemas,ModelsSize
from ibm_watson_machine_learning.libs.repo.swagger_client.models import EvaluationDefinitionMetrics, ConnectionObjectWithNameRepository, ArrayDataInputRepository, EvaluationDefinitionRepositoryMetrics
from ibm_watson_machine_learning.libs.repo.swagger_client.models import MlAssetsCreateModelInput, FrameworkOutputRepository, AuthorRepository, EvaluationDefinitionRepository, FrameworkOutputRepositoryLibraries,FrameworkOutputRepositoryRuntimes
from ibm_watson_machine_learning.libs.repo.swagger_client.models import ContentLocation,HyperParameters
from ibm_watson_machine_learning.libs.repo.util.json_2_object_mapper import Json2ObjectMapper
from ibm_watson_machine_learning.libs.repo.util.exceptions import UnsupportedTFSerializationFormat,InvalidCaffeModelArchive
from ibm_watson_machine_learning.libs.repo.swagger_client.models import  MlAssetsModelSizeOutput
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate
from ibm_watson_machine_learning.libs.repo.swagger_client.models.framework_output_repository_runtimes import FrameworkOutputRepositoryRuntimes


logger = logging.getLogger('ModelCollection')


class ModelCollection:
    """
    Client operating on models in repository service.

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
        Gets info about all models which belong to this user.

        Not complete information is provided by all(). To get detailed information about model use get().

        :return: info about models
        :rtype: list[ModelArtifact]
        """
        logger.debug('Fetching information about all models')
        all_models = self.repository_api.repository_list_models(queryMap)
        list_model_artifact = []
        if all_models is not None:
            resr = all_models.resources
            for iter1 in resr:
                model_entity = iter1.entity
                ver_url = iter1.entity['model_version']
                list_model_artifact.append(ModelAdapter(iter1, ver_url, self.client).artifact())
            return list_model_artifact
        else:
            return []

    def get(self, artifact_id, space_id=None, project_id=None):
        """
        Gets detailed information about model.

        :param str artifact_id: uid used to identify model
        :return: returned object has all attributes of SparkPipelineModelArtifact but its class name is ModelArtifact
        :rtype: ModelArtifact(SparkPipelineModelLoader)
        """
        logger.debug('Fetching information about pipeline model: {}'.format(artifact_id))

        if not isinstance(artifact_id, str) and not isinstance(artifact_id, unicode):
            raise ValueError('Invalid type for artifact_id: {}'.format(artifact_id.__class__.__name__))
        if space_id is not None:
            model_output = self.repository_api.v3_ml_assets_models_artifact_id_get(artifact_id, space_id=space_id)
        else:
            if project_id is not None:
                model_output = self.repository_api.v3_ml_assets_models_artifact_id_get(artifact_id, project_id=project_id)
            else:
                model_output = self.repository_api.v3_ml_assets_models_artifact_id_get(artifact_id)

        if model_output is not None:
            latest_version = model_output.metadata.href
            return ModelAdapter(model_output, latest_version, self.client).artifact()
        else:
            logger.debug('Model with guid={} not found'.format(artifact_id))
            raise ApiException('Model with guid={} not found'.format(artifact_id))

    def versions(self, artifact_id):
        """
        Gets all available versions.

        Not implemented yet.

        :param str artifact_id: uid used to identify model
        :return: ???
        :rtype: list[str]
        """

        if not isinstance(artifact_id, str) and not isinstance(artifact_id, unicode):
            raise ValueError('Invalid type for artifact_id: {}'.format(artifact_id.__class__.__name__))

        logger.debug('Fetching information about pipeline model: {}'.format(artifact_id))

        model_output = self.repository_api.repository_list_model_versions(artifact_id)

        list_model_version_artifact = [ModelArtifact]
        if model_output is not None:
            resr = model_output.resources
            for iter1 in resr:
                model_entity = iter1.entity
                ver_url = iter1.entity['model_version']
                list_model_version_artifact.append(ModelAdapter(iter1, iter1.entity['model_version'], self.client).artifact())
            return list_model_version_artifact
        else:
            logger.debug('Model with guid={} not found'.format(artifact_id))
            raise ApiException('Model with guid={} not found'.format(artifact_id))

    def version(self, artifact_id, ver):
        """
        Gets model version with given artifact_id and ver
        :param str artifact_id: uid used to identify model
        :param str ver: uid used to identify version of model
        :return: ModelArtifact(SparkPipelineModelLoader) -- returned object has all attributes of SparkPipelineModelArtifact but its class name is ModelArtifact
        """
        logger.debug('Fetching information about model version: {}, {}'.format(artifact_id, ver))

        if not isinstance(artifact_id, str) and not isinstance(artifact_id, unicode):
            raise ValueError('Invalid type for artifact_id: {}'.format(artifact_id.__class__.__name__))

        if not isinstance(ver, str) and not isinstance(ver, unicode):
            raise ValueError('Invalid type for ver: {}'.format(ver.__class__.__name__))

        model_version_output = self.repository_api.repository_get_model_version(artifact_id, ver)
        if model_version_output is not None:
            if model_version_output is not None:
                return ModelAdapter(model_version_output, model_version_output.entity['model_version'], self.client).artifact()
            else:
                raise Exception('Model with guid={} not found'.format(artifact_id))
        else:
            raise Exception('Model with guid={} not found'.format(artifact_id))

    def version_from_href(self, artifact_version_href):
        """
        Gets model version from given href

        :param str artifact_version_href: href identifying artifact and version
        :return: returned object has all attributes of SparkPipelineModelArtifact but its class name is PipelineModelArtifact
        :rtype: PipelineModelArtifact(SparkPipelineModelLoader)
        """

        if not isinstance(artifact_version_href, str) and not isinstance(artifact_version_href, unicode):
            raise ValueError('Invalid type for artifact_version_href: {}'.format(artifact_version_href.__class__.__name__))

        #if artifact_version_href.startswith(self.base_path):
        matched = re.search('.*/v3/ml_assets/models/([A-Za-z0-9\-]+)/versions/([A-Za-z0-9\-]+)',
                            artifact_version_href)
        matchedV2 = re.search('.*/v2/artifacts/models/([A-Za-z0-9\-]+)/versions/([A-Za-z0-9\-]+)',
                              artifact_version_href)
        if matched is not None:
            artifact_id = matched.group(1)
            version_id = matched.group(2)
            return self.version(artifact_id, version_id)
        elif matchedV2 is not None:
            artifact_id = matchedV2.group(1)
            version_id = matchedV2.group(2)
            return self.version(artifact_id, version_id)
        else:
            raise ValueError('Unexpected artifact version href: {} format'.format(artifact_version_href))

    def remove(self, artifact_id):
        """
        Removes model with given artifact_id.

        :param str artifact_id: uid used to identify model
        """

        if not isinstance(artifact_id, str):
            raise ValueError('Invalid type for artifact_id: {}'.format(artifact_id.__class__.__name__))

        return self.repository_api.v3_ml_assets_models_artifact_id_delete(artifact_id)

    def save(self, artifact, query_param=None):

        if artifact.meta.prop(MetaNames.TYPE).startswith("scikit-learn"):
            return self._save_scikit_pipeline_model(artifact, query_param)
        elif artifact.meta.prop(MetaNames.TYPE).startswith("xgboost"):
            return self._save_xgboost_model(artifact, query_param)
        elif artifact.meta.prop(MetaNames.TYPE).startswith("tensorflow"):
            if isinstance(artifact.ml_pipeline_model, str):
                return self._save_tensorflow_pipeline_model_tar(artifact, query_param)
            else:
                return self._save_tensorflow_pipeline_model(artifact, query_param)
        elif MetaNames.is_archive_framework(artifact.meta.prop(MetaNames.TYPE)):
            if isinstance(artifact.ml_pipeline_model, str):
               return self._save_generic_archive_pipeline_model(artifact, query_param)
            elif artifact.meta.prop(MetaNames.TYPE).startswith("mllib"):
                return self._save_spark_pipeline_model(artifact, query_param)
            else:
                raise ValueError('Invalid type for artifact_id: {}'.format(artifact.__class__.__name__))
        elif artifact.meta.prop(MetaNames.TYPE).startswith("hybrid"):
            return self._save_hybird_pipeline_model(artifact, query_param)
        else:
            return self._save_spark_pipeline_model(artifact, query_param)

    def upload_content(self,model_artifact, query_param, no_delete=None):
        self._upload_pipeline_model_content(model_artifact, query_param, no_delete)

    def _save_scikit_pipeline_model(self, artifact, query_param=None):
        """
        Saves model in repository service.

        :param ScikitPipelineModelArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: ScikitPipelineModelArtifact
        """
        logger.debug('Creating a new scikit pipeline model: {}'.format(artifact.name))

        if not issubclass(type(artifact), ModelArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        if artifact.meta.prop(MetaNames.MODEL_VERSION_URL) is not None:
            raise ApiException(400, 'Invalid operation: save the same model artifact twice')

        try:
            if artifact.uid is None:
                model_artifact = self._create_pipeline_model(artifact, query_param)
            else:
                model_artifact = artifact

            if model_artifact.uid is None:
                raise RuntimeError('Internal Error: Model without ID')
            else:
                return model_artifact
        except Exception as e:
            logger.info('Error in pipeline model creation')
            import traceback
            print(traceback.format_exc())
            raise e

    def _save_hybird_pipeline_model(self, artifact, query_param=None):
        """
        Saves model in repository service.

        :param ScikitPipelineModelArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: ScikitPipelineModelArtifact
        """
        logger.debug('Creating a new hybrid model: {}'.format(artifact.name))

        if not issubclass(type(artifact), ModelArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        if artifact.meta.prop(MetaNames.MODEL_VERSION_URL) is not None:
            raise ApiException(400, 'Invalid operation: save the same model artifact twice')

        try:
            if artifact.uid is None:
                model_artifact = self._create_pipeline_model(artifact, query_param)
            else:
                model_artifact = artifact

            if model_artifact.uid is None:
                raise RuntimeError('Internal Error: Model without ID')
            else:
                return model_artifact
        except Exception as e:
            logger.info('Error in hybrid model creation')
            import traceback
            print(traceback.format_exc())


    def _save_tensorflow_pipeline_model(self, artifact, query_param=None):
        """
        Saves model in repository service.

        :param ScikitPipelineModelArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: ScikitPipelineModelArtifact
        """
        logger.debug('Creating a new tensorflow pipeline model: {}'.format(artifact.name))

        if not issubclass(type(artifact), ModelArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        if artifact.meta.prop(MetaNames.MODEL_VERSION_URL) is not None:
            raise ApiException(400, 'Invalid operation: save the same model artifact twice')

        try:
            if artifact.uid is None:
                model_artifact = self._create_pipeline_model(artifact, query_param)
            else:
                model_artifact = artifact

            if model_artifact.uid is None:
                raise RuntimeError('Internal Error: Model without ID')
            else:
                return model_artifact
        except Exception as e:
            logger.info('Error in pipeline model creation')
            import traceback
            print(traceback.format_exc())
            raise e

    def _save_tensorflow_pipeline_model_tar(self, artifact, query_param=None):
        """
        Saves model in repository service.

        :param TensorflowPipelineModelTarArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: TensorflowPipelineModelTarArtifact
        """
        logger.debug('Creating a new tensorflow model artifact: {}'.format(artifact.name))

        if not issubclass(type(artifact), ModelArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        if artifact.meta.prop(MetaNames.MODEL_VERSION_URL) is not None:
            raise ApiException(400, 'Invalid operation: Attempted to save the same model artifact twice')

        # validate if the artifact supplied is a valid artifact for Tensorflow

        keras_version = artifact.get_keras_version()
        if keras_version is not None:
            artifact.update_keras_version_meta(keras_version)

        if (not artifact.is_valid_tf_archive()) and keras_version is None:
            raise UnsupportedTFSerializationFormat('The specified compressed archive is invalid. Please ensure the '
                                                   'Tensorflow model is serialized using '
                                                   'tensorflow.saved_model.builder.SavedModelBuilder API. If using '
                                                   'Keras, ensure save() of is used to save the model')

        try:
            if artifact.uid is None:
                model_artifact = self._create_pipeline_model(artifact, query_param)
            else:
                model_artifact = artifact

            if model_artifact.uid is None:
                raise RuntimeError('Internal Error: Model without ID')
            else:
                return model_artifact
        except Exception as e:
            logger.info('Error in pipeline model creation')
            import traceback
            print(traceback.format_exc())
            raise e

    def _save_generic_archive_pipeline_model(self, artifact, query_param = None):
        """
        Saves model in repository service.

        :param GenericArchiveModelArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: GenericArchiveModelArtifact
        """

        logger.debug('Creating a new archive model artifact: {}'.format(artifact.name))

        if not issubclass(type(artifact), ModelArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        if artifact.meta.prop(MetaNames.MODEL_VERSION_URL) is not None:
            raise ApiException(400, 'Invalid operation: Attempted to save the same model artifact twice')

        if not os.path.exists(artifact.ml_pipeline_model):
            raise IOError('The artifact specified ( {} ) does not exist.'.format(artifact.ml_pipeline_model))

        if "caffe" in artifact.meta.prop(MetaNames.TYPE):
            extracted_path= artifact.extract_tar_file()
            if (not artifact.is_valid_caffe_archive(extracted_path)) is None:
                raise InvalidCaffeModelArchive('The specified compressed caffe model archive is invalid.')


        try:
            if artifact.uid is None:
                model_artifact = self._create_pipeline_model(artifact, query_param)
            else:
                model_artifact = artifact

            if model_artifact.uid is None:
                raise RuntimeError('Internal Error: Model without ID')
            else:
                return model_artifact
        except Exception as e:
            logger.info('Error in pipeline model creation')
            import traceback
            print(traceback.format_exc())
            raise e

    def _save_xgboost_model(self, artifact, query_param = None):
        """
        Saves model in repository service.

        :param ScikitPipelineModelArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: ScikitPipelineModelArtifact
        """
        logger.debug('Creating a new xgboost model: {}'.format(artifact.name))

        if not issubclass(type(artifact), ModelArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        if artifact.meta.prop(MetaNames.MODEL_VERSION_URL) is not None:
            raise ApiException(400, 'Invalid operation: save the same model artifact twice')

        try:
            if artifact.uid is None:
                model_artifact = self._create_pipeline_model(artifact, query_param)
            else:
                model_artifact = artifact

            if model_artifact.uid is None:
                raise RuntimeError('Internal Error: Model without ID')
            else:
                return model_artifact

        except Exception as e:
            logger.info('Error in pipeline model creation')
            import traceback
            print(traceback.format_exc())
            raise e

    def _save_spark_pipeline_model(self, artifact, query_param=None):
        """
        Saves model in repository service.

        :param SparkPipelineModelArtifact artifact: artifact to be saved in the repository service
        :return: saved artifact with changed MetaProps
        :rtype: SparkPipelineModelArtifact
        """
        logger.debug('Creating a new pipeline model: {}'.format(artifact.name))

        if not issubclass(type(artifact), ModelArtifact):
            raise ValueError('Invalid type for artifact: {}'.format(artifact.__class__.__name__))

        #if artifact.meta.prop(MetaNames.MODEL_VERSION_URL) is not None:
        #    raise ApiException(400, 'Invalid operation: save the same model artifact twice')
        try:
            if artifact.uid is None:
                model_artifact = self._create_pipeline_model(artifact, query_param)
            else:
                model_artifact = artifact

            if model_artifact.uid is None:
                raise RuntimeError('Internal Error: Model without ID')
            else:
                return model_artifact
        except Exception as e:
            logger.info('Error in pipeline model creation')
            import traceback
            print(traceback.format_exc())
            raise e

    def _get_experiment_id_from_url(self, artifact_version_url):
        """
        Gets experiment id from given url

        :return: returned object has all attributes of SparkPipelineArtifact but its class name is PipelineArtifact
        """

        if not isinstance(artifact_version_url, str):
            raise ValueError('Invalid type for artifact_version_href: {}'
                             .format(artifact_version_url.__class__.__name__))

        #if artifact_version_url.startswith(self.base_path):
        matched = re.search(
            '.*/v3/ml_assets/training_definitions/([A-Za-z0-9\-]+)/versions/([A-Za-z0-9\-]+)', artifact_version_url)
        if matched is not None:
            experiment_id = matched.group(1)
            return experiment_id
        else:
            raise ValueError('Unexpected artifact version url in metaprop: {} format'.format(artifact_version_url))
            #else:
            #    raise ValueError('The artifact version href: {} is not within the client host: {}').format(
            #        artifact_version_url,
            #        self.base_path
            #    )

    def _get_pipeline(self, pipeline_version_url):
        return self.client.pipelines.version_from_href(pipeline_version_url)

    def _create_pipeline(self, pipeline_artifact):
        return self.client.pipelines.save(pipeline_artifact)

    def _create_pipeline_model(self, model_artifact, query_param=None):
        model_artifact = self._create_pipeline_model_v4_cloud(model_artifact, query_param)
        return model_artifact

    def _create_pipeline_model_version(self, model_artifact, query_param=None):
        model_version_input = self._get_version_input(model_artifact)
        r = self.repository_api.repository_model_version_creation(model_artifact.uid, model_version_input)
        location = r[2].get('Location')
        if location is not None:
            logger.debug('New model version created at: {}'.format(location))
            try:
                new_version_artifact = self.version_from_href(location)
                new_version_artifact.model_instance = lambda: model_artifact.ml_pipeline_model
                model_artifact_with_version = model_artifact._copy(meta_props=new_version_artifact.meta)
                if MetaNames.CONTENT_LOCATION not in model_artifact_with_version.meta.get():
                    status_url = self._upload_pipeline_model_content(model_artifact_with_version, query_param)
                    if status_url is not None and status_url != "":
                        place_holder =model_artifact_with_version.meta.add(MetaNames.STATUS_URL, status_url)
                        new_async_artifact = model_artifact_with_version._copy(meta_props=place_holder)
                        return new_async_artifact
                    else:
                        return new_version_artifact
            except Exception as ex:
                raise ex
        else:
            logger.info('Location of the new model version not found')
            raise ApiException(404, 'No artifact location')

    def _upload_pipeline_model_content(self, model_artifact, query_param=None, no_delete=None ):
        # if query_param is not None and 'version' in query_param.keys():
        #     model_id = model_artifact.id
        # else:
        model_id = model_artifact.uid
        #version_id = model_artifact.meta.prop(MetaNames.VERSION)
        asyncValue = "false"
        if query_param is not None:
            for key in query_param:
                if (key != 'async' and key != 'space_id' and key != 'project_id'  and key != 'version' and
                        key != 'content_format' and key != 'userfs'):
                    raise ValueError("Got an unexpected keyword argument '%s'" % key)
                else:
                    if (key == 'async'):
                        asyncValue = query_param[key]
                        if (asyncValue != 'true' and asyncValue != 'false'):
                            raise ValueError(
                                "Got an unexpected value '%s' for keyword argument '%s'" % (asyncValue, key))
        # if version_id is None:
        #     raise RuntimeError('Model meta `{}` not set'.format(MetaNames.VERSION))

        content_stream = model_artifact.reader().read()
        if(query_param is not None and asyncValue == 'true'):

                def upload_call():
                    self.repository_api.upload_pipeline_model_version(model_id, None, content_stream, query_param)

                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future = executor.submit(upload_call)
                    try:
                        future.result()
                        logger.info("After future.result future completed:%s "%(str(future.done())))
                    except Exception as exc:
                         logger.info(' Upload to version_id %r generated  exception: %s' % (model_id, exc))
                    else:
                         logger.info("Upload to version id %r success with response." %model_id)


                model_url = model_artifact.meta.prop(MetaNames.MODEL_URL)
                status_url = '%s/content/status' %(model_url)

        else:
            if (query_param is not None and 'space_id' in query_param.keys()):
                logger.info("In case , where space_id is passed.")
                if ('version' in query_param.keys()):
                    self.repository_api.upload_pipeline_model_v4_cloud(model_id, content_stream, query_param)
                else:
                    self.repository_api.upload_pipeline_model_version(model_id, None, content_stream, query_param)

                status_url = ""
            elif (query_param is not None and 'project_id' in query_param.keys()):
                logger.info("In case , where space_id is passed.")
                if ('version' in query_param.keys()):
                    self.repository_api.upload_pipeline_model_v4_cloud(model_id, content_stream, query_param, no_delete)
                else:
                    self.repository_api.upload_pipeline_model_version(model_id, None, content_stream, query_param)
                status_url = ""
            else:
                logger.info("In case of either no query param passed  or 'async':'false' is passed")
                self.repository_api.upload_pipeline_model_version(model_id, None, content_stream)
                status_url = ""

        content_stream.close()
        model_artifact.reader().close()
        logger.debug('Content uploaded for model version created at: {}'.format(model_id))
        return status_url

    def get_status(self, url):
        polling_status = self.repository_api.get_async_status(url)
        if polling_status is not None:
            if(polling_status.entity['status_message']=="Running" or polling_status.entity['status_message'] == "Completed"):
                return polling_status.entity['status_message']
            else:
                error_msg=("'Status' = 'ERROR', 'Message' = '%s'" %polling_status.entity['status_message'])
                return error_msg

        else:
            raise ValueError("Request with url ='%s'failed" %url)

    @staticmethod
    def _prepare_model_input(artifact):
        meta = artifact.meta
        runtime = artifact.meta.prop(MetaNames.RUNTIME_UID)

        runtimes = artifact.meta.prop(MetaNames.RUNTIMES)

        framework_runtimes = artifact.meta.prop(MetaNames.FRAMEWORK_RUNTIMES)

        hyperparameters = artifact.meta.prop(MetaNames.HYPER_PARAMETERS)
        output_data_schema = artifact.meta.prop(MetaNames.OUTPUT_DATA_SCHEMA)
        runtime_input = artifact.meta.prop(MetaNames.RUNTIMES.URL)
        type_model = artifact.meta.prop(MetaNames.TYPE)
        space = artifact.meta.prop(MetaNames.SPACE_UID)
        project= None
        if artifact.meta.prop(MetaNames.PROJECT_UID) is not None:
            projectobj=artifact.meta.prop(MetaNames.PROJECT_UID)
            project = SpaceModels(projectobj.get('href'))
        pipeline = artifact.meta.prop(MetaNames.PIPELINE_UID)
        training_lib = artifact.meta.prop(MetaNames.TRAINING_LIB_UID)
        model_definition = artifact.meta.prop(MetaNames.MODEL_DEFINITON)

        model_input_evaluation =  artifact.meta.prop(MetaNames.EVALUATION_METHOD)
        import_location =  artifact.meta.prop(MetaNames.IMPORT)
        custom = artifact.meta.prop(MetaNames.CUSTOM)
        metrics = artifact.meta.prop(MetaNames.METRICS)
        fairness_info = artifact.meta.prop(MetaNames.FAIRNESS_INFO)
        size = artifact.meta.prop(MetaNames.SIZE)
        software_spec = artifact.meta.prop(MetaNames.SOFTWARE_SPEC)

        description = artifact.meta.prop(MetaNames.DESCRIPTION)
        transformed_label = artifact.meta.prop(MetaNames.TRANSFORMED_LABEL_FIELD)
        training_data_list = None
        if artifact.meta.prop(MetaNames.TRAINING_DATA_REFERENCES) is not None:
            dataref_list = artifact.meta.prop(MetaNames.TRAINING_DATA_REFERENCES)
            training_data_list = []
            if isinstance(dataref_list, list):
                for iter1 in dataref_list:
                    training_ref_obj = iter1
                    type_ = training_ref_obj.get('type', None)
                    connection = training_ref_obj.get('connection', None)
                    location = training_ref_obj.get('location', None)
                    schema = training_ref_obj.get('schema', None)
                    if schema is None and artifact.meta.prop(MetaNames.TRAINING_DATA_SCHEMA) is not None:
                        training_schema = artifact.meta.prop(MetaNames.TRAINING_DATA_SCHEMA)
                        training_schema["id"] = "1"
                        schema = training_schema

                    training_ref = {
                                    "type": type_,
                                    "connection": connection,
                                    "location": location,
                                    "schema": schema
                                    }

                    if training_ref_obj.get('id'):
                        training_ref['id'] = training_ref_obj.get('id')

                    training_data_list.append(training_ref)
        else:
            if artifact.meta.prop(MetaNames.TRAINING_DATA_SCHEMA):
                if artifact.meta.prop(MetaNames.INPUT_DATA_SCHEMA) is None:
                    input_schema_obj = artifact.meta.prop(MetaNames.TRAINING_DATA_SCHEMA)
                    if artifact.meta.prop(MetaNames.LABEL_FIELD):
                        input_schema_obj['id'] = str(MetaNames.INPUT_DATA_SCHEMA)
                        input_schema_obj['fields'] = [f for f in input_schema_obj['fields'] if
                                                      f['name'] != artifact.meta.prop(MetaNames.LABEL_FIELD)]

                    artifact.meta.add(MetaNames.INPUT_DATA_SCHEMA, input_schema_obj)

                # note: issue #29958 set training schema in training_data_reference.schema to keep backward compatibility
                training_data_list = []
                training_data_ref_obj = artifact.meta.prop(MetaNames.TRAINING_DATA_SCHEMA)

                training_data_ref_obj["id"] = "1"
                import os
                if "DEPLOYMENT_PLATFORM" in os.environ and os.environ["DEPLOYMENT_PLATFORM"] == "private":
                    type_ = "fs"
                else:
                    type_ = "container"

                training_data_ref = {
                    "id": "1",
                    "type": type_,
                    "location": {},
                    "schema": training_data_ref_obj
                }

                training_data_list.append(training_data_ref)
                # end note

        if artifact.meta.prop(MetaNames.SPACE_UID) is not None:
            spaceobj=artifact.meta.prop(MetaNames.SPACE_UID)
            space = SpaceModels(
                spaceobj.get('id', None)
            )
        if artifact.meta.prop(MetaNames.SIZE) is not None:
            sizeobj=artifact.meta.prop(MetaNames.SIZE)
            size = ModelsSize(
                sizeobj.get('in_memory', None),
                sizeobj.get('content', None)
                )

        if artifact.meta.prop(MetaNames.PIPELINE_UID) is not None:
            pipelineobj=artifact.meta.prop(MetaNames.PIPELINE_UID)
            pipeline = PipelinesModels(href=pipelineobj.get('href'), id=pipelineobj.get('id'))
        if artifact.meta.prop(MetaNames.MODEL_DEFINITON) is not None:
            model_definition_obj = artifact.meta.prop(MetaNames.MODEL_DEFINITON)
            model_definition = ModelDefinitionModels(
                model_definition_obj.get('id', None)
            )
        if artifact.meta.prop(MetaNames.TRAINING_LIB_UID) is not None:
             training_libobj=artifact.meta.prop(MetaNames.TRAINING_LIB_UID)
             training_lib = TrainingModels(
                 training_libobj.get('href',None)
                 )
        if artifact.meta.prop(MetaNames.IMPORT) is not None:
            importobj=artifact.meta.prop(MetaNames.IMPORT)
            import_location = ModelContentLocation(
                importobj.get('name', None),
                importobj.get('type', None),
                importobj.get('connection', None),
                importobj.get('location', None)
            )
        schemas=None
        input_schema = []
        output_schema = []
        if artifact.meta.prop(MetaNames.INPUT_DATA_SCHEMA) is not None:
            if isinstance(artifact.meta.prop(MetaNames.INPUT_DATA_SCHEMA), dict):
                input_schema = [artifact.meta.prop(MetaNames.INPUT_DATA_SCHEMA)]
            else:
                input_schema = artifact.meta.prop(MetaNames.INPUT_DATA_SCHEMA)

        if artifact.meta.prop(MetaNames.OUTPUT_DATA_SCHEMA) is not None:
            output_schema = [artifact.meta.prop(MetaNames.OUTPUT_DATA_SCHEMA)]
        if len(input_schema) != 0 or len(output_schema) != 0:
            schemas = ModelSchemas(input_schema,output_schema)
        elif artifact.meta.prop('schemas') is not None:
            schemas = artifact.meta.prop('schemas')

        tags_data_list = artifact.meta.prop(MetaNames.TAGS)
        if isinstance(artifact.meta.prop(MetaNames.TAGS), str):
            tags_list = json.loads(artifact.meta.prop(MetaNames.TAGS))
            tags_data_list = []
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

        if artifact.meta.prop(MetaNames.RUNTIME_UID) is not None:
            runtimeobj=artifact.meta.prop(MetaNames.RUNTIME_UID)
            runtime = RuntimeModels(
                runtimeobj.get('href',None)
                )


        if runtimes is not None:
            if isinstance(artifact.meta.prop(MetaNames.RUNTIMES), str):
                runtimes = json.loads(artifact.meta.prop(MetaNames.RUNTIMES))
            else:
                runtimes = artifact.meta.prop(MetaNames.RUNTIMES)
            if not issubclass(type(runtimes), list):
                raise ValueError('Invalid data format for runtimes.')

        if framework_runtimes is not None and software_spec is None:
            if isinstance(artifact.meta.prop(MetaNames.FRAMEWORK_RUNTIMES), str):
                framework_runtimes = json.loads(artifact.meta.prop(MetaNames.FRAMEWORK_RUNTIMES))
            else:
                framework_runtimes = artifact.meta.prop(MetaNames.FRAMEWORK_RUNTIMES)
            if not issubclass(type(framework_runtimes), list):
                raise ValueError('Invalid data format for framework_runtimes.')

        hyper_param_list = None
        if isinstance(hyperparameters, str):
            hyper_param_list = []
            hyperparameters_list = json.loads(artifact.meta.prop(MetaNames.HYPER_PARAMETERS))
            if isinstance(hyperparameters_list, list):
              for iter1 in hyperparameters_list:
                hyper_param = HyperParameters()
                for key in iter1:
                    if key == 'name':
                        hyper_param.name = iter1['name']
                    if key == 'string_value':
                        hyper_param.string_value = iter1['string_value']
                    if key == 'double_value':
                       hyper_param.double_value = iter1['double_value']
                    if key == 'int_value':
                       hyper_param.int_value = iter1['int_value']
                hyper_param_list.append(hyper_param)

        if artifact.meta.prop(MetaNames.SOFTWARE_SPEC) is not None:
            specobj = artifact.meta.prop(MetaNames.SOFTWARE_SPEC)

            software_spec = SoftwareSpecModels(
                specobj.get('id',None)
                )
            #json.load(artifact.meta.prop(MetaNames.SOFTWARE_SPEC))

        label_column=artifact.meta.prop(MetaNames.LABEL_FIELD)
        domain = artifact.meta.prop(MetaNames.DOMAIN)
        project_id = artifact.meta.prop(MetaNames.PROJECT_ID)
        space_id = artifact.meta.prop(MetaNames.SPACE_ID)

        if artifact.meta.prop(MetaNames.EVALUATION_METHOD) is not None:
            metrics = Json2ObjectMapper.read(artifact.meta.prop(MetaNames.EVALUATION_METRICS))
            metrics = list(map(
                lambda metrics_set: EvaluationDefinitionRepositoryMetrics(metrics_set["name"], metrics_set["threshold"],
                                                                          metrics_set["value"]),
                metrics
            ))

        if artifact.meta.prop(MetaNames.FAIRNESS_INFO) and artifact.meta.prop(MetaNames.METRICS):
            raise ValueError("FAIRNESS_INFO and METRICS cannot be set at the same time.")
        elif artifact.meta.prop(MetaNames.FAIRNESS_INFO):
            metrics = [{"context": fairness_info}]

        return MlAssetsCreateModelInput(
            tags=tags_data_list,
            space=space,
            project=project,
            pipeline=pipeline,
            type=type_model,
            training_lib=training_lib,
            name=artifact.name,
            description=meta.prop(MetaNames.DESCRIPTION),
            transformed_label_column=transformed_label,
            training_data_references=training_data_list,
            label_column=label_column,
            import_location=import_location,
            hyper_parameters=hyper_param_list,
            schemas=schemas,
            domain=domain,
            custom=custom,
            runtime=runtime,
            metrics=metrics,
            size=size,
            software_spec=software_spec,
            model_definition=model_definition,
            project_id=project_id,
            space_id=space_id
        )

    @staticmethod
    def _get_version_input(artifact):
        meta = artifact.meta
        if artifact.meta.prop(MetaNames.TYPE).startswith("scikit-model-") \
                or artifact.meta.prop(MetaNames.TYPE).startswith("xgboost"):
            return ModelVersionInput()
        else:
            training_data_ref = Json2ObjectMapper.read(meta.prop(MetaNames.TRAINING_DATA_REFERENCE))
            #if not training_data_ref: #check if is empty dict
            #    training_data_ref = None

            metrics = Json2ObjectMapper.read(meta.prop(MetaNames.EVALUATION_METRICS))
            metrics = list(map(
                lambda metrics_set: EvaluationDefinitionMetrics(metrics_set["name"], metrics_set["threshold"], metrics_set["value"]),
                metrics
            ))

            return ModelVersionInput(training_data_ref, EvaluationDefinition(
                meta.prop(MetaNames.EVALUATION_METHOD),
                metrics
            ))

    def _wsd_create_model_asset(self, url, input_payload, artifact, params, headers):
        import json
        import ibm_watson_machine_learning._wrappers.requests as requests
        import urllib
        params = params
        headers = headers

        try:
            ## create model asset type
            atype_body = {
                "name": "wml_model"
            }
            aheaders = {
                'Content-Type': "application/json"
            }
            atype_payload = json.dumps(atype_body, separators=(',', ':'))
            asset_type_response = requests.post(
                self.client.service_instance._href_definitions.get_wsd_asset_type_href(),
                params=params,
                data=atype_payload,
                headers=aheaders
            )
            if asset_type_response.status_code != 200 and \
                    asset_type_response.status_code != 201 and asset_type_response.status_code != 409:
                raise Exception("Failed to create asset type. Try again.")

            create_response = requests.post(
                self.client.service_instance._href_definitions.get_assets_href(),
                params=params,
                json=input_payload,
                headers=headers
            )
            if create_response.status_code == 201:
                try:
                    asset_details = create_response.json()
                except Exception as e:
                    raise Exception(u'Failure during parsing json response: \'{}\''.format(create_response.text), e)
            else:
                raise Exception(u'Failure during {}.'.format('Model creation'), create_response)

            content_stream = artifact.reader().read()

            # Upload model content to desktop project using polyfill
            if create_response.status_code == 201:
                asset_uid = create_response.json()["metadata"]["asset_id"]
                file_name_to_attach = 'wml_model_attachment'
                content_upload_url = self.client.service_instance._href_definitions.get_wsd_model_attachment_href() + '/' + \
                                     urllib.parse.quote("wml_model/" + asset_uid + "/" + file_name_to_attach, safe='')
                attach_url = self.client.service_instance._href_definitions.get_attachments_href(asset_uid)
                fdata = content_stream
                response = requests.put(
                    content_upload_url,
                    files={'file': ('native', fdata, 'application/gzip', {'Expires': '0'})},
                    params=params
                )
                if response.status_code == 201:
                    # update the attachement url with details :
                    asset_body = {
                        "asset_type": "wml_model",
                        "name": "native",
                        "object_key": asset_uid + "/" + file_name_to_attach,
                        "object_key_is_read_only": True
                    }
                    attach_payload = json.dumps(asset_body, separators=(',', ':'))

                    attach_response = requests.post(attach_url,
                                                    data=attach_payload,
                                                    params=params,
                                                    headers=headers)
                    if attach_response.status_code == 201:
                        artifact.reader().close()
                        return asset_details
                    else:
                        raise Exception('Failed to create model.')
                else:
                    raise Exception("Failed while creating a model. Try again.")
            else:
                raise Exception("Failed while creating a model. Try again.")

        except Exception as e:
            raise e

    def wsd_save(self, url, artifact, meta_props, payload_input, query_param=None, headers=None):
        try:
            cams_entity = copy.deepcopy(payload_input)
            #cams_entity.pop('name')
            if cams_entity.get('description') is not None:
                cams_entity.pop('description')
            origin_country = "US"

            payload_metadata = {
                "name": meta_props['name'],
                "asset_type": 'wml_model',
                "origin_country": origin_country,
                "assetCategory": "USERS"
            }
            if 'description' in meta_props and meta_props['description'] is not None:
                payload_metadata.update({'description': meta_props['description']})

            cams_entity.update({'content_status': {
                "state": "persisted"}})
            if 'trainingDataSchema' in meta_props and meta_props['trainingDataSchema'] is not None:
                training_schema_field = meta_props['trainingDataSchema']['fields']
                training_data_reference=[
                    {'location': {'bucket': 'not_applicable'},
                     'type': 'fs',
                     'connection': {'access_key_id': 'not_applicable',
                                    'secret_access_key': 'not_applicable',
                                    'endpoint_url': 'not_applicable'},
                     'schema': {
                         'id': '1',
                         'type': 'struct',
                         'fields': training_schema_field
                     }}]
                cams_entity.pop('trainingDataSchema')
                cams_entity.update({'training_data_references': training_data_reference})
            cams_payload = {
                "metadata": payload_metadata,
                "entity": {
                    "wml_model": cams_entity
                }
            }
            model_details = self._wsd_create_model_asset(url, cams_payload, artifact, query_param, headers)

            return model_details
        except Exception as e:
            logger.info('Error in pipeline model creation')
            import traceback
            print(traceback.format_exc())
            raise e

    def _create_pipeline_model_v4_cloud(self, model_artifact, query_param=None):
        model_input = self._prepare_model_input(model_artifact)
        headers = self.client.api_client.default_headers

        model_output = self.repository_api.ml_assets_model_creation_v4_cloud(model_input, query_param,headers)
        if model_output is not None:
            location = model_output.metadata.id
            status_url = None
            #martifact = model_artifact._copy(uid=model_id)
            new_artifact = ModelAdapter(model_output, location, self.client).artifact()

            new_artifact.load_model = lambda: model_artifact.ml_pipeline_model
            new_artifact.model_instance = lambda: model_artifact.ml_pipeline_model
            query_param.update({'content_format': 'native'})
            model_artifact_with_version = model_artifact._copy(meta_props=new_artifact.meta,uid=location)
            status_url = self._upload_pipeline_model_content(model_artifact_with_version, query_param)
            if status_url is not None and status_url != "":
                place_holder =new_artifact.meta.add(MetaNames.STATUS_URL, status_url)
                new_async_artifact = new_artifact._copy(meta_props=place_holder)
                return new_async_artifact
            else:
                return new_artifact
        else:
            logger.info('Location of the new pipeline model not found')
            raise ApiException(404, 'No artifact location')

    def _get_v4_cloud_model(self, artifact_id, query_param=None):
        logger.debug('Fetching information about pipeline model: {}'.format(artifact_id))
        if not isinstance(artifact_id, str) and not isinstance(artifact_id, unicode):
            raise ValueError('Invalid type for artifact_id: {}'.format(artifact_id.__class__.__name__))
        model_output = self.repository_api.v4_ml_assets_models_artifact_id_get(artifact_id, query_param)
        if model_output is not None:
            latest_version = model_output.metadata.id
            return ModelAdapter(model_output, latest_version, self.client).artifact()
        else:
            logger.debug('Model with guid={} not found'.format(artifact_id))
            raise ApiException('Model with guid={} not found'.format(artifact_id))
