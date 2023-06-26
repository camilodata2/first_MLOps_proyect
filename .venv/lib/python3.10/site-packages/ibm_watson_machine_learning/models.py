#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function

from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact import MLRepositoryArtifact
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.utils.autoai.utils import load_file_from_file_system_nonautoai, init_cos_client, \
    prepare_auto_ai_model_to_publish, get_autoai_run_id_from_experiment_metadata, check_if_ts_pipeline_is_winner, \
 prepare_auto_ai_model_to_publish_normal_scenario
from ibm_watson_machine_learning.utils import MODEL_DETAILS_TYPE, INSTANCE_DETAILS_TYPE, load_model_from_directory, \
    is_lale_pipeline
from ibm_watson_machine_learning.metanames import ModelMetaNames,LibraryMetaNames
import os
import copy
import urllib.parse
import time
import copy

from ibm_watson_machine_learning.utils.deployment.errors import ModelPromotionFailed, PromotionFailed
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure, UnexpectedType
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.libs.repo.util.compression_util import CompressionUtil
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES,PIPELINES, LIBRARIES, EXPERIMENTS, RUNTIMES, DEPLOYMENTS, SOFTWARE_SPEC


import shutil
import re

_DEFAULT_LIST_LENGTH = 50


class Models(WMLResource):
    """Store and manage models."""

    ConfigurationMetaNames = ModelMetaNames()
    """MetaNames for models creation."""

    LibraryMetaNames = LibraryMetaNames()
    """MetaNames for libraries creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not self._client.WSD and not self._client.CLOUD_PLATFORM_SPACES:
            Models._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Models._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self._ICP = client.ICP
        self._CAMS = client.CAMS
        if self._CAMS:
            self.default_space_id = client.default_space_id

    def _save_library_archive(self, ml_pipeline):

        id_length = 20
        gen_id = uid_generate(id_length)
        temp_dir_name = '{}'.format("library" + gen_id)
        # if (self.hummingbird_env == 'HUMMINGBIRD') is True:
        #     temp_dir = os.path.join('/home/spark/shared/wml/repo/extract_', temp_dir_name)
        # else:
        temp_dir = os.path.join('.', temp_dir_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        ml_pipeline.write().overwrite().save(temp_dir)
        archive_path = self._compress_artifact(temp_dir)
        shutil.rmtree(temp_dir)
        return archive_path

    def _compress_artifact(self, compress_artifact):
        tar_filename = '{}_content.tar'.format('library')
        gz_filename = '{}.gz'.format(tar_filename)
        CompressionUtil.create_tar(compress_artifact, '.', tar_filename)
        CompressionUtil.compress_file_gzip(tar_filename, gz_filename)
        os.remove(tar_filename)
        return gz_filename

    def _create_pipeline_input(self,lib_href,name,space_uid=None, project_uid=None):

        if self._client.CAMS or self._client.WSD:
            metadata = {
                self._client.pipelines.ConfigurationMetaNames.NAME: name + "_"+uid_generate(8),
                self._client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                    "doc_type": "pipeline",
                    "version": "2.0",
                    "primary_pipeline": "dlaas_only",
                    "pipelines": [
                        {
                            "id": "dlaas_only",
                            "runtime_ref": "spark",
                            "nodes": [
                                {
                                    "id": "repository",
                                    "type": "model_node",
                                    "inputs": [
                                    ],
                                    "outputs": [],
                                    "parameters": {
                                        "training_lib_href": lib_href
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        else:  # for cloud do not change anything
            metadata = {
                self._client.pipelines.ConfigurationMetaNames.NAME: name + "_" + uid_generate(8),
                self._client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                    "doc_type": "pipeline",
                    "version": "2.0",
                    "primary_pipeline": "dlaas_only",
                    "pipelines": [
                        {
                            "id": "dlaas_only",
                            "runtime_ref": "spark",
                            "nodes": [
                                {
                                    "id": "repository",
                                    "type": "model_node",
                                    "inputs": [
                                    ],
                                    "outputs": [],
                                    "parameters": {
                                        "training_lib_href": lib_href + "/content"
                                    }
                                }
                            ]
                        }
                    ]
                }
            }

        if space_uid is not None:
            metadata.update({self._client.pipelines.ConfigurationMetaNames.SPACE_UID: space_uid})

        if self._client.default_project_id is not None:
            metadata.update({'project': {"href": "/v2/projects/" + self._client.default_project_id}})
        return metadata

    def _tf2x_load_model_instance(self, model_id):
        artifact_url = self._client.service_instance._href_definitions.get_model_last_version_href(model_id)
        params = self._client._params()
        id_length = 20
        gen_id = uid_generate(id_length)

        # Step1 :  Download the model content

        url = self._client.service_instance._href_definitions.get_published_model_href(model_id)
        params.update({'content_format': 'native'})
        artifact_content_url = str(artifact_url + u'/download')
        r = requests.get(artifact_content_url, params=params,
                         headers=self._client._get_headers(), stream=True)

        if r.status_code != 200:
            raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), r)

        downloaded_model = r.content
        self._logger.info(u'Successfully downloaded artifact with artifact_url: {}'.format(artifact_url))

        # Step 2 :  copy the downloaded tar.gz in to a temp folder
        try:
            temp_dir_name = '{}'.format('hdfs' + gen_id)
            # temp_dir = os.path.join('.', temp_dir_name)
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            gz_filename = temp_dir + "/download.tar.gz"
            tar_filename = temp_dir + "/download.tar"
            with open(gz_filename, 'wb') as f:
                f.write(downloaded_model)

        except IOError as e:
            raise WMLClientError(u'Saving model with artifact_url: \'{}\' failed.'.format(model_id), e)

        # create model instance based on the type using load_model
        try:
            CompressionUtil.decompress_file_gzip(gzip_filepath=gz_filename,filepath=tar_filename)
            CompressionUtil.extract_tar(tar_filename, temp_dir)
            os.remove(tar_filename)
            import tensorflow as tf
            import glob
            h5format = True
            if not glob.glob(temp_dir + '/sequential_model.h5'):
                h5format = False
            if h5format is True:
                model_instance = tf.keras.models.load_model(temp_dir + '/sequential_model.h5', custom_objects=None,
                                                            compile=True)
                return model_instance
            elif glob.glob(temp_dir + '/saved_model.pb'):
                model_instance = tf.keras.models.load_model(temp_dir, custom_objects=None, compile=True)
                return model_instance
            else:
                raise WMLClientError(u'Load model with artifact_url: \'{}\' failed.'.format(model_id))

        except IOError as e:
            raise WMLClientError(u'Saving model with artifact_url: \'{}\' failed.'.format(model_id), e)

    def _publish_from_object(self, model, meta_props, training_data=None, training_target=None, pipeline=None, feature_names=None, label_column_names=None, project_id=None):
        """Store model from object in memory into Watson Machine Learning repository on Cloud."""
        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, str, True)
        if self._client.ICP_47 is None and self._client.ICP_46 is None and self._client.ICP_45 is None and self._client.ICP_40 is None and self._client.ICP_35 is None and self._client.WSD_20 is None:
            if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props and \
               self.ConfigurationMetaNames.RUNTIME_UID not in meta_props:
                raise WMLClientError("Invalid input. It is mandatory to provide RUNTIME_UID in meta_props.")
        else:
            if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props and \
               self.ConfigurationMetaNames.RUNTIME_UID not in meta_props:
                  raise WMLClientError("Invalid input. It is mandatory to provide RUNTIME_UID or "
                                       "SOFTWARE_SPEC_UID in meta_props. RUNTIME_UID is deprecated")

        # self._validate_type(project_id, 'project_id',str, True)
        try:
            if 'pyspark.ml.pipeline.PipelineModel' in str(type(model)):
                if(pipeline is None or training_data is None):
                    raise WMLClientError(u'pipeline and training_data are expected for spark models.')
                if not self._client.CAMS and not self._client.WSD:
                    name = meta_props[self.ConfigurationMetaNames.NAME]
                    version = "1.0"
                    platform = {"name": "python", "versions": ["3.5"]}
                    library_tar = self._save_library_archive(pipeline)
                    lib_metadata = {
                        self.LibraryMetaNames.NAME: name + "_" + uid_generate(8),
                        self.LibraryMetaNames.VERSION: version,
                        self.LibraryMetaNames.PLATFORM: platform,
                        self.LibraryMetaNames.FILEPATH: library_tar
                    }

                    if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                        lib_metadata.update(
                            {self.LibraryMetaNames.SPACE_UID: meta_props[
                                self._client.repository.ModelMetaNames.SPACE_UID]})

                    if self._client.CAMS:
                        if self._client.default_space_id is not None:
                            lib_metadata.update(
                                {self.LibraryMetaNames.SPACE_UID: self._client.default_space_id})
                        else:
                            if self._client.default_project_id is None:
                                raise WMLClientError("It is mandatory is set the space or Project. \
                                                                       Use client.set.default_space(<SPACE_GUID>) to set the space or Use client.set.default_project(<PROJECT_ID)")

                    library_artifact = self._client.runtimes.store_library(meta_props=lib_metadata)
                    lib_href = self._client.runtimes.get_library_href(library_artifact)
                    space_uid = None

                    if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                        space_uid = meta_props[self._client.repository.ModelMetaNames.SPACE_UID]
                    if self._client.CAMS:
                        space_uid = self._client.default_space_id
                    pipeline_metadata = self._create_pipeline_input(lib_href, name, space_uid=space_uid)
                    pipeline_save = self._client.pipelines.store(pipeline_metadata)

                    pipeline_href = self._client.pipelines.get_href(pipeline_save)

                if self._client.CAMS or self._client.WSD:
                    name = meta_props[self.ConfigurationMetaNames.NAME]
                    version = "1.0"
                    platform = {"name": "python", "versions": ["3.6"]}
                    library_tar = self._save_library_archive(pipeline)
                    model_definition_props = {
                        self._client.model_definitions.ConfigurationMetaNames.NAME: name + "_" + uid_generate(8),
                        self._client.model_definitions.ConfigurationMetaNames.VERSION: version,
                        self._client.model_definitions.ConfigurationMetaNames.PLATFORM: platform,
                    }
                    training_lib = self._client.model_definitions.store(library_tar, model_definition_props)
                    if self._client.WSD:
                        lib_href = self._client.service_instance._href_definitions.get_base_asset_href(training_lib['metadata']['asset_id'])
                        pipeline_metadata = self._create_pipeline_input(lib_href, name, space_uid=None)
                        pipeline_save = self._client.pipelines.store(pipeline_metadata)
                        pipeline_href = self._client.service_instance._href_definitions.get_base_asset_href(pipeline_save['metadata']['asset_id'])
                        meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = {
                            "href": pipeline_href}
                    else:
                        lib_href = self._client.model_definitions.get_href(training_lib)
                        lib_href = lib_href.split("?", 1)[0] # temp fix for stripping space_id
                    # space_uid = None

                    # if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                    #     space_uid = meta_props[self._client.repository.ModelMetaNames.SPACE_UID]
                    # if self._client.CAMS:
                    #     space_uid = self._client.default_space_id
                    pipeline_metadata = self._create_pipeline_input(lib_href, name, space_uid=None)
                    pipeline_save = self._client.pipelines.store(pipeline_metadata)

                    pipeline_href = self._client.pipelines.get_href(pipeline_save)

                if not self._client.WSD:
                    meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = {"href": pipeline_href}

                if self.ConfigurationMetaNames.SPACE_UID in meta_props and meta_props[self._client.repository.ModelMetaNames.SPACE_UID] is not None:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SPACE_UID, str, False)
                    meta_props[self._client.repository.ModelMetaNames.SPACE_UID] = {"href": API_VERSION + SPACES + "/" + meta_props[self._client.repository.ModelMetaNames.SPACE_UID]}
                else:
                    if self._client.default_project_id is not None:
                        meta_props['project'] = {"href": "/v2/projects/" + self._client.default_project_id}

                if self.ConfigurationMetaNames.RUNTIME_UID in meta_props:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.RUNTIME_UID, str, False)
                    meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID] = {"href": API_VERSION + RUNTIMES + "/" + meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID]}

                if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props:
                    if self._client.WSD_20:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID, str,
                                                 False)
                        meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID] = {
                            "base_id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}
                    else:
                        meta_props.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_UID)

                if self._client.WSD_20:
                    if self.ConfigurationMetaNames.MODEL_DEFINITION_UID in meta_props:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_UID, str, False)
                        meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID] = {"id":meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID]}
                else:
                    if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.TRAINING_LIB_UID, str, False)
                        meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_UID] = {"href": API_VERSION + LIBRARIES + "/" + meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_UID]}

                meta_data = MetaProps(self._client.repository._meta_props_to_repository_v3_style(meta_props))

                model_artifact = MLRepositoryArtifact(model, name=str(meta_props[self.ConfigurationMetaNames.NAME]), meta_props=meta_data, training_data=training_data, feature_names=feature_names, label_column_names=label_column_names)
            else:
                if self.ConfigurationMetaNames.SPACE_UID in meta_props and meta_props[self._client.repository.ModelMetaNames.SPACE_UID] is not None:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SPACE_UID, str, False)
                    meta_props[self._client.repository.ModelMetaNames.SPACE_UID] = {"href": API_VERSION + SPACES + "/" + meta_props[self._client.repository.ModelMetaNames.SPACE_UID]}
                if self._client.CAMS:
                    if self._client.default_space_id is not None:
                        meta_props[self._client.repository.ModelMetaNames.SPACE_UID] = {
                            "href": API_VERSION + SPACES + "/" + self._client.default_space_id}
                    else:
                        if self._client.default_project_id is not None:
                            meta_props['project'] = {"href": "/v2/projects/" + self._client.default_project_id}
                        else:
                            raise WMLClientError("It is mandatory is set the space or Project. \
                             Use client.set.default_space(<SPACE_GUID>) to set the space or Use client.set.default_project(<PROJECT_ID)")

                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                        self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                    if self._client.default_space_id is not None:
                        meta_props['space_id'] = self._client.default_space_id
                    else:
                        if self._client.default_project_id is not None:
                            meta_props['project_id'] = self._client.default_project_id
                        else:
                            raise WMLClientError("It is mandatory is set the space or Project. \
                             Use client.set.default_space(<SPACE_GUID>) to set the space or"
                                                 " Use client.set.default_project(<PROJECT_ID)")

                if self.ConfigurationMetaNames.RUNTIME_UID in meta_props:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.RUNTIME_UID, str, False)
                    meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID] = {"href": API_VERSION + RUNTIMES + "/" + meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID]}
                if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props:
                    if self._client.ICP_35 or self._client.ICP_40 or self._client.ICP_45 or \
                            self._client.ICP_46 or self._client.ICP_47:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID, str, False)
                        meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID] = \
                            {"id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}
                    elif self._client.WSD_20:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID, str,
                                                 False)
                        meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID] = \
                            {"base_id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}
                    else:
                        meta_props.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_UID)

                if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.PIPELINE_UID, str, False)
                    if self._client.WSD:
                        meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = {"href": self._client.service_instance._href_definitions.get_base_asset_href(meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID])}
                    else:
                        meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = {"href": API_VERSION + PIPELINES + "/" + meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID]}

                if self._client.WSD_20:
                    if self.ConfigurationMetaNames.MODEL_DEFINITION_UID in meta_props:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_UID, str,
                                                 False)
                        meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID] = {
                            "id": meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID]}
                else:
                    if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.TRAINING_LIB_UID, str, False)
                        meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_UID] = {"href": API_VERSION + LIBRARIES + "/" + meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_UID]}


                meta_data = MetaProps(self._client.repository._meta_props_to_repository_v3_style(meta_props))
                model_artifact = MLRepositoryArtifact(model, name=str(meta_props[self.ConfigurationMetaNames.NAME]), meta_props=meta_data, training_data=training_data, training_target=training_target, feature_names=feature_names, label_column_names=label_column_names)
            if self._client.CAMS:
                query_param_for_repo_client = self._client._params()

            else:
                if self._client.WSD:
                    if self._client.default_project_id is not None:
                        query_param_for_repo_client = {'project_id': self._client.default_project_id}
                        meta_props['project'] = {"href": "/v2/projects/" + self._client.default_project_id}
                    else:
                        raise WMLClientError(u'Project id is not set.')
                else:
                    query_param_for_repo_client = None
            if self._client.WSD:
                #payload = self._create_model_payload(meta_props, feature_names=None, label_column_names=None)
                wsd_url = self._client.service_instance._href_definitions.get_wsd_base_href()
                saved_model = self._client.repository._ml_repository_client.models.wsd_save(wsd_url, model_artifact, meta_props, meta_props,
                                                                                             query_param=query_param_for_repo_client,
                                                                                             headers= self._client._get_headers())
                return self.get_details(u'{}'.format(saved_model['asset_id']))
            else:
                saved_model = self._client.repository._ml_repository_client.models.save(model_artifact, query_param=query_param_for_repo_client)
        except Exception as e:
            raise WMLClientError(u'Publishing model failed.', e)
        else:
            return self.get_details(u'{}'.format(saved_model.uid))

    def _get_subtraining_object(self, trainingobject, subtrainingId):
        subtrainings = trainingobject["resources"]
        for each_training in subtrainings:
            if each_training["metadata"]["guid"] == subtrainingId:
                return each_training
        raise WMLClientError("The subtrainingId " + subtrainingId + " is not found")

    ##TODO not yet supported

    def _publish_from_training(self, model_uid, meta_props, subtrainingId=None, feature_names=None, label_column_names=None, round_number=None):
        """Store trained model from object storage into Watson Machine Learning repository on IBM Cloud."""
        model_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            client=self._client
        )
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            if self.ConfigurationMetaNames.RUNTIME_UID in meta_props:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.RUNTIME_UID, str, False)
                model_meta.update({self.ConfigurationMetaNames.RUNTIME_UID: {
                    "href": API_VERSION + RUNTIMES + "/" + meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID]}})
            if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props:
                if self._client.WSD_20:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID, str, False)
                    model_meta.update({self.ConfigurationMetaNames.SOFTWARE_SPEC_UID: {
                        "base_id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}})
                else:
                    model_meta.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_UID)

            if self.ConfigurationMetaNames.SPACE_UID in meta_props and meta_props[self._client.repository.ModelMetaNames.SPACE_UID] is not None:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SPACE_UID, str, False)
                model_meta.update({self.ConfigurationMetaNames.SPACE_UID: {
                    "href": API_VERSION + SPACES + "/" + meta_props[self._client.repository.ModelMetaNames.SPACE_UID]}})
            if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.PIPELINE_UID, str, False)
                if self._client.WSD:
                    model_meta.update({self.ConfigurationMetaNames.PIPELINE_UID: {
                        "href": self._client.service_instance._href_definitions.get_base_asset_href(meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID])}})
                else:
                    model_meta.update({self.ConfigurationMetaNames.PIPELINE_UID: {
                        "href": API_VERSION + PIPELINES + "/" + meta_props[
                            self._client.repository.ModelMetaNames.PIPELINE_UID]}})

            if self._client.WSD_20:
                if self.ConfigurationMetaNames.MODEL_DEFINITION_UID in meta_props:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_UID, str, False)
                    meta_props[self.ConfigurationMetaNames.MODEL_DEFINITION_UID] = {
                        "id": meta_props[self.ConfigurationMetaNames.MODEL_DEFINITION_UID]}
            else:
                if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.TRAINING_LIB_UID, str, False)
                    model_meta.update({self.ConfigurationMetaNames.TRAINING_LIB_UID: {
                        "href": API_VERSION + LIBRARIES + "/" + meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_UID]}})
            if self._client.default_project_id is not None:
                model_meta.update({'project': {"href": "/v2/projects/" + self._client.default_project_id}})

            input_schema = []
            output_schema = []
            if self.ConfigurationMetaNames.INPUT_DATA_SCHEMA in meta_props and \
               meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] is not None:
                if self._client.WSD_20:
                    if isinstance(meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA], dict):
                        input_schema = [meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]]
                    else:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, list, False)
                        input_schema = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
                else:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, dict, False)
                    input_schema = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
                model_meta.pop(self.ConfigurationMetaNames.INPUT_DATA_SCHEMA)
            if self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA in meta_props:
                if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith('do-') and self._client.WSD_20:
                    try:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict, False)
                        output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]
                    except WMLClientError:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA,list, False)
                        output_schema = meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]
                else:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict, False)
                    output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]
                model_meta.pop(self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA)

            if len(input_schema) != 0 or len(output_schema) != 0:
                model_meta.update({"schemas": {
                    "input": input_schema,
                    "output": output_schema}
                })

            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, str, True)
        else:
            model_meta =  self._create_cloud_model_payload(meta_props, feature_names=feature_names, label_column_names=label_column_names)

        try:
            details = self._client.training.get_details(model_uid, _internal=True)

        except ApiRequestFailure as e:
            raise UnexpectedType('model parameter', 'model path / training_id', model_uid)
        model_type = ""

        ##Check if the training is created from pipeline or experiment
        if "pipeline" in details["entity"]:
            if any([self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40,
                    self._client.ICP_45, self._client.ICP_46, self._client.ICP_47]):
                pipeline_id = details['entity']['pipeline']['id']
            else:
                href = details["entity"]["pipeline"]["href"]
                pipeline_id = href.split("/")[3]
            if "model_type" in details["entity"]["pipeline"]:
                model_type = details["entity"]["pipeline"]["model_type"]

        if "experiment" in details["entity"]:

            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                    self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                url = self._wml_credentials['url'] + '/ml/v4/trainings?parent_id='+model_uid
            else:
                url = self._wml_credentials['url'] + '/v4/trainings?parent_id='+model_uid,

            details_parent = requests.get(url,
                params=self._client._params(),
                headers=self._client._get_headers()
            )
            details_json = self._handle_response(200,"Get training details",details_parent)
            subtraining_object = self._get_subtraining_object(details_json,subtrainingId)
            model_meta.update({"import": subtraining_object["entity"]["results_reference"]})
            if "pipeline" in subtraining_object["entity"]:
                if any([self._client.ICP_35, self._client.ICP_40, self._client.ICP_45,
                        self._client.ICP_46, self._client.ICP_47, self._client.CLOUD_PLATFORM_SPACES]):
                    pipeline_id =  subtraining_object["entity"]["pipeline"]["id"]
                else:
                    pipeline_id = subtraining_object["entity"]["pipeline"]["id"].split("/")[3]
                if "model_type" in subtraining_object["entity"]["pipeline"]:
                     model_type = subtraining_object["entity"]["pipeline"]["model_type"]
        else:
            model_meta.update({"import": details["entity"]["results_reference"]})

        if "pipeline" in details["entity"] or "experiment" in details["entity"]:
            if "experiment" in details["entity"]:
                if any([self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40,
                        self._client.ICP_45, self._client.ICP_46, self._client.ICP_47]):
                    url = self._wml_credentials['url'] + '/ml/v4/trainings?parent_id=' + model_uid
                else:
                    url = self._wml_credentials['url'] + '/v4/trainings?parent_id=' + model_uid

                details_parent = requests.get(url,
                                              params=self._client._params(),
                                              headers=self._client._get_headers()
                                              )
                details_json = self._handle_response(200, "Get training details", details_parent)
                subtraining_object = self._get_subtraining_object(details_json, subtrainingId)
                if "pipeline" in subtraining_object["entity"]:
                    definition_details = self._client.pipelines.get_details(pipeline_id)
                    runtime_uid = definition_details["entity"]["document"]["runtimes"][0]["name"] + "_" + \
                                  definition_details["entity"]["document"]["runtimes"][0]["version"].split("-")[0] + "-py3"
                    if model_type == "":
                        model_type = definition_details["entity"]["document"]["runtimes"][0]["name"] + "_" + \
                                     definition_details["entity"]["document"]["runtimes"][0]["version"].split("-")[0]

                    if self.ConfigurationMetaNames.TYPE not in meta_props:
                        model_meta.update({"type": model_type})

                    if self.ConfigurationMetaNames.RUNTIME_UID not in meta_props:
                        model_meta.update({"runtime": {"href": "/v4/runtimes/" + runtime_uid}})
            else:
                definition_details = self._client.pipelines.get_details(pipeline_id)
                runtime_uid = definition_details["entity"]["document"]["runtimes"][0]["name"] + "_" + \
                              definition_details["entity"]["document"]["runtimes"][0].get("version", "0.1").split("-")[0] + "-py3"
                if model_type == "":
                    model_type = definition_details["entity"]["document"]["runtimes"][0]["name"] + "_" + \
                                 definition_details["entity"]["document"]["runtimes"][0].get("version", "0.1").split("-")[0]

                if self.ConfigurationMetaNames.TYPE not in meta_props:
                    model_meta.update({"type": model_type})

                if self.ConfigurationMetaNames.RUNTIME_UID not in meta_props and not self._client.CLOUD_PLATFORM_SPACES \
                        and not self._client.ICP_35 and not self._client.ICP_40 and not self._client.ICP_45 \
                        and not self._client.ICP_46 and not self._client.ICP_47:
                    model_meta.update({"runtime": {"href": "/v4/runtimes/" + runtime_uid}})

        if label_column_names:
            model_meta['label_column'] = label_column_names[0]

        if any([self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40, self._client.ICP_45,
                self._client.ICP_46, self._client.ICP_47]):
            if details.get('entity').get('status').get('state') == 'failed' or details.get('entity').get('status').get(
                    'state') == 'pending':
                raise WMLClientError(
                    'Training is not successfully completed for the given training_id. Please check the status of training run. Training should be completed successfully to store the model')

            model_dir = model_uid
            if "federated_learning" in details["entity"]  and round_number is not None :
                if not details.get('entity').get('federated_learning').get('save_intermediate_models',False) : 
                    raise WMLClientError('The Federated Learning experiment was not configured to save intermediate models')

                rounds = details.get('entity').get('federated_learning').get('rounds')
                if isinstance(round_number, int) and 0 < round_number and round_number <= rounds :
                    if round_number < rounds :
                        # intermediate models
                        model_dir = model_dir + '_' + str(round_number)
                else :
                    raise WMLClientError('Invalid input. round_number should be an int between 1 and {}'.format(rounds))
                
                
            asset_url = details['entity']['results_reference']['location'][
                            'assets_path'] + "/" + model_dir + "/resources/wml_model/request.json"

            if self._client.ICP_PLATFORM_SPACES:
                try:
                    asset_parts = asset_url.split('/')
                    asset_url = '/'.join(asset_parts[asset_parts.index('assets')+1:])
                    request_str = load_file_from_file_system_nonautoai(wml_client=self._client,
                                                                       file_path=asset_url).read().decode()

                    import json
                    if json.loads(request_str).get('code') == 404:
                        raise Exception('Not found file.')
                except:
                    asset_url = "trainings/" + model_uid + "/assets/" + model_dir + "/resources/wml_model/request.json"
                    request_str = load_file_from_file_system_nonautoai(wml_client=self._client, file_path=asset_url).read().decode()
            else:
                if len(details['entity']['results_reference']['connection']) > 1:
                    cos_client = init_cos_client(details['entity']['results_reference']['connection'])
                    bucket = details['entity']['results_reference']['location']['bucket']
                else:
                    results_reference = DataConnection._from_dict(details['entity']['results_reference'])
                    results_reference.set_client(self._client)
                    results_reference._check_if_connection_asset_is_s3()
                    results_reference = results_reference._to_dict()
                    cos_client = init_cos_client(results_reference['connection'])
                    bucket = results_reference['location']['bucket']
                cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=asset_url)
                with open('request.json', 'r') as f:
                    request_str = f.read()
            from typing import Dict
            import json
            request_json: Dict[str, dict] = json.loads(request_str)
            request_json['name'] = meta_props[self.ConfigurationMetaNames.NAME]
            request_json['content_location']['connection'] = details['entity']['results_reference']['connection']
            if 'space_id' in model_meta:
                request_json['space_id'] = model_meta['space_id']

            else:
                request_json['project_id'] = model_meta['project_id']

            if 'label_column' in model_meta:
                request_json['label_column'] = model_meta['label_column']

            if 'pipeline' in request_json:
                request_json.pop('pipeline')  # not needed for other space
            if 'training_data_references' in request_json:
                request_json.pop('training_data_references')
            if 'software_spec' in request_json:
                request_json.pop('software_spec')
                request_json.update({'software_spec': {'id': meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}})

        if not self._ICP:
            if self._client.CLOUD_PLATFORM_SPACES:
                params = {}
                params.update({'version': self._client.version_param})
                creation_response = requests.post(
                       self._wml_credentials['url'] + '/ml/v4/models',
                       headers=self._client._get_headers(),
                       json=request_json,
                       params = params
                )
                model_details = self._handle_response(202, u'creating new model', creation_response)
                model_uid = model_details['metadata']['id']
            else:
                creation_response = requests.post(
                    self._wml_credentials['url'] + '/v4/models',
                    headers=self._client._get_headers(),
                    json=model_meta
                )
                model_details = self._handle_response(201, u'creating new model', creation_response)
                model_uid = model_details['metadata']['id']
        else:
            if self._client.ICP_35 or self._client.ICP_40 or self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                params = {}
                params.update({'version': self._client.version_param})
                creation_response = requests.post(
                    self._wml_credentials['url'] + '/ml/v4/models',
                    headers=self._client._get_headers(),
                    json=request_json,
                    params=params
                )
            else:
                creation_response = requests.post(
                    self._wml_credentials['url'] + '/v4/models',
                    headers=self._client._get_headers(),
                    json=model_meta
                )
            if self._client.ICP_35 or self._client.ICP_40 or self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                model_details = self._handle_response(202, u'creating new model', creation_response)
                model_uid = model_details['metadata']['id']
            else:
                model_details = self._handle_response(202, u'creating new model', creation_response)
                model_uid = model_details['metadata']['guid']
        return self.get_details(model_uid)

    def _store_autoAI_model(self,model_path, meta_props, feature_names=None, label_column_names=None):
        """Store trained model from object storage into Watson Machine Learning repository on IBM Cloud."""
        model_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            client=self._client
        )
        if not any([self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40, self._client.ICP_45,
                    self._client.ICP_46, self._client.ICP_47]):
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, str, True)

            # note: remove pipeline-model.json part from the string to allow correct regexp
            if model_path.endswith('pipeline-model.json'):
                x = re.findall(r"[0-9A-Za-z-]+-[0-9A-Za-z-]+", '/'.join(model_path.split('/')[:-1]))
                # --- end note
            else:
                x = re.findall(r"[0-9A-Za-z-]+-[0-9A-Za-z-]+", model_path)
            model_uid = x[-1] if x else ''
            details = self._client.training.get_details(model_uid, _internal=True)

            if self.ConfigurationMetaNames.RUNTIME_UID in meta_props:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.RUNTIME_UID, str, False)
                model_meta.update({self.ConfigurationMetaNames.RUNTIME_UID: {
                    "href": API_VERSION + RUNTIMES + "/" + meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID]}})
            if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props:
                if self._client.WSD_20:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID, str, False)
                    model_meta.update({self.ConfigurationMetaNames.SOFTWARE_SPEC_UID: {
                        "base_id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}})
                else:
                    model_meta.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_UID)

            if self.ConfigurationMetaNames.SPACE_UID in meta_props and \
                    meta_props[self._client.repository.ModelMetaNames.SPACE_UID] is not None:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SPACE_UID, str, False)
                model_meta.update({self.ConfigurationMetaNames.SPACE_UID: {
                    "href": API_VERSION + SPACES + "/" + meta_props[self._client.repository.ModelMetaNames.SPACE_UID]}})
            if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.PIPELINE_UID, str, False)
                if self._client.WSD:
                    model_meta.update({self.ConfigurationMetaNames.PIPELINE_UID: {
                        "href": self._client.service_instance._href_definitions.get_base_asset_href(meta_props[
                            self._client.repository.ModelMetaNames.PIPELINE_UID])}})
                else:
                    model_meta.update({self.ConfigurationMetaNames.PIPELINE_UID: {
                        "href": API_VERSION + PIPELINES + "/" + meta_props[
                            self._client.repository.ModelMetaNames.PIPELINE_UID]}})
            if self._client.WSD_20:
                if self.ConfigurationMetaNames.MODEL_DEFINITION_UID in meta_props:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_UID, str, False)
                    meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID] = {
                        "id": meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID]}
            else:
                if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.TRAINING_LIB_UID, str, False)
                    model_meta.update({self.ConfigurationMetaNames.TRAINING_LIB_UID: {
                        "href": API_VERSION + LIBRARIES + "/" + meta_props[
                            self._client.repository.ModelMetaNames.TRAINING_LIB_UID]}})
            if self._client.default_project_id is not None:
                model_meta.update({'project': {
                    "href": "/v2/projects" + self._client.default_project_id}})

            model_meta.update({"import": details["entity"]["results_reference"]})
            model_meta["import"]["location"]["path"] = model_path
            runtime_uid = 'hybrid_0.1'
            model_type = 'wml-hybrid_0.1'

            if self.ConfigurationMetaNames.TYPE not in meta_props:
                model_meta.update({"type": model_type})

            if self.ConfigurationMetaNames.RUNTIME_UID not in meta_props and \
                    self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props:
                model_meta.update({"runtime": {"href": "/v4/runtimes/"+runtime_uid}})
            input_schema = []
            output_schema = []
            if self.ConfigurationMetaNames.INPUT_DATA_SCHEMA in meta_props and \
                    meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] is not None:
                if self._client.WSD_20:
                    if isinstance(meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA], dict):
                        input_schema = [meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]]
                    else:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, list, False)
                        input_schema = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
                else:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, dict, False)
                    input_schema = [meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]]
                model_meta.pop(self.ConfigurationMetaNames.INPUT_DATA_SCHEMA)

            if self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA in meta_props and \
                    meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA] is not None:
                if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith('do-') and self._client.WSD_20:
                    try:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict,
                                                 False)
                        output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]
                    except WMLClientError:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, list,
                                                 False)
                        output_schema = meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]
                else:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict, False)
                    output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]
                model_meta.pop(self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA)

            if len(input_schema) != 0 or len(output_schema) != 0:
                model_meta.update({"schemas": {
                    "input": input_schema,
                    "output": output_schema}
                })

            if label_column_names:
                model_meta['label_column'] = label_column_names[0]

            creation_response = requests.post(
               self._wml_credentials['url'] + '/v4/models',
               headers=self._client._get_headers(),
               json=model_meta
            )

            if creation_response.status_code == 201:
                model_details = self._handle_response(201, u'creating new model', creation_response)
                model_uid = model_details['metadata']['id']
            else:
                model_details = self._handle_response(202, u'creating new model', creation_response)
                model_uid = model_details['metadata']['guid']
        else:
            # For V4 cloud prepare the metadata
            if "autoai_sdk" in model_path:
                input_payload = meta_props

            else:
                input_payload = self._create_cloud_model_payload(model_meta, feature_names=feature_names, label_column_names=label_column_names).deepcopy()

            if any([self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40, self._client.ICP_45,
                    self._client.ICP_46, self._client.ICP_47]):
                params = {}
                params.update({'version': self._client.version_param})
                url = self._wml_credentials['url'] + '/ml/v4/models'
            else:
                params = self._client._params()
                url = self._client.service_instance._href_definitions.get_published_models_href()

            if label_column_names:
                input_payload['label_column'] = label_column_names[0]

            creation_response = requests.post(
                url,
                params= params,
                headers=self._client._get_headers(),
                json=input_payload
            )
            if creation_response.status_code == 201:
                model_details = self._handle_response(201, u'creating new model', creation_response)
            else:
                model_details = self._handle_response(202, u'creating new model', creation_response)
            model_uid = model_details['metadata']['id']

            if 'entity' in model_details:
                start_time = time.time()
                elapsed_time = 0
                while model_details['entity'].get('content_import_state') == 'running' and elapsed_time < 60:
                    time.sleep(2)
                    elapsed_time = time.time()-start_time
                    model_details = self.get_details(model_uid)

        return self.get_details(model_uid)

    def _publish_from_file(self, model, meta_props=None, training_data=None, training_target=None, ver=False,
                           artifactid=None, feature_names=None, label_column_names=None):
        """Store saved model into Watson Machine Learning repository on IBM Cloud."""
        if not self._client.WSD_20 and not self._client.CLOUD_PLATFORM_SPACES \
           and not self._client.ICP_35 and not self._client.ICP_40 and not self._client.ICP_45 \
                and not self._client.ICP_46 and not self._client.ICP_47:
            if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props and \
              self.ConfigurationMetaNames.RUNTIME_UID not in meta_props:
                raise WMLClientError("Invalid input. It is mandatory to provide RUNTIME_UID in metaprop.")
        else:
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                    self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props:
                    raise WMLClientError("Invalid input. It is mandatory to provide SOFTWARE_SPEC_UID in metaprop.")
            else:
                if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props and \
                   self.ConfigurationMetaNames.RUNTIME_UID not in meta_props:
                    raise WMLClientError("Invalid input. It is mandatory to provide RUNTIME_UID in metaprop or or "
                                     "SOFTWARE_SPEC_UID in meta_props. RUNTIME_UID is deprecated")
        if(ver==True):
            #check if artifactid is passed
            Models._validate_type(artifactid, 'model_uid', str, True)
            return self._publish_from_archive(model, meta_props, version=ver, artifactid=artifactid, feature_names=feature_names, label_column_names=label_column_names)

        def is_xml(model_filepath):
            if (os.path.splitext(os.path.basename(model_filepath))[-1] == '.pmml'):
                raise WMLClientError('The file name has an unsupported extension. Rename the file with a .xml extension.')
            return os.path.splitext(os.path.basename(model_filepath))[-1] == '.xml'

        def is_h5(model_filepath):
            return os.path.splitext(os.path.basename(model_filepath))[-1] == '.h5'

        def is_json(model_filepath):
            return os.path.splitext(os.path.basename(model_filepath))[-1] == '.json'

        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, str, True)

        import tarfile
        import zipfile

        model_filepath = model
        if os.path.isdir(model):
            # TODO this part is ugly, but will work. In final solution this will be removed
            if "tensorflow" in meta_props[self.ConfigurationMetaNames.TYPE]:
                # TODO currently tar.gz is required for tensorflow - the same ext should be supported for all frameworks
                if os.path.basename(model) == '':
                    model = os.path.dirname(model)
                filename = os.path.basename(model) + '.tar.gz'
                current_dir = os.getcwd()
                os.chdir(model)
                target_path = os.path.dirname(model)

                with tarfile.open(os.path.join('..', filename), mode='w:gz') as tar:
                    tar.add('.')

                os.chdir(current_dir)
                model_filepath = os.path.join(target_path, filename)
                if tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath) or is_xml(model_filepath):
                    if self._client.WSD:
                        return self._wsd_publish_from_archive(model_filepath, meta_props, feature_names=feature_names, label_column_names=label_column_names)
                    else:
                        return self._publish_from_archive(model_filepath, meta_props, feature_names=feature_names, label_column_names=label_column_names)
            else:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.TYPE, str, True)
                if ('caffe' in meta_props[self.ConfigurationMetaNames.TYPE]):
                    raise WMLClientError(u'Invalid model file path  specified for: \'{}\'.'.format(meta_props[self.ConfigurationMetaNames.TYPE]))

                loaded_model = load_model_from_directory(meta_props[self.ConfigurationMetaNames.TYPE], model)
                if self._client.WSD:
                    return self._wsd_publish_from_archive(model_filepath, meta_props, feature_names=feature_names, label_column_names=label_column_names)
                else:
                    if self._client.CLOUD_PLATFORM_SPACES:
                        saved_model = self._publish_from_object_cloud(loaded_model, meta_props, training_data,
                                                                      training_target, feature_names=feature_names,
                                                                      label_column_names=label_column_names)
                    else:
                        saved_model = self._publish_from_object(loaded_model, meta_props, training_data,
                                                                training_target, feature_names=feature_names,
                                                                label_column_names=label_column_names)
                return saved_model

        elif is_xml(model_filepath):
            try:
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_35 and not self._client.ICP_40 \
                        and not self._client.ICP_45 and not self._client.ICP_46 and not self._client.ICP_47:
                    if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SPACE_UID, str, False)
                        meta_props[self._client.repository.ModelMetaNames.SPACE_UID] = {"href": API_VERSION + SPACES + "/" + meta_props[self._client.repository.ModelMetaNames.SPACE_UID]}

                    if self._client.CAMS:
                        if self._client.default_space_id is not None:
                            meta_props[self._client.repository.ModelMetaNames.SPACE_UID] = {
                                "href": API_VERSION + SPACES + "/" + self._client.default_space_id}
                        else:
                            if self._client.default_project_id is not None:
                                meta_props.update({"project": {"href": "/v2/projects/" + self._client.default_project_id}})
                            else:
                                raise WMLClientError(
                                      "It is mandatory to set the space or project." 
                                      " Use client.set.default_space(<SPACE_GUID>) to set the space or Use client.set.default_project(<PROJECT_GUID>).")

                    if self.ConfigurationMetaNames.RUNTIME_UID in meta_props:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.RUNTIME_UID, str, False)
                        meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID] = {"href": API_VERSION + RUNTIMES + "/" + meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID]}

                    if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props:
                        if self._client.WSD_20:
                            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID,
                                                     str,
                                                     False)
                            meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID] = {
                                "base_id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}
                        else:
                            meta_props.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_UID)

                    if self._client.WSD_20:
                        if self.ConfigurationMetaNames.MODEL_DEFINITION_UID in meta_props:
                            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_UID,
                                                     str, False)
                            meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID] = {
                                "id": meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID]}
                    else:
                        if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props:
                            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.TRAINING_LIB_UID, str, False)
                            meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_UID] = {"href": API_VERSION + LIBRARIES + "/" + meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_UID]}
                    if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
                        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.PIPELINE_UID, str, False)
                        if self._client.WSD:
                            meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = {"href": self._client.service_instance._href_definitions.get_base_asset_href(meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID])}
                        else:
                            meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = {
                                "href": API_VERSION + PIPELINES + "/" + meta_props[
                                    self._client.repository.ModelMetaNames.PIPELINE_UID]}
                    meta_data = MetaProps(self._client.repository._meta_props_to_repository_v3_style(meta_props))

                else:
                    # New V4 cloud flow
                    import copy
                    input_meta_data = copy.deepcopy(self._create_cloud_model_payload(meta_props, feature_names=feature_names, label_column_names=label_column_names))
                    meta_data = MetaProps(self._client.repository._meta_props_to_repository_v3_style(input_meta_data))

                model_artifact = MLRepositoryArtifact(str(model_filepath),
                                                      name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                                                      meta_props=meta_data,
                                                      feature_names=feature_names,
                                                      label_column_names=label_column_names)
                if self._client.CAMS:
                    if self._client.default_space_id is not None:
                        query_param_for_repo_client = {'space_id': self._client.default_space_id}
                    else:
                        if self._client.default_project_id is not None:
                            query_param_for_repo_client = {'project_id': self._client.default_project_id}
                        else:
                            query_param_for_repo_client = None
                else:
                    if self._client.WSD:
                        if self._client.default_project_id is not None:
                            query_param_for_repo_client = {'project_id': self._client.default_project_id}
                        else:
                            WMLClientError("Project id is not set for Watson Studio Desktop.")
                    else:
                        query_param_for_repo_client = None
                if self._client.WSD:
                    wsd_url = self._client.service_instance._href_definitions.get_wsd_base_href()
                    saved_model = self._client.repository._ml_repository_client.models.wsd_save(wsd_url, model_artifact,
                                                                                                meta_props, meta_props,
                                                                                                query_param=query_param_for_repo_client,
                                                                                                headers=self._client._get_headers())
                    return self.get_details(u'{}'.format(saved_model['asset_id']))
                else:
                    if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                            self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                        query_param_for_repo_client = self._client._params()
                    saved_model = self._client.repository._ml_repository_client.models.save(model_artifact,query_param_for_repo_client)
            except Exception as e:
                raise WMLClientError(u'Publishing model failed.', e)
            else:
                return self.get_details(saved_model.uid)

        elif tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath):
            if self._client.WSD:
                return self._wsd_publish_from_archive(model, meta_props, feature_names=feature_names, label_column_names=label_column_names)
            else:
                return self._publish_from_archive(model, meta_props, feature_names=feature_names, label_column_names=label_column_names)
        elif is_h5(model_filepath) and self.ConfigurationMetaNames.TYPE in meta_props and \
                meta_props[self.ConfigurationMetaNames.TYPE] == 'tensorflow_2.1':
            return self._store_tf_model(model_filepath, meta_props, feature_names=feature_names, label_column_names=label_column_names)
        elif is_json(model_filepath) and self.ConfigurationMetaNames.TYPE in meta_props and \
                meta_props[self.ConfigurationMetaNames.TYPE] in [f'xgboost_{version}' for version in ("1.3", "1.5")]:

            # validation
            with open(model, 'r') as file:
                try:
                    import json
                    json.loads(file.read())
                except:
                    raise WMLClientError("Json file has invalid content. Please, validate if it was generated with xgboost>=1.3.")

            output_filename = model.replace('.json', '.tar.gz')

            try:
                with tarfile.open(output_filename, "w:gz") as tar:
                    tar.add(model, arcname=os.path.basename(model))
                return self._publish_from_archive(output_filename, meta_props, feature_names=feature_names, label_column_names=label_column_names)
            finally:
                os.remove(output_filename)
        else:
            raise WMLClientError(u'Saving trained model in repository failed. \'{}\' file does not have valid format'.format(model_filepath))

    # TODO make this way when all frameworks will be supported
    # def _publish_from_archive(self, path_to_archive, meta_props=None):
    #     self._validate_meta_prop(meta_props, self.ModelMetaNames.FRAMEWORK_NAME, str, True)
    #     self._validate_meta_prop(meta_props, self.ModelMetaNames.FRAMEWORK_VERSION, str, True)
    #     self._validate_meta_prop(meta_props, self.ModelMetaNames.NAME, str, True)
    #
    #     try:
    #     try:
    #         meta_data = MetaProps(Repository._meta_props_to_repository_v3_style(meta_props))
    #
    #         model_artifact = MLRepositoryArtifact(path_to_archive, name=str(meta_props[self.ModelMetaNames.NAME]), meta_props=meta_data)
    #
    #         saved_model = self._ml_repository_client.models.save(model_artifact)
    #     except Exception as e:
    #         raise WMLClientError(u'Publishing model failed.', e)
    #     else:
    #         return self.get_details(u'{}'.format(saved_model.uid))
    def _create_model_payload(self, meta_props, feature_names=None, label_column_names=None):
        payload = {
            "name": meta_props[self.ConfigurationMetaNames.NAME],
        }
        if self.ConfigurationMetaNames.TAGS in meta_props:
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.TAGS, list, False)
            payload.update({self.ConfigurationMetaNames.TAGS: meta_props[self.ConfigurationMetaNames.TAGS]})
        if self.ConfigurationMetaNames.SPACE_UID in meta_props and \
                meta_props[self._client.repository.ModelMetaNames.SPACE_UID] is not None:
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SPACE_UID, str, False)
            payload.update({self.ConfigurationMetaNames.SPACE_UID: {
                "href": API_VERSION + SPACES + "/" + meta_props[self._client.repository.ModelMetaNames.SPACE_UID]}})

        if self._client.CAMS:
            if self._client.default_space_id is not None:
                meta_props[self._client.repository.ModelMetaNames.SPACE_UID] = {
                    "href": API_VERSION + SPACES + "/" + self._client.default_space_id}
            else:
                if self._client.default_project_id is not None:
                    payload.update({'project': {
                        "href":  "/v2/projects/" + self._client.default_project_id}})
                else:
                    raise WMLClientError(
                        "It is mandatory is set the space. Use client.set.default_space(<SPACE_GUID>) to set the space.")

        if self._client.WSD:
            if self._client.default_project_id is not None:
                payload.update({'project': {
                    "href": "/v2/projects/" + self._client.default_project_id}})
            else:
                raise WMLClientError(
                    "It is mandatory is set the project. Use client.set.default_project(<project id>) to set the project.")
        if self.ConfigurationMetaNames.RUNTIME_UID in meta_props:
            if not self._client.WSD:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.RUNTIME_UID, str, False)
            payload.update({self.ConfigurationMetaNames.RUNTIME_UID: {
                "href": API_VERSION + RUNTIMES + "/" + meta_props[self._client.repository.ModelMetaNames.RUNTIME_UID]}})
        if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props:
            if self._client.WSD_20:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID, str, False)
                payload.update({self.ConfigurationMetaNames.SOFTWARE_SPEC_UID: {
                    "base_id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}})

        if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.PIPELINE_UID, str, False)
            if self._client.WSD:
                payload.update({self.ConfigurationMetaNames.PIPELINE_UID: {
                    "href": self._client.service_instance._href_definitions.get_base_asset_href(meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID])}})
            else:
                payload.update({self.ConfigurationMetaNames.PIPELINE_UID: {
                    "href": API_VERSION + PIPELINES + "/" + meta_props[
                        self._client.repository.ModelMetaNames.PIPELINE_UID]}})

        if self._client.WSD_20:
            if self.ConfigurationMetaNames.MODEL_DEFINITION_UID in meta_props:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_UID, str, False)
                payload.update({self.ConfigurationMetaNames.MODEL_DEFINITION_UID: {
                    "id": meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID]}})
        else:
            if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.TRAINING_LIB_UID, str, False)
                payload.update({self.ConfigurationMetaNames.TRAINING_LIB_UID: {
                    "href": API_VERSION + LIBRARIES + "/" + meta_props[
                        self._client.repository.ModelMetaNames.TRAINING_LIB_UID]}})

        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            payload.update({'description': meta_props[self.ConfigurationMetaNames.DESCRIPTION]})

        # if self.ConfigurationMetaNames.LABEL_FIELD in meta_props and \
        #         meta_props[self.ConfigurationMetaNames.LABEL_FIELD] is not None:
        #     payload.update({'label_column': json.loads(meta_props[self.ConfigurationMetaNames.LABEL_FIELD])})

        if self.ConfigurationMetaNames.TYPE in meta_props:
            payload.update({'type': meta_props[self.ConfigurationMetaNames.TYPE]})

        if self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES in meta_props\
                and meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES] is not None:
            payload.update(
                {'training_data_references': meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES]})

        if self.ConfigurationMetaNames.IMPORT in meta_props and\
                meta_props[self.ConfigurationMetaNames.IMPORT] is not None:
            payload.update({'import': meta_props[self.ConfigurationMetaNames.IMPORT]})
        if self.ConfigurationMetaNames.CUSTOM in meta_props and \
                meta_props[self.ConfigurationMetaNames.CUSTOM] is not None:
            payload.update({'custom': meta_props[self.ConfigurationMetaNames.CUSTOM]})
        if self.ConfigurationMetaNames.DOMAIN in meta_props and\
                meta_props[self.ConfigurationMetaNames.DOMAIN] is not None :
            payload.update({'domain': meta_props[self.ConfigurationMetaNames.DOMAIN]})

        if self.ConfigurationMetaNames.HYPER_PARAMETERS in meta_props and \
                meta_props[self.ConfigurationMetaNames.HYPER_PARAMETERS] is not None:
            payload.update({'hyper_parameters': meta_props[self.ConfigurationMetaNames.HYPER_PARAMETERS]})
        if self.ConfigurationMetaNames.METRICS in meta_props and \
                meta_props[self.ConfigurationMetaNames.METRICS] is not None:
            payload.update({'metrics': meta_props[self.ConfigurationMetaNames.METRICS]})

        input_schema = []
        output_schema = []
        if self.ConfigurationMetaNames.INPUT_DATA_SCHEMA in meta_props and \
                meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] is not None:
            if self._client.WSD_20:
                if isinstance(meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA], dict):
                    input_schema = [meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]]
                else:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, list, False)
                    input_schema = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
            else:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, dict, False)
                input_schema = [meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]]

        if self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA in meta_props and \
                meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA] is not None:
            if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith('do-') and self._client.WSD_20:
                try:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict, False)
                    output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]
                except WMLClientError:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, list, False)
                    output_schema = meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]
            else:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict, False)
                output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]

        if len(input_schema) != 0 or len(output_schema) != 0:
            payload.update({"schemas": {
                    "input": input_schema,
                    "output": output_schema}
                })

        if label_column_names:
            payload['label_column'] = label_column_names[0]

        return payload

    def _publish_from_archive(self, path_to_archive, meta_props=None,version=False,artifactid=None, feature_names=None, label_column_names=None):
        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, str, True)
        def is_xml(model_filepath):
            return os.path.splitext(os.path.basename(model_filepath))[-1] == '.xml'

        url = self._client.service_instance._href_definitions.get_published_models_href()
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or self._client.ICP_45 or \
                self._client.ICP_46 or self._client.ICP_47:
            payload = self._create_cloud_model_payload(meta_props, feature_names=feature_names, label_column_names=label_column_names)
            response = requests.post(
                url,
                json=payload,
                params=self._client._params(),
                headers=self._client._get_headers()
            )
            result = self._handle_response(201, u'creating model', response)
            model_uid = self._get_required_element_from_dict(result, 'model_details', ['metadata', 'id'])
        else:
            payload = self._create_model_payload(meta_props, feature_names=None, label_column_names=None)
            if(version==True):
                response = requests.put(
                    url + "/" + artifactid,
                    json=payload,
                    params=self._client._params(),
                    headers=self._client._get_headers()
                )

            else:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self._client._get_headers()
                    )

            result = self._handle_response(201, u'creating model', response)
            model_uid = self._get_required_element_from_dict(result, 'model_details', ['metadata', 'guid'])
        url = self._client.service_instance._href_definitions.get_published_model_href(model_uid)+"/content"
        with open(path_to_archive, 'rb') as f:
            qparams = self._client._params()
            if is_xml(path_to_archive):
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                        self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                    qparams.update({'content_format': 'coreML'})
                response = requests.put(
                   url,
                   data=f,
                   params=qparams,
                   headers=self._client._get_headers(content_type='application/xml')
                )
            else:
                if not self._ICP:
                    if self._client.CLOUD_PLATFORM_SPACES:
                        qparams.update({'content_format': 'native'})
                        qparams.update({'version': self._client.version_param})
                        model_type = meta_props[self.ConfigurationMetaNames.TYPE]
                        # update the content path for the Auto-ai model.
                        if model_type == 'wml-hybrid_0.1':
                            response = self._upload_autoai_model_content(f, url, qparams)
                        else:
                            # other type of models
                            response = requests.put(
                                url,
                                data=f,
                                params=qparams,
                                headers=self._client._get_headers(content_type='application/octet-stream')
                            )
                    else:
                        response = requests.put(
                            url,
                            data=f,
                            params=self._client._params(),
                            headers=self._client._get_headers(content_type='application/octet-stream')
                        )
                else:
                    if self._client.ICP_35 or self._client.ICP_40 or self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                        qparams.update({'content_format': 'native'})
                        qparams.update({'version': self._client.version_param})
                        model_type = meta_props[self.ConfigurationMetaNames.TYPE]
                        # update the content path for the Auto-ai model.
                        if model_type == 'wml-hybrid_0.1':
                            response = self._upload_autoai_model_content(f, url, qparams)
                        else:
                            response = requests.put(
                                url,
                                data=f,
                                params=qparams,
                                headers=self._client._get_headers(content_type='application/octet-stream')
                            )
                    else:
                        response = requests.put(
                            url,
                            data=f,
                            params=qparams,
                            headers=self._client._get_headers(content_type='application/octet-stream')
                        )
            if response.status_code != 200 and response.status_code != 201 :
                self._delete(model_uid)
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                    self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                self._handle_response(201, u'uploading model content', response, False)
            else:
                self._handle_response(200, u'uploading model content', response, False)
            if(version==True):
                return self._client.repository.get_details(artifactid+"/versions/"+model_uid)
            return self.get_details(model_uid)

    def _wsd_publish_from_archive(self, path_to_archive, meta_props=None, artifactid=None, feature_names=None, label_column_names=None):
        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, str, True)

        def is_xml(model_filepath):
            return os.path.splitext(os.path.basename(model_filepath))[-1] == '.xml'

        url = self._client.service_instance._href_definitions.get_published_models_href()
        payload = self._create_model_payload(meta_props, feature_names=feature_names, label_column_names=label_column_names)

        return self._wsd_create_asset("wml_model", payload, meta_props, path_to_archive, True)

    def store(self, model, meta_props=None, training_data=None, training_target=None, pipeline=None, version=False,
              artifactid=None, feature_names=None, label_column_names=None,subtrainingId=None, round_number=None,
              experiment_metadata=None, training_id=None):
        """Create a model.

        :param model: Can be one of following:

            - The train model object:
                - scikit-learn
                - xgboost
                - spark (PipelineModel)
            - path to saved model in format:\n
                - keras (.tgz)
                - pmml (.xml)
                - scikit-learn (.tar.gz)
                - tensorflow (.tar.gz)
                - spss (.str)
                - spark (.tar.gz)
            - directory containing model file(s):\n
                - scikit-learn
                - xgboost
                - tensorflow
            - unique id of trained model
        :type model: str (for filename or path) or object (corresponding to model type)
        :param meta_props: meta data of the models configuration. To see available meta names use:

            .. code-block:: python

                client._models.ConfigurationMetaNames.get()

        :type meta_props: dict, optional
        :param training_data: Spark DataFrame supported for spark models. Pandas dataframe, numpy.ndarray or array
            supported for scikit-learn models
        :type training_data: spark dataframe, pandas dataframe, numpy.ndarray or array, optional
        :param training_target: array with labels required for scikit-learn models
        :type training_target: array, optional
        :param pipeline: pipeline required for spark mllib models
        :type pipeline: object, optional
        :param feature_names: feature names for the training data in case of Scikit-Learn/XGBoost models,
            this is applicable only in the case where the training data is not of type - pandas.DataFrame
        :type feature_names: numpy.ndarray or list, optional
        :param label_column_names: label column names of the trained Scikit-Learn/XGBoost models
        :type label_column_names: numpy.ndarray or list, optional
        :param round_number: round number of a Federated Learning experiment that has been configured to save
            intermediate models, this applies when model is a training id
        :type round_number: int, optional
        :param experiment_metadata: metadata retrieved from the experiment that created the model
        :type experiment_metadata: dict, optional
        :param training_id: Run id of AutoAI experiment.
        :type training_id: str, optional

        :return: metadata of the model created
        :rtype: dict

        .. note::

            * For a keras model, model content is expected to contain a .h5 file and an archived version of it.

            * `feature_names` is an optional argument containing the feature names for the training data
              in case of Scikit-Learn/XGBoost models. Valid types are numpy.ndarray and list.
              This is applicable only in the case where the training data is not of type - pandas.DataFrame.

            * If the `training_data` is of type pandas.DataFrame and `feature_names` are provided,
              `feature_names` are ignored.

            * For INPUT_DATA_SCHEMA meta prop use list even when passing single input data schema. You can provide
              multiple schemas as dictionaries inside a list.

        **Examples**

        .. code-block:: python

            stored_model_details = client._models.store(model, name)

        In more complicated cases you should create proper metadata, similar to this one:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('scikit-learn_0.23-py3.7')

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'customer satisfaction prediction model',
                client._models.ConfigurationMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
                client._models.ConfigurationMetaNames.TYPE: 'scikit-learn_0.23'
            }

        In case when you want to provide input data schema of the model, you can provide it as part of meta:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('spss-modeler_18.1')

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'customer satisfaction prediction model',
                client._models.ConfigurationMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
                client._models.ConfigurationMetaNames.TYPE: 'spss-modeler_18.1',
                client._models.ConfigurationMetaNames.INPUT_DATA_SCHEMA: [{'id': 'test',
                                                                      'type': 'list',
                                                                      'fields': [{'name': 'age', 'type': 'float'},
                                                                                 {'name': 'sex', 'type': 'float'},
                                                                                 {'name': 'fbs', 'type': 'float'},
                                                                                 {'name': 'restbp', 'type': 'float'}]
                                                                      },
                                                                      {'id': 'test2',
                                                                       'type': 'list',
                                                                       'fields': [{'name': 'age', 'type': 'float'},
                                                                                  {'name': 'sex', 'type': 'float'},
                                                                                  {'name': 'fbs', 'type': 'float'},
                                                                                  {'name': 'restbp', 'type': 'float'}]
                }]
            }

        ``store()`` method used with a local tar.gz file that contains a model:

        .. code-block:: python

            stored_model_details = client._models.store(path_to_tar_gz, meta_props=metadata, training_data=None)

        ``store()`` method used with a local directory that contains model files:

        .. code-block:: python

            stored_model_details = client._models.store(path_to_model_directory, meta_props=metadata, training_data=None)

        ``store()`` method used with the GUID of a trained model:

        .. code-block:: python

            stored_model_details = client._models.store(trained_model_guid, meta_props=metadata, training_data=None)

        ``store()`` method used with a pipeline that was generated by an AutoAI experiment:

        .. code-block:: python

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'AutoAI prediction model stored from object'
            }
            stored_model_details = client._models.store(pipeline_model, meta_props=metadata, experiment_metadata=experiment_metadata)

        .. code-block:: python

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'AutoAI prediction Pipeline_1 model'
            }
            stored_model_details = client._models.store(model="Pipeline_1", meta_props=metadata, training_id = training_id)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if (self._client.CAMS and self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40
            or self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47) and self._client.default_space_id is None \
                and self._client.default_project_id is None:
            raise WMLClientError("It is mandatory is set the space or project. Use client.set.default_space(<SPACE_GUID>) to set the space or client.set.default_project(<PROJECT_GUID>).")
        if self._client.WSD and self._client.default_project_id is None:
            raise WMLClientError("It is mandatory is set the project. Use client.set.default_project(<PROJECT_GUID>).")

        if type(meta_props) is dict and ('project' in meta_props or 'space' in meta_props):
            raise WMLClientError(f"'project' (MetaNames.PROJECT_UID) and 'space' (MetaNames.SPACE_UID) meta names are deprecated and considered as invalid. Instead use client.set.default_space(<SPACE_GUID>) to set the space or client.set.default_project(<PROJECT_GUID>).")

        Models._validate_type(model, u'model', object, True)
        meta_props = copy.deepcopy(meta_props)
        if meta_props.get(self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES):
            meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES] = [p._to_dict() if type(p)==DataConnection else p for p in meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES]]
        if self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES in meta_props and \
            self.ConfigurationMetaNames.INPUT_DATA_SCHEMA not in meta_props and \
            meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES][0].get('schema'):

            try:
                if not meta_props.get(self.ConfigurationMetaNames.LABEL_FIELD) and \
                        meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES][0]['schema'].get('fields'):
                        fields = meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES][0]['schema']['fields']
                        target_fields = [f['name'] for f in fields if f['metadata'].get('modeling_role') == 'target']
                        if target_fields:
                            meta_props[self.ConfigurationMetaNames.LABEL_FIELD] = target_fields[0]

                def is_field_non_label(f):
                    if meta_props.get(self.ConfigurationMetaNames.LABEL_FIELD):
                        return f['name'] != meta_props[self.ConfigurationMetaNames.LABEL_FIELD]
                    else:
                        return True

                if meta_props.get(self.ConfigurationMetaNames.LABEL_FIELD):
                    input_data_schema = {'fields': [f for f in meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES][0].get('schema')['fields'] if \
                                        is_field_non_label(f)], 'type': 'struct', 'id': '1'}

                    meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] = input_data_schema
            except:
                pass
        Models._validate_type(meta_props, u'meta_props', [dict, str], True)
        # Repository._validate_type(training_data, 'training_data', object, False)
        # Repository._validate_type(training_target, 'training_target', list, False)

        if type(meta_props) is str:
            meta_props = {
                self.ConfigurationMetaNames.NAME: meta_props
            }
        if self._client.CAMS and not self._client.ICP_35 and not self._client.ICP_40 and not self._client.ICP_45 and not self._client.ICP_46 and not self._client.ICP_47:
            if self._client.default_space_id is not None:
                meta_props.update(
                    {self._client.repository.ModelMetaNames.SPACE_UID: self._client.default_space_id}
                    )
            if self._client.default_project_id is not None:
                meta_props.update(
                    {'project': {"href": "/v2/projects/" +self._client.default_project_id}}
                )
        if self._client.WSD:
            if self._client.default_project_id is not None:
                meta_props.update(
                    {'project': {"href": "/v2/projects/" +self._client.default_project_id}}
                )
            if "space" in meta_props:
                raise WMLClientError(
                    u'Invalid input SPACE_UID in meta_props. SPACE_UID not supported for WSD.')

        # note: do not validate metaprops when we have them from training microservice (always correct)
        if isinstance(model, str) and "autoai_sdk" in model:
            pass
        elif experiment_metadata or training_id:
            # note: if experiment_metadata are not None it means that the model is created from experiment,
            # and all required information are known from the experiment metadata and the origin
            Models._validate_type(meta_props, u'meta_props', dict, True)
            Models._validate_type(meta_props['name'], u'meta_props.name', str, True)
        else:
            self.ConfigurationMetaNames._validate(meta_props)

        if ("frameworkName" in meta_props):
            framework_name = meta_props["frameworkName"].lower()
            if version == True and (framework_name == "mllib" or framework_name == "wml"):
                raise WMLClientError(u'Unsupported framework name: \'{}\' for creating a model version'.format(framework_name))
        if self.ConfigurationMetaNames.TYPE in meta_props and meta_props[self.ConfigurationMetaNames.TYPE] == u'mllib_2.3':
           # print(
            #    "NOTE!! DEPRECATED!! Spark 2.3 framework for Watson Machine Learning client is deprecated and will be removed on December 1, 2020. Use Spark 2.4 instead. For details, see https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html ")
           raise WMLClientError("Storing Spark 2.3 framework model for Watson Machine Learning client is not supported as of Dec 1st, 2020. Use Spark 2.4 instead. For supported frameworks, see https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html")
            # print( "Storing Spark 2.3 framework model for Watson Machine Learning client is not supported. It is supported in read only mode and will be remove on December 1, 2020. Use Spark 2.4 instead. For details, see https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html ")
            # return
        elif self.ConfigurationMetaNames.RUNTIME_UID in meta_props and meta_props[self.ConfigurationMetaNames.RUNTIME_UID] == u'spark-mllib_2.3':
            #print(
             #   "NOTE!! DEPRECATED!! Spark 2.3 framework for Watson Machine Learning client is deprecated and will be removed on December 1, 2020. Use Spark 2.4 instead. For details, see https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html ")
            raise WMLClientError("Storing Spark 2.3 framework model for Watson Machine Learning client is not supported as of Dec 1st, 2020. Use Spark 2.4 instead. For supported frameworks, see https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html")
            # print(
            #     "Storing Spark 2.3 framework model for Watson Machine Learning client is not supported. It is supported in read only mode and will be remove on December 1, 2020. Use Spark 2.4 instead. For details, see https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html ")
            # return
        if self._client.CAMS and not self._client.ICP_35 and not self._client.ICP_40 and not self._client.ICP_45 \
                and not self._client.ICP_46 and not self._client.ICP_47:
            if self._client.default_space_id is not None:
                meta_props["space"] = self._client.default_space_id

        if not isinstance(model, str):
            if version == True:
                raise WMLClientError(u'Unsupported type: object for param model. Supported types: path to saved model, training ID')
            else:
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                        self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                    if experiment_metadata or training_id:
                        if experiment_metadata:
                            training_id = get_autoai_run_id_from_experiment_metadata(experiment_metadata)

                        # Note: validate if training_id is from AutoAI experiment
                        run_params = self._client.training.get_details(training_uid=training_id, _internal=True)
                        pipeline_id = run_params['entity'].get('pipeline', {}).get('id')
                        pipeline_nodes_list = self._client.pipelines.get_details(pipeline_id)['entity'].get('document', {}).get('pipelines', []) if pipeline_id else []
                        if len(pipeline_nodes_list) == 0 or pipeline_nodes_list[0]['id'] != 'autoai':
                            raise WMLClientError("Parameter training_id or experiment_metadata is not connected to AutoAI training")

                        if is_lale_pipeline(model):
                            model = model.export_to_sklearn_pipeline()

                        schema, artifact_name = prepare_auto_ai_model_to_publish(
                            pipeline_model=model,
                            run_params=run_params,
                            run_id=training_id,
                            wml_client=self._client)

                        new_meta_props = {
                            self._client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                            self._client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                                self._client.software_specifications.get_uid_by_name("hybrid_0.1")
                        }

                        if experiment_metadata:
                            if 'prediction_column' in experiment_metadata:
                                new_meta_props[self._client.repository.ModelMetaNames.LABEL_FIELD] \
                                    = experiment_metadata.get('prediction_column')

                            if 'training_data_references' in experiment_metadata:
                                new_meta_props[self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES] \
                                    = [e._to_dict() if type(e) == DataConnection else e for e in experiment_metadata.get('training_data_references')]
                                if len(new_meta_props[self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES]) > 0:
                                    new_meta_props[self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES][0]['schema'] = schema

                            if 'test_data_references' in experiment_metadata:
                                new_meta_props[self._client.repository.ModelMetaNames.TEST_DATA_REFERENCES] \
                                    = [e._to_dict() if type(e) == DataConnection else e for e in experiment_metadata.get('test_data_references')]
                        else:  # if training_id
                            label_column = None
                            pipeline_details = self._client.pipelines.get_details(run_params['entity']['pipeline']['id'])
                            for node in pipeline_details['entity']['document']['pipelines'][0]['nodes']:
                                if "automl" in node['id'] or 'autoai' in node['id']:
                                    label_column = node.get('parameters', {}).get('optimization', {}).get('label', None)

                            if label_column is not None:
                                new_meta_props[self._client.repository.ModelMetaNames.LABEL_FIELD] = label_column

                            # TODO Is training_data_references and test_data_references needed in meta props??
                            if 'training_data_references' in run_params['entity']:
                                new_meta_props[self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES] \
                                    = run_params['entity']['training_data_references']

                            if 'test_data_references' in run_params['entity']:
                                new_meta_props[self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES] \
                                    = run_params['entity']['test_data_references']

                        if run_params['entity'].get('pipeline', {}).get('id'):
                            new_meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = \
                            run_params['entity']['pipeline']['id']

                        new_meta_props.update(meta_props)

                        saved_model = self._client.repository.store_model(
                            model=artifact_name,
                            meta_props=new_meta_props)
                    else:
                        saved_model = self._publish_from_object_cloud(model=model,
                                                                      meta_props=meta_props,
                                                                      training_data=training_data,
                                                                      training_target=training_target,
                                                                      pipeline=pipeline,
                                                                      feature_names=feature_names,
                                                                      label_column_names=label_column_names)
                else:
                    saved_model = self._publish_from_object(model=model,
                                                            meta_props=meta_props,
                                                            training_data=training_data,
                                                            training_target=training_target,
                                                            pipeline=pipeline,
                                                            feature_names=feature_names,
                                                            label_column_names=label_column_names)
        else:
            if ("tensorflow.python.keras.engine.training.Model" in str(type(model)) or "tf." in str(type(model))) and \
                    str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith('tensorflow_2.1') and \
                    (self._client.ICP_PLATFORM_SPACES or self._client.CLOUD_PLATFORM_SPACES):
                save_model = self._store_tf_model(model, meta_props, feature_names=feature_names,
                                                  label_column_names=label_column_names)

            elif ((model.endswith('.pickle') or model.endswith('pipeline-model.json')) and os.path.sep in model):
                # AUTO AI Trained model
                # pipeline-model.json is needed for OBM + KB
                saved_model = self._store_autoAI_model(model_path=model, meta_props=meta_props,
                                                       feature_names=feature_names,
                                                       label_column_names=label_column_names)
            elif model.startswith("Pipeline_") and (experiment_metadata or training_id):
                if experiment_metadata:
                    training_id = get_autoai_run_id_from_experiment_metadata(experiment_metadata)

                # Note: validate if training_id is from AutoAI experiment
                run_params = self._client.training.get_details(training_uid=training_id, _internal=True)

                # raise an error when TS pipeline is discarded one
                check_if_ts_pipeline_is_winner(details=run_params, model_name=model)

                # Note: We need to fetch credentials when 'container' is the type
                if run_params['entity']['results_reference']['type'] == 'container':
                    data_connection = DataConnection._from_dict(_dict=run_params['entity']['results_reference'])
                    data_connection.set_client(self._client)
                else:
                    data_connection = None
                # --- end note

                artifact_name, model_props = prepare_auto_ai_model_to_publish_normal_scenario(
                    pipeline_model=model,
                    run_params=run_params,
                    run_id=training_id,
                    wml_client=self._client,
                    result_reference=data_connection
                )
                model_props.update(meta_props)

                saved_model = self._client.repository.store_model(
                    model=artifact_name,
                    meta_props=model_props)

            elif (os.path.sep in model) or os.path.isfile(model) or os.path.isdir(model):
                if not os.path.isfile(model) and not os.path.isdir(model):
                    raise WMLClientError(u'Invalid path: neither file nor directory exists under this path: \'{}\'.'.format(model))
                saved_model = self._publish_from_file(model=model, meta_props=meta_props, training_data=training_data,
                                                      training_target=training_target, ver=version,
                                                      artifactid=artifactid, feature_names=feature_names,
                                                      label_column_names=label_column_names)
            else:
                 saved_model = self._publish_from_training(model_uid=model, meta_props=meta_props,
                                                           subtrainingId=subtrainingId, feature_names=feature_names,
                                                           label_column_names=label_column_names, round_number=round_number)
        if "system" in saved_model and 'warnings' in saved_model['system'] and saved_model['system']['warnings']:
            if saved_model['system']['warnings'] is not None:
                message = saved_model['system']['warnings'][0].get('message', None)
                print("Note: Warnings!! : ", message)
        return saved_model

    def update(self, model_uid, meta_props=None, update_model=None):
        """Update existing model.

        :param model_uid: UID of model which define what should be updated
        :type model_uid: str

        :param meta_props: new set of meta_props that needs to updated
        :type meta_props: dict, optional

        :param update_model: archived model content file or path to directory containing archived model file
            which should be changed for specific model_uid, this parameters is valid only for CP4D 3.0.0
        :type update_model: object or model, optional

        :return: updated metadata of model
        :rtype: dict

        **Example**

        .. code-block:: python

            model_details = client.models.update(model_uid, update_model=updated_content)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.WSD:
            raise WMLClientError('Update operation is not for IBM Watson Studio Desktop')
        Models._validate_type(model_uid, 'model_uid', str, True)
        Models._validate_type(meta_props, 'meta_props', dict, True)

        if meta_props is not None: # TODO
            #raise WMLClientError('Meta_props update unsupported.')
            self._validate_type(meta_props, u'meta_props', dict, True)

            url = self._client.service_instance._href_definitions.get_published_model_href(model_uid)

            response = requests.get(
                url,
                params=self._client._params(),
                headers=self._client._get_headers()
            )

            if response.status_code != 200:
                if response.status_code == 404:
                    raise WMLClientError(
                        u'Invalid input. Unable to get the details of model_uid provided.')
                else:
                    raise ApiRequestFailure(u'Failure during {}.'.format("getting model to update"), response)

            details = self._handle_response(200, "Get model details", response)
            model_type = details['entity']['type']
            # update the content path for the Auto-ai model.
            if model_type == 'wml-hybrid_0.1' and update_model is not None:
                # The only supported format is a zip file containing `pipeline-model.json` and pickled model compressed
                # to tar.gz format.
                if not update_model.endswith('.zip'):
                    raise WMLClientError(
                        u'Invalid model content. The model content file should bre zip archive containing'
                        u' ".pickle.tar.gz" file or "pipline-model.json", for the model type\'{}\'.'.format(
                            model_type))

            # with validation should be somewhere else, on the begining, but when patch will be possible
            patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], meta_props, with_validation=True)
            response_patch = requests.patch(url, json=patch_payload, params=self._client._params(),headers=self._client._get_headers())
            updated_details = self._handle_response(200, u'model version patch', response_patch)
            if (self._client.ICP_35 or self._client.ICP_40 or self._client.ICP_45 or
                self._client.ICP_46 or self._client.ICP_47 or self._client.CLOUD_PLATFORM_SPACES ) and update_model is not None:
                self._update_model_content(model_uid, details, update_model)
            return updated_details

        return self.get_details(model_uid)

    def load(self, artifact_uid):
        """Load model from repository to object in local environment.

        :param artifact_uid: stored model UID
        :type artifact_uid: str

        :return: trained model
        :rtype: object

        **Example**

        .. code-block:: python

            model = client.models.load(model_uid)
        """
        if self._client.WSD:
            raise WMLClientError('Load model operation is not supported in IBM Watson Studio Desktop.')
        Models._validate_type(artifact_uid, u'artifact_uid', str, True)
        #check if this is tensorflow 2.x model type
        if self._client.CAMS or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
            model_details = self.get_details(artifact_uid)
            if model_details.get('entity').get('type').startswith('tensorflow_2.'):
                return self._tf2x_load_model_instance(artifact_uid)
        try:
            # Cloud Convergence: CHK IF THIS CONDITION IS CORRECT since loaded_model
            # functionality below
            if self._client.CAMS or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 \
                    or self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                if self._client.default_space_id is None and self._client.default_project_id is None:
                    raise WMLClientError(
                        "It is mandatory is set the space or project. \
                        Use client.set.default_space(<SPACE_GUID>) to set the space or client.set.default_project(<PROJECT_GUID>).")
                else:
                    if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                            self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                        query_param = self._client._params()
                        loaded_model = self._client.repository._ml_repository_client.models._get_v4_cloud_model(artifact_uid,
                                                                                                query_param=query_param)
                    elif self._client.default_project_id is not None:
                        loaded_model = self._client.repository._ml_repository_client.models.get(artifact_uid,
                                                                                                project_id=self._client.default_project_id)
                    else:
                        loaded_model = self._client.repository._ml_repository_client.models.get(artifact_uid, space_id=self._client.default_space_id)
            else:
                loaded_model = self._client.repository._ml_repository_client.models.get(artifact_uid)
            loaded_model = loaded_model.model_instance()
            self._logger.info(u'Successfully loaded artifact with artifact_uid: {}'.format(artifact_uid))
            return loaded_model
        except Exception as e:
            raise WMLClientError(u'Loading model with artifact_uid: \'{}\' failed.'.format(artifact_uid), e)

    def download(self, model_uid, filename='downloaded_model.tar.gz', rev_uid=None, format=None):
        """Download model from repository to local file.

        :param model_uid: stored model UID
        :type model_uid: str

        :param filename: name of local file to create
        :type filename: str, optional

        :param rev_uid: revision id, applicable only for IBM Cloud Pak® for Data 3.0 onwards
        :type rev_uid: str, optional

        :param format: format of the content, applicable only for IBM Cloud version V4
        :type format: str, optional

        **Example**

        .. code-block:: python

            client.models.download(model_uid, 'my_model.tar.gz')
        """
        if os.path.isfile(filename):
            raise WMLClientError(u'File with name: \'{}\' already exists.'.format(filename))
        if rev_uid is not None and not self._client.CLOUD_PLATFORM_SPACES\
                and self._client.ICP_35 is None and self._client.ICP_40 is None and self._client.ICP_45 is None \
                and self._client.ICP_46 is None and self._client.ICP_47 is None:
            raise WMLClientError(u'Applicable only for IBM Cloud Pak® For Data 3.0 onwards')

        Models._validate_type(model_uid, u'model_uid', str, True)
        Models._validate_type(filename, u'filename', str, True)

        if filename.endswith('.json'):
            is_json = True
            json_filename = filename
            import uuid
            filename = f'tmp_{uuid.uuid4()}.tar.gz'
        else:
            is_json = False

        artifact_url = self._client.service_instance._href_definitions.get_model_last_version_href(model_uid)
        params = self._client._params()
        try:
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                import urllib
                url = self._client.service_instance._href_definitions.get_published_model_href(model_uid)
                model_get_response = requests.get(url,
                                                  params=self._client._params(),
                                                  headers=self._client._get_headers())

                model_details = self._handle_response(200, u'get model', model_get_response)
                if rev_uid is not None:
                    params.update({'revision_id': rev_uid})

                model_type = model_details['entity']['type']
                if (model_type.startswith('keras_') or model_type.startswith('scikit-learn_')
                    or model_type.startswith('xgboost_')) and format is not None:
                    Models._validate_type(format, u'format', str, False)
                    if str(format).upper() == 'COREML':
                        params.update({'content_format': 'coreml'})
                    else:
                        params.update({'content_format': 'native'})
                else:
                    params.update({'content_format': 'native'})
                artifact_content_url = str(artifact_url + u'/download')
                if model_details['entity']['type'] == 'wml-hybrid_0.1':
                    self._download_auto_ai_model_content(model_uid, artifact_content_url, filename)
                    print(u'Successfully saved model content to file: \'{}\''.format(filename))
                    return os.getcwd() + "/" + filename
                else:
                    r = requests.get(artifact_content_url, params=params,
                                     headers=self._client._get_headers(), stream=True)

            elif not self._client.ICP and not self._client.WSD and not self._client.CLOUD_PLATFORM_SPACES:
                #// old cloud path but not convergence platform.
                artifact_content_url = str(artifact_url + u'/content')
                r = requests.get(artifact_content_url, params=self._client._params(),headers=self._client._get_headers(), stream=True)
            else:
                #except ICP_35 all other non cloud cases
                artifact_content_url = str(artifact_url + u'/content')
                if rev_uid is not None:
                    params.update({'revision_id': rev_uid})
                if not self._client.ICP_35 and not self._client.ICP_40 and not self._client.ICP_45 \
                        and not self._client.ICP_46 and not self._client.ICP_47:
                    r = requests.get(artifact_content_url, params=params,headers=self._client._get_headers(), stream=True)
            if r.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), r)

            downloaded_model = r.content
            self._logger.info(u'Successfully downloaded artifact with artifact_url: {}'.format(artifact_url))
        except WMLClientError as e:
            raise e
        except Exception as e:
            if artifact_url is not None:
                raise WMLClientError(u'Downloading model with artifact_url: \'{}\' failed.'.format(artifact_url), e)
            else:
                raise WMLClientError(u'Downloading model failed.',e)
        finally:
            if is_json:
                try:
                    os.remove(filename)
                except:
                    pass
        try:
            with open(filename, 'wb') as f:
                f.write(downloaded_model)

            if is_json:
                import tarfile

                tar = tarfile.open(filename, "r:gz")
                file_name = tar.getnames()[0]
                if not file_name.endswith('.json'):
                    raise WMLClientError('Downloaded model is not json.')
                tar.extractall()
                tar.close()
                os.rename(file_name, json_filename)

                os.remove(filename)
                filename = json_filename

            print(u'Successfully saved model content to file: \'{}\''.format(filename))
            return os.getcwd() + "/"+filename
        except IOError as e:
            raise WMLClientError(u'Saving model with artifact_url: \'{}\' failed.'.format(filename), e)

    def delete(self, model_uid):
        """Delete model from repository.

        :param model_uid: stored model UID
        :type model_uid: str

        **Example**

        .. code-block:: python

            client.models.delete(model_uid)
        """
        Models._validate_type(model_uid, u'model_uid', str, True)

        if self._client.WSD:
            model_endpoint = self._client.service_instance._href_definitions.get_model_definition_assets_href() + "/" + model_uid
        else:
            model_endpoint = self._client.service_instance._href_definitions.get_published_model_href(model_uid)

        self._logger.debug(u'Deletion artifact model endpoint: {}'.format(model_endpoint))
        if not self._ICP and not self._client.WSD:
            if self._client.CLOUD_PLATFORM_SPACES and self._if_deployment_exist_for_asset(model_uid):
                raise WMLClientError(
                    u'Cannot delete model that has existing deployments. Please delete all associated deployments and try again')

            response_delete = requests.delete(model_endpoint, params=self._client._params(),headers=self._client._get_headers())
        else:
            #check if the model as a corresponding deployment
            if not self._client.WSD and self._if_deployment_exist_for_asset(model_uid):
                raise WMLClientError(u'Cannot delete model that has existing deployments. Please delete all associated deployments and try again')
            response_delete = requests.delete(model_endpoint, params=self._client._params(),headers=self._client._get_headers())
        return self._handle_response(204, u'model deletion', response_delete, False)

    def _delete(self, model_uid):
        Models._validate_type(model_uid, u'model_uid', str, True)

        if self._client.WSD:
            model_endpoint = self._client.service_instance._href_definitions.get_model_definition_assets_href() + "/" + model_uid
        else:
            model_endpoint = self._client.service_instance._href_definitions.get_published_model_href(model_uid)

        self._logger.debug(u'Deletion artifact model endpoint: {}'.format(model_endpoint))
        if not self._ICP and not self._client.WSD:
            response_delete = requests.delete(model_endpoint, params=self._client._params(),
                                              headers=self._client._get_headers())
        else:
            response_delete = requests.delete(model_endpoint, params=self._client._params(),
                                              headers=self._client._get_headers())

    def get_details(self, model_uid=None, limit=None, asynchronous=False, get_all=False, spec_state=None):
        """Get metadata of stored models. If model uid is not specified returns all models metadata.

        :param model_uid: stored model, definition or pipeline UID
        :type model_uid: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional
        
        :param spec_state: software specification state, can be used only when `model_uid` is None
        :type spec_state: SpecStates, optional

        :return: stored model(s) metadata
        :rtype: dict (if UID is not None) or {"resources": [dict]} (if UID is None)

        .. note::
            In current implementation setting `spec_state=True` may break set `limit`,
            returning less records than stated by set `limit`.

        **Example**

        .. code-block:: python

            model_details = client.models.get_details(model_uid)
            models_details = client.models.get_details()
            models_details = client.models.get_details(limit=100)
            models_details = client.models.get_details(limit=100, get_all=True)
            models_details = []
            for entry in client.models.get_details(limit=100, asynchronous=True, get_all=True):
                models_details.extend(entry)
        """
        if limit and spec_state:
            print('Warning: In current implementation setting `spec_state=True` may break set `limit`, '
                  'returning less records than stated by set `limit`.')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_uid, u'model_uid', str, False)
        Models._validate_type(limit, u'limit', int, False)

        if not self._ICP:
            if self._client.WSD:
                url = self._client.service_instance._href_definitions.get_model_definition_assets_href()
                response = self._get_artifact_details(url, model_uid, limit, 'models')
                return self._wsd_get_required_element_from_response(response)
            else:
                url = self._client.service_instance._href_definitions.get_published_models_href()
        else:
            url = self._client.service_instance._href_definitions.get_published_models_href()

        if model_uid is None:
            filter_func = self._get_filter_func_by_spec_ids(self._get_and_cache_spec_ids_for_state(spec_state)) \
                if spec_state else None

            return self._get_artifact_details(url, model_uid, limit, 'models', _async=asynchronous, _all=get_all,
                                              _filter_func=filter_func)

        else:
            return self._get_artifact_details(url, model_uid, limit, 'models')

    @staticmethod
    def get_href(model_details):
        """Get url of stored model.

        :param model_details: stored model details
        :type model_details: dict

        :return: url to stored model
        :rtype: str

        **Example**

        .. code-block:: python

            model_url = client.models.get_href(model_details)
        """

        Models._validate_type(model_details, u'model_details', object, True)

        if 'asset_id' in model_details['metadata']:
            return WMLResource._get_required_element_from_dict(model_details, u'model_details', [u'metadata', u'href'])
        else:
            if 'id' not in model_details['metadata']:
                Models._validate_type_of_details(model_details, MODEL_DETAILS_TYPE)
                return WMLResource._get_required_element_from_dict(model_details, u'model_details', [u'metadata', u'href'])
            else:
                model_id = WMLResource._get_required_element_from_dict(model_details, u'model_details', [u'metadata', u'id'])
                return "/ml/v4/models/" + model_id

    @staticmethod
    def get_uid(model_details):
        """Get uid of stored model.

        :param model_details: stored model details
        :type model_details: dict

        :return: uid of stored model
        :rtype: str

        **Example**

        .. code-block:: python

            model_uid = client.models.get_uid(model_details)
        """
        Models._validate_type(model_details, u'model_details', object, True)

        if 'asset_id' in model_details['metadata']:
            return WMLResource._get_required_element_from_dict(model_details, u'model_details', [u'metadata', u'asset_id'])
        else:
            if 'id' not in model_details['metadata']:
                Models._validate_type_of_details(model_details, MODEL_DETAILS_TYPE)
                return WMLResource._get_required_element_from_dict(model_details, u'model_details', [u'metadata', u'guid'])
            else:
                return WMLResource._get_required_element_from_dict(model_details, u'model_details',
                                                                   [u'metadata', u'id'])

    @staticmethod
    def get_id(model_details):
        """Get id of stored model.

        :param model_details: stored model details
        :type model_details: dict

        :return: uid of stored model
        :rtype: str

        **Example**

        .. code-block:: python

            model_id = client.models.get_id(model_details)
        """
        Models._validate_type(model_details, u'model_details', object, True)

        if 'asset_id' in model_details['metadata']:
            return WMLResource._get_required_element_from_dict(model_details, u'model_details',
                                                               [u'metadata', u'asset_id'])
        else:
            if 'id' not in model_details['metadata']:
                Models._validate_type_of_details(model_details, MODEL_DETAILS_TYPE)
                return WMLResource._get_required_element_from_dict(model_details, u'model_details',
                                                                   [u'metadata', u'guid'])
            else:
                return WMLResource._get_required_element_from_dict(model_details, u'model_details',
                                                                   [u'metadata', u'id'])

    def list(self, limit=None, asynchronous=False, get_all=False, return_as_df=True):
        """Print stored models in a table format. If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed models or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.models.list()
            client.models.list(limit=100)
            client.models.list(limit=100, get_all=True)
            [entry for entry in client.models.list(limit=100, asynchronous=True, get_all=True)]
        """
        ##For CP4D, check if either spce or project ID is set
        def process_resources(self, model_resources: dict, return_as_df=True):
            model_resources = model_resources['resources']
            if not self._client.CLOUD_PLATFORM_SPACES and self._client.ICP_35 is None\
                    and self._client.ICP_40 is None and self._client.ICP_45 is None and self._client.ICP_46 is None and self._client.ICP_47 is None:
                model_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], m[u'entity'][u'type']) for m in model_resources]

                self._list(model_values, [u'GUID', u'NAME', u'CREATED', u'TYPE'], limit, _DEFAULT_LIST_LENGTH)
            else:
                sw_spec_info = {s['id']: s
                                for s in self._client.software_specifications.get_details(state_info=True)['resources']}

                def get_spec_info(spec_id, prop):
                    if spec_id and spec_id in sw_spec_info:
                        return sw_spec_info[spec_id].get(prop, '')
                    else:
                        return ''

                model_values = [(m[u'metadata'][u'id'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at'],
                                 m[u'entity'][u'type'],
                                 get_spec_info(m['entity'].get('software_spec', {}).get('id'), 'state'),
                                 get_spec_info(m['entity'].get('software_spec', {}).get('id'), 'replacement'))
                                for m in model_resources]

                table = self._list(model_values, [u'ID', u'NAME', u'CREATED', u'TYPE', 'SPEC_STATE', 'SPEC_REPLACEMENT'], limit, _DEFAULT_LIST_LENGTH)
                if return_as_df:
                    return table

        if self._client.WSD:
            self._wsd_list(limit)

        else:
            self._client._check_if_either_is_set()
            if asynchronous:
                return (process_resources(self, model_resources, return_as_df) for model_resources in
                        self.get_details(limit=limit, asynchronous=asynchronous, get_all=get_all))


            else:
                model_resources = self.get_details(limit=limit, get_all=get_all)
                return process_resources(self, model_resources, return_as_df)

    def _wsd_list(self, limit=None):

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        href = self._client.service_instance._href_definitions.get_asset_search_href("wml_model")
        if limit is None:
            data = {
                "query": "*:*"
            }
        else:
            Models._validate_type(limit, u'limit', int, False)
            data = {
                "query": "*:*",
                "limit": limit
            }

        response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(),
                                     json=data)
        self._handle_response(200, u'model list', response)
        asset_details = self._handle_response(200, u'model list', response)["results"]
        model_def_values = [
            (m[u'metadata'][u'name'], m[u'metadata'][u'asset_type'], m[u'metadata'][u'asset_id']) for
            m in asset_details]

        self._list(model_def_values, [u'NAME', u'ASSET_TYPE', u'GUID'], limit, _DEFAULT_LIST_LENGTH)

    def create_revision(self, model_uid):
        """Create revision for the given model uid.

        :param model_uid: stored model UID
        :type model_uid: str

        :return: stored model revisions metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            model_details = client.models.create_revision(model_uid)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_uid, u'model_uid', str, False)

        if not self._client.CLOUD_PLATFORM_SPACES\
                and self._client.ICP_35 is None and self._client.ICP_40 is None and self._client.ICP_45 is None \
                and self._client.ICP_46 is None and self._client.ICP_47 is None:
            raise WMLClientError(
                u'Revisions APIs are not supported in this release.')
        else:
            url = self._client.service_instance._href_definitions.get_published_models_href()
            return self._create_revision_artifact(url, model_uid, 'models')

    def list_revisions(self, model_uid, limit=None, return_as_df=True):
        """Print all revision for the given model uid in a table format.

        :param model_uid: Unique id of stored model
        :type model_uid: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed revisions or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.models.list_revisions(model_uid)
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Models._validate_type(model_uid, u'model_uid', str, True)

        table=None

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_35 \
                and not self._client.ICP_40 and not self._client.ICP_45 and not self._client.ICP_46 and not self._client.ICP_47:
            raise WMLClientError(
                u'Revision APIs are not supported in this release.')
        else:
            url = self._client.service_instance._href_definitions.get_published_models_href() + "/" + model_uid
            model_resources = self._get_artifact_details(url, "revisions", limit, 'model revisions')[u'resources']
            model_values = [
                (m[u'metadata'][u'rev'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at']) for m in
                model_resources]

            table = self._list(model_values, [u'ID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def get_revision_details(self, model_uid, rev_uid):
        """Get metadata of stored models specific revision.

        :param model_uid: stored model, definition or pipeline UID
        :type model_uid: str

        :param rev_uid: Unique Id of the stored model revision
        :type rev_uid: int

        :return: stored model(s) metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            model_details = client.models.get_revision_details(model_uid, rev_uid)
        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_uid, u'model_uid', str, True)
        Models._validate_type(rev_uid, u'rev_uid', int, True)
       # Models._validate_type(limit, u'limit', int, False)

        if not self._client.CLOUD_PLATFORM_SPACES\
                and not self._client.ICP_35 and not self._client.ICP_40 and not self._client.ICP_45 \
                and not self._client.ICP_46 and not self._client.ICP_47:
            raise WMLClientError(
                'Revision APIs are not supported in this release.')
        else:
            url = self._client.service_instance._href_definitions.get_published_models_href() + "/" + model_uid
            return self._get_with_or_without_limit(url, limit=None, op_name="model",
                                                   summary=None, pre_defined=None, revision=rev_uid)

    def promote(self, model_id: str, source_project_id: str, target_space_id: str):
        """Promote model from project to space. Supported only for IBM Cloud Pak® for Data.
        
        *Deprecated:* Use `client.spaces.promote(model_id, source_project_id, target_space_id)` instead.
        """
        print("Note: Function `client.repository.promote_model(model_id, source_project_id, target_space_id)` "
              "has been deprecated. Use `client.spaces.promote(model_id, source_project_id, target_space_id)` instead.")
        try:
            return self._client.spaces.promote(model_id, source_project_id, target_space_id)
        except PromotionFailed as e:
            raise ModelPromotionFailed(e.project_id, e.space_id, e.promotion_response, e.reason)

    def _update_model_content(self, model_uid, updated_details, update_model):

        model = copy.copy(update_model)
        model_type = updated_details['entity']['type']

        def is_xml(model_filepath):
            if (os.path.splitext(os.path.basename(model_filepath))[-1] == '.pmml'):
                raise WMLClientError(
                    'The file name has an unsupported extension. Rename the file with a .xml extension.')
            return os.path.splitext(os.path.basename(model_filepath))[-1] == '.xml'

        import tarfile
        import zipfile
        model_filepath = model

        if 'scikit-learn_' in model_type or 'mllib_' in model_type:
            meta_props = updated_details['entity']
            meta_data = MetaProps(self._client.repository._meta_props_to_repository_v3_style(meta_props))
            name = updated_details['metadata']['name']
            model_artifact = MLRepositoryArtifact(update_model, name=name,
                                                  meta_props=meta_data, training_data=None)
            model_artifact.uid = model_uid
            query_params = self._client._params()
            query_params.update({'content_format': "native"})
            self._client.repository._ml_repository_client.models.upload_content(model_artifact,
                                                                                query_param=query_params, no_delete=True)
        else:
            if (os.path.sep in update_model) or os.path.isfile(update_model) or os.path.isdir(update_model):
                if not os.path.isfile(update_model) and not os.path.isdir(update_model):
                    raise WMLClientError(
                        u'Invalid path: neither file nor directory exists under this path: \'{}\'.'.format(model))

            if os.path.isdir(model):
                if "tensorflow" in model_type:
                    # TODO currently tar.gz is required for tensorflow - the same ext should be supported for all frameworks
                    if os.path.basename(model) == '':
                        model = os.path.dirname(update_model)
                    filename = os.path.basename(update_model) + '.tar.gz'
                    current_dir = os.getcwd()
                    os.chdir(model)
                    target_path = os.path.dirname(model)

                    with tarfile.open(os.path.join('..', filename), mode='w:gz') as tar:
                        tar.add('.')

                    os.chdir(current_dir)
                    model_filepath = os.path.join(target_path, filename)
                    if tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath) or is_xml(model_filepath):
                        path_to_archive = model_filepath
                else:
                    if 'caffe' in model_type:
                        raise WMLClientError(u'Invalid model file path  specified for: \'{}\'.'.format(model_type))
                    loaded_model = load_model_from_directory(model_type, model)
                    path_to_archive = loaded_model
            elif is_xml(model_filepath):
                 path_to_archive = model_filepath
            elif tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath):
                 path_to_archive = model_filepath

            else:
                raise WMLClientError(
                    u'Saving trained model in repository failed. \'{}\' file does not have valid format'.format(
                        model_filepath))

            url = self._client.service_instance._href_definitions.get_published_model_href(model_uid) + "/content"
            with open(path_to_archive, 'rb') as f:
                if is_xml(path_to_archive):
                    response = requests.put(
                        url,
                        data=f,
                        params=self._client._params(),
                        headers=self._client._get_headers(content_type='application/xml')
                    )
                else:
                    qparams = self._client._params()

                    if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or
                        self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47) and model_type.startswith('wml-hybrid_0'):
                        response = self._upload_autoai_model_content(f, url, qparams)
                    else:
                        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                                self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                            qparams.update({'content_format': "native"})
                        response = requests.put(
                            url,
                            data=f,
                            params=qparams,
                            headers=self._client._get_headers(content_type='application/octet-stream')
                        )
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                        self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                    self._handle_response(201, u'uploading model content', response, False)
                else:
                    self._handle_response(200, u'uploading model content', response, False)

    def _create_cloud_model_payload(self, meta_props, feature_names=None, label_column_names=None):
        metadata = copy.deepcopy(meta_props)
        if self._client.default_space_id is not None:
            metadata['space_id'] = self._client.default_space_id
        else:
            if self._client.default_project_id is not None:
                metadata.update({'project_id': self._client.default_project_id})
            else:
                raise WMLClientError("It is mandatory is set the space or Project. \
                 Use client.set.default_space(<SPACE_GUID>) to set the space or"
                                     " Use client.set.default_project(<PROJECT_ID)")

        if self.ConfigurationMetaNames.RUNTIME_UID in meta_props and  \
            self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props:
            raise WMLClientError(
                u'Invalid input.  RUNTIME_UID is not supported in cloud environment. Specify SOFTWARE_SPEC_UID')

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props:
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID, str, True)
            metadata.update({self.ConfigurationMetaNames.SOFTWARE_SPEC_UID: {
                    "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}})

        if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.PIPELINE_UID, str, False)
            metadata.update({self.ConfigurationMetaNames.PIPELINE_UID: {
                    "id":  meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID]}})

        if self.ConfigurationMetaNames.MODEL_DEFINITION_UID in meta_props:
            self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_UID, str, False)
            metadata.update({self.ConfigurationMetaNames.MODEL_DEFINITION_UID: {
                "id": meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID]}})

        if self.ConfigurationMetaNames.IMPORT in meta_props and\
                meta_props[self.ConfigurationMetaNames.IMPORT] is not None:
            print("WARNING: Invalid input. IMPORT is not supported in cloud environment.")

        if self.ConfigurationMetaNames.TRAINING_LIB_UID in meta_props and \
           meta_props[self.ConfigurationMetaNames.IMPORT] is not None:
            print("WARNING: Invalid input. TRAINING_LIB_UID is not supported in cloud environment.")

        input_schema = []
        output_schema = []
        if self.ConfigurationMetaNames.INPUT_DATA_SCHEMA in meta_props and \
           meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] is not None:
            if isinstance(meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA], list):
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, list, False)
                input_schema = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
            else:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, dict, False)
                input_schema = [meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]]
            metadata.pop(self.ConfigurationMetaNames.INPUT_DATA_SCHEMA)

        if self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA in meta_props and \
                meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA] is not None:
            if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith('do-'):
                if isinstance(meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA], dict):
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict, False)
                    output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]
                else:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, list, False)
                    output_schema = meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]
            else:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict, False)
                output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]
            metadata.pop(self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA)

        if len(input_schema) != 0 or len(output_schema) != 0:
            metadata.update({"schemas": {
                "input": input_schema,
                "output": output_schema}
            })

        if label_column_names:
            metadata['label_column'] = label_column_names[0]

        return metadata

    def _publish_from_object_cloud(self, model, meta_props, training_data=None, training_target=None, pipeline=None,
                             feature_names=None, label_column_names=None, project_id=None):
        """Store model from object in memory into Watson Machine Learning repository on Cloud."""
        self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.NAME, str, True)
        if self.ConfigurationMetaNames.RUNTIME_UID in meta_props and \
           self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props:
            raise WMLClientError("Invalid input. RUNTIME_UID is no longer supported, instead of that provide SOFTWARE_SPEC_UID in meta_props.")
        else:
            if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID not in meta_props and \
                    self.ConfigurationMetaNames.RUNTIME_UID not in meta_props:
                raise WMLClientError(
                    "Invalid input. It is mandatory to provide SOFTWARE_SPEC_UID in meta_props.")
        try:
            if 'pyspark.ml.pipeline.PipelineModel' in str(type(model)):
                if (pipeline is None or training_data is None):
                    raise WMLClientError(u'pipeline and training_data are expected for spark models.')
                name = meta_props[self.ConfigurationMetaNames.NAME]
                version = "1.0"
                platform = {"name": "python", "versions": ["3.6"]}
                library_tar = self._save_library_archive(pipeline)
                model_definition_props = {
                    self._client.model_definitions.ConfigurationMetaNames.NAME: name + "_" + uid_generate(8),
                    self._client.model_definitions.ConfigurationMetaNames.VERSION: version,
                    self._client.model_definitions.ConfigurationMetaNames.PLATFORM: platform,
                }
                model_definition_details = self._client.model_definitions.store(library_tar, model_definition_props)
                model_definition_id = self._client.model_definitions.get_uid(model_definition_details)
               #create a pipeline for model definition
                pipeline_metadata = {
                    self._client.pipelines.ConfigurationMetaNames.NAME: name + "_" + uid_generate(8),
                    self._client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                        "doc_type": "pipeline",
                        "version": "2.0",
                        "primary_pipeline": "dlaas_only",
                        "pipelines": [
                            {
                                "id": "dlaas_only",
                                "runtime_ref": "spark",
                                "nodes": [
                                    {
                                        "id": "repository",
                                        "type": "model_node",
                                        "inputs": [
                                        ],
                                        "outputs": [],
                                        "parameters": {
                                            "model_definition": {"id": model_definition_id}
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }

                pipeline_save = self._client.pipelines.store(pipeline_metadata)
                pipeline_id = self._client.pipelines.get_uid(pipeline_save)
                meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = {"id": pipeline_id}
            else:
                if self.ConfigurationMetaNames.PIPELINE_UID in meta_props:
                    self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.PIPELINE_UID, str, False)
                    meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID] = {
                        "id":  meta_props[self._client.repository.ModelMetaNames.PIPELINE_UID]}

            if self.ConfigurationMetaNames.SPACE_UID in meta_props and \
               meta_props[self._client.repository.ModelMetaNames.SPACE_UID] is not None:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SPACE_UID, str, False)
                meta_props['space_id'] = meta_props[self._client.repository.ModelMetaNames.SPACE_UID]
                meta_props.pop(self.ConfigurationMetaNames.SPACE_UID)
            else:
                if self._client.default_project_id is not None:
                    meta_props['project_id'] = self._client.default_project_id
                else:
                    meta_props['space_id'] = self._client.default_space_id

            if self.ConfigurationMetaNames.SOFTWARE_SPEC_UID in meta_props:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_UID, str,
                                         True)
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID] = {
                    "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_UID]}

            if self.ConfigurationMetaNames.MODEL_DEFINITION_UID in meta_props:
                self._validate_meta_prop(meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_UID, str,
                                         False)
                meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID] = {
                    "id": meta_props[self._client.repository.ModelMetaNames.MODEL_DEFINITION_UID]}

            tf_model_types = [f'tensorflow_{version}' for version in ('2.1', '2.4', '2.7', '2.9')]
            if str(meta_props[self.ConfigurationMetaNames.TYPE]) in tf_model_types and \
                    (self._client.ICP_PLATFORM_SPACES or self._client.CLOUD_PLATFORM_SPACES):
                saved_model = self._store_tf_model(model, meta_props, feature_names=feature_names, label_column_names=label_column_names)
                return saved_model
            else:
                meta_data = MetaProps(self._client.repository._meta_props_to_repository_v3_style(meta_props))
                model_artifact = MLRepositoryArtifact(model, name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                                                      meta_props=meta_data,
                                                      training_data=training_data,
                                                      training_target=training_target,
                                                      feature_names=feature_names,
                                                      label_column_names=label_column_names)
                query_param_for_repo_client = self._client._params()
                saved_model = self._client.repository._ml_repository_client.models.save(model_artifact,
                                                                                        query_param=query_param_for_repo_client)
                return self.get_details(u'{}'.format(saved_model.uid))

        except Exception as e:
            raise WMLClientError(u'Publishing model failed.', e)

    def _upload_autoai_model_content(self, file, url, qparams):
        import zipfile
        import json
        node_ids = None
        with zipfile.ZipFile(file, 'r') as zipObj:
            # Get a list of all archived file names from the zip
            listOfFileNames = zipObj.namelist()
            t1 = zipObj.extract('pipeline-model.json')
            with open(t1, 'r') as f2:
                data = json.load(f2)
                # note: we can have multiple nodes (OBM + KB)
                node_ids = [node.get('id') for node in data.get('pipelines')[0].get('nodes')]

            if node_ids is None:
                raise ApiRequestFailure(u'Invalid pipline-model.json content file. There is no node id value found')

            qparams.update({'content_format': 'native'})
            qparams.update({'name': 'pipeline-model.json'})

            if 'pipeline_node_id' in qparams.keys():
                qparams.pop('pipeline_node_id')

            response = requests.put(
                url,
                data=open(t1, 'rb').read(),
                params=qparams,
                headers=self._client._get_headers(content_type='application/json')
            )

            listOfFileNames.remove('pipeline-model.json')

            # note: the file order is importand, should be OBM model first then KB model
            for fileName, node_id in zip(listOfFileNames, node_ids):
                # Check filename endswith json
                if fileName.endswith('.tar.gz') or fileName.endswith('.zip'):
                    # Extract a single file from zip
                    qparams.update({'content_format': 'pipeline-node'})
                    qparams.update({'pipeline_node_id': node_id})
                    qparams.update({'name': fileName})
                    t2 = zipObj.extract(fileName)
                    response = requests.put(
                        url,
                        data=open(t2, 'rb').read(),
                        params=qparams,
                        headers=self._client._get_headers(content_type='application/octet-stream')
                    )
        return response

    def _download_auto_ai_model_content(self, model_id, content_url, filename):
        import zipfile
        with zipfile.ZipFile(filename, 'w') as zip:
            # writing each file one by one
            pipeline_model_file = 'pipeline-model.json'
            with open(pipeline_model_file, 'wb') as f:
                params = self._client._params()
                params.update({'content_format': 'native'})
                r = requests.get(content_url, params=params,
                                 headers=self._client._get_headers(), stream=True)
                if r.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), r)
                self._logger.info(u'Successfully downloaded artifact pipeline_model.json artifact_url: {}'.format(content_url))
                f.write(r.content)
            f.close()
            zip.write(pipeline_model_file)
            mfilename = 'model_' + model_id + '.pickle.tar.gz'
            with open(mfilename, 'wb') as f:
                params1 = self._client._params()
                params1.update({'content_format': 'pipeline-node'})
                res = requests.get(content_url, params=params1,
                                 headers=self._client._get_headers(), stream=True)
                if res.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), r)
                f.write(res.content)
                self._logger.info(u'Successfully downloaded artifact with artifact_url: {}'.format(content_url))
            f.close()
            zip.write(mfilename)

    def _store_tf_model(self, model, meta_props, feature_names=None, label_column_names=None):
        # Model type is
        import tensorflow as tf
        url = self._client.service_instance._href_definitions.get_published_models_href()
        id_length = 20
        gen_id = uid_generate(id_length)

        tf_meta = None
        options = None
        signature = None
        save_format = None
        include_optimizer = None
        if 'tf_model_params' in meta_props and meta_props[self.ConfigurationMetaNames.TF_MODEL_PARAMS] is not None:
            tf_meta = copy.deepcopy(meta_props[self.ConfigurationMetaNames.TF_MODEL_PARAMS])
            save_format = tf_meta.get('save_format')
            options = tf_meta.get('options')
            signature = tf_meta.get('signature')
            include_optimizer = tf_meta.get('include_optimizer')

        if save_format == "tf" or \
                (save_format is None and "tensorflow.python.keras.engine.training.Model" in str(type(model))):
            temp_dir_name = '{}'.format('pb' + gen_id)
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            import tensorflow as tf
            tf.saved_model.save(model, temp_dir,
                                signatures=signature, options=options)

        elif save_format == "h5" or \
                (save_format is None and "tensorflow.python.keras.engine.sequential.Sequential" in str(type(model))):
            temp_dir_name = '{}'.format('hdfs' + gen_id)
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            model_file = temp_dir + "/sequential_model.h5"
            tf.keras.models.save_model(
                model, model_file, include_optimizer=include_optimizer, save_format='h5',
                signatures=None, options=options)
        elif isinstance(model, str) and os.path.splitext(os.path.basename(model))[-1] == '.h5':
            temp_dir_name = '{}'.format('hdfs' + gen_id)
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                import shutil
                os.makedirs(temp_dir)
                shutil.copy2(model, temp_dir)
        else:

            raise WMLClientError(
                'Saving the tensorflow model requires the model of either tf format or h5 format for Sequential model.'
            )

        path_to_archive = self._model_content_compress_artifact(temp_dir_name,temp_dir)
        payload = copy.deepcopy(meta_props)
        if label_column_names:
            payload['label_column'] = label_column_names[0]

        response = requests.post(
            url,
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers()
        )
        result = self._handle_response(201, u'creating model', response)
        model_uid = self._get_required_element_from_dict(result, 'model_details', ['metadata', 'id'])

        url = self._client.service_instance._href_definitions.get_published_model_href(model_uid) + "/content"
        with open(path_to_archive, 'rb') as f:
            qparams = self._client._params()

            qparams.update({'content_format': 'native'})
            qparams.update({'version': self._client.version_param})
            model_type = meta_props[self.ConfigurationMetaNames.TYPE]
            # update the content path for the Auto-ai model.

            response = requests.put(
                url,
                data=f,
                params=qparams,
                headers=self._client._get_headers(content_type='application/octet-stream')
            )
            if response.status_code != 200 and response.status_code != 201:
                self._delete(model_uid)
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                    self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                self._handle_response(201, u'uploading model content', response, False)
            else:
                self._handle_response(200, u'uploading model content', response, False)

            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                os.remove(path_to_archive)
            return self.get_details(model_uid)

    def _model_content_compress_artifact(self, type_name, compress_artifact):
        tar_filename = '{}_content.tar'.format(type_name)
        gz_filename = '{}.gz'.format(tar_filename)
        CompressionUtil.create_tar(compress_artifact, '.', tar_filename)
        CompressionUtil.compress_file_gzip(tar_filename, gz_filename)
        os.remove(tar_filename)
        return gz_filename

    def _is_h5(self, model_filepath):
        return os.path.splitext(os.path.basename(model_filepath))[-1] == '.h5'

