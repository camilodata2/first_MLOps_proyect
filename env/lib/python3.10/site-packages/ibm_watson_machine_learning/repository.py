#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.utils import get_url, INSTANCE_DETAILS_TYPE, is_python_2, inherited_docstring
from ibm_watson_machine_learning.metanames import ModelMetaNames, ExperimentMetaNames, FunctionMetaNames, PipelineMetanames, SpacesMetaNames, MemberMetaNames, FunctionNewMetaNames
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.models import Models
from ibm_watson_machine_learning.experiments import Experiments
from ibm_watson_machine_learning.functions import Functions
from ibm_watson_machine_learning.pipelines import Pipelines
from ibm_watson_machine_learning.spaces import Spaces
from multiprocessing import Pool
from ibm_watson_machine_learning.libs.repo.mlrepositoryclient import MLRepositoryClient
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES
import os
from warnings import warn

_DEFAULT_LIST_LENGTH = 50


class Repository(WMLResource):
    """Store and manage models, functions, spaces, pipelines and experiments
    using Watson Machine Learning Repository.
    
    To view ModelMetaNames, use:

    .. code-block:: python

        client.repository.ModelMetaNames.show()

    To view ExperimentMetaNames, use:

    .. code-block:: python

        client.repository.ExperimentMetaNames.show()

    To view FunctionMetaNames, use:

    .. code-block:: python

        client.repository.FunctionMetaNames.show()

    To view PipelineMetaNames, use:

    .. code-block:: python

        client.repository.PipelineMetaNames.show()

    """

    cloud_platform_spaces = False
    icp_platform_spaces = False

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not client.WSD and not client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            Repository._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Repository._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self._ICP = client.ICP
        self._WSD = client.WSD
        self._ml_repository_client = None
        Repository.cloud_platform_spaces = client.CLOUD_PLATFORM_SPACES
        Repository.icp_platform_spaces = client.ICP_PLATFORM_SPACES

        self.ExperimentMetaNames = ExperimentMetaNames()
        if not client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            self.FunctionMetaNames = FunctionMetaNames()
        else:
            self.FunctionMetaNames = FunctionNewMetaNames()
        self.PipelineMetaNames = PipelineMetanames()
        self.SpacesMetaNames = SpacesMetaNames()
        self.ModelMetaNames = ModelMetaNames()
        self.MemberMetaNames = MemberMetaNames()

        self._refresh_repo_client() # regular token is initialized in service_instance

    def _refresh_repo_client(self):
        # If apiKey is passed in credentials then refresh repoclient with IAM token else MLToken
        self._ml_repository_client = MLRepositoryClient(self._wml_credentials[u'url'])
        if self._client.proceed is True:
            if self._client.service_instance._is_iam() is not None:
                self._ml_repository_client.authorize_with_token(self._client.wml_token)
                self._ml_repository_client._add_header('X-WML-User-Client', 'PythonClient')
                if self._client.project_id is not None:
                    self._ml_repository_client._add_header('X-Watson-Project-ID', self._client.project_id)
            else:
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    platform_spaces = True
                else:
                    platform_spaces = False

                self._ml_repository_client.authorize_with_iamtoken(self._client.wml_token,
                                                                   self._wml_credentials[u'instance_id'],
                                                                   platform_spaces)
                self._ml_repository_client._add_header('X-WML-User-Client', 'PythonClient')
                # Cloud Convergence
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                   self._ml_repository_client._add_header('ML-Instance-ID', self._wml_credentials[u'instance_id'])
                if self._client.project_id is not None:
                    self._ml_repository_client._add_header('X-Watson-Project-ID', self._client.project_id)
        else:
            if self._client._is_IAM():
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    platform_spaces = True
                else:
                    platform_spaces = False

                self._ml_repository_client.authorize_with_iamtoken(self._client.wml_token,
                                                                   self._wml_credentials[u'instance_id'],
                                                                   platform_spaces)
                self._ml_repository_client._add_header('X-WML-User-Client', 'PythonClient')
                # Cloud Convergence
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                   self._ml_repository_client._add_header('ML-Instance-ID', self._wml_credentials[u'instance_id'])
                if self._client.project_id is not None:
                    self._ml_repository_client._add_header('X-Watson-Project-ID', self._client.project_id)
            else:
                if self._ICP:
                    self._repotoken = self._client._get_icptoken()
                    self._ml_repository_token = self._repotoken.replace('Bearer', '')
                    self._ml_repository_client.authorize_with_token(self._ml_repository_token)
                else:
                    if not self._client.WSD:
                        self._ml_repository_client.authorize(self._wml_credentials[u'username'], self._wml_credentials[u'password'])
                        self._ml_repository_client._add_header('X-WML-User-Client', 'PythonClient')
                        if self._client.project_id is not None:
                           self._ml_repository_client._add_header('X-Watson-Project-ID', self._client.project_id)

    @inherited_docstring(Experiments.store, {'experiments.get_href': 'repository.get_experiment_href'})
    def store_experiment(self, meta_props):
        if self._client.WSD:
            raise WMLClientError(u'Experiment APIs are not supported in Watson Studio Desktop.')

        return self._client.experiments.store(meta_props)

    @inherited_docstring(Spaces.store)
    def store_space(self, meta_props):
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        return self._client.spaces.store(meta_props)

    @inherited_docstring(Spaces.create_member)
    def create_member(self, space_uid, meta_props):
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        return self._client.spaces.create_member(space_uid, meta_props)

    @staticmethod
    def _meta_props_to_repository_v3_style(meta_props):
        if is_python_2():
            new_meta_props = meta_props.copy()

            for key in new_meta_props:
                if type(new_meta_props[key]) is unicode:
                    new_meta_props[key] = str(new_meta_props[key])

            return new_meta_props
        else:
            return meta_props

    @inherited_docstring(Pipelines.store)
    def store_pipeline(self, meta_props):
        return self._client.pipelines.store(meta_props)

    @inherited_docstring(Models.store, {'store()': 'store_model()'})
    def store_model(self, model, meta_props=None, training_data=None, training_target=None, pipeline=None,
                    feature_names=None, label_column_names=None,subtrainingId=None, round_number=None,
                    experiment_metadata=None, training_id=None):
        return self._client._models.store(model, meta_props=meta_props, training_data=training_data,
                                          training_target=training_target, pipeline=pipeline,
                                          feature_names=feature_names, label_column_names=label_column_names,
                                          subtrainingId=subtrainingId, round_number=round_number,
                                          experiment_metadata=experiment_metadata, training_id=training_id)

    def clone(self, artifact_id, space_id=None, action="copy", rev_id=None):
        # """ it is not supported in v4ga
        #     Create a new resource(models, runtimes, libraries, experiments, functions, pipelines) identical with the model either in the same space or in a new space. All dependent assets will be cloned too.
        #
        #     **Parameters**
        #
        #     .. important::
        #         #. **model_id**:  Guid of the artifact to be cloned:\n
        #
        #            **type**: str\n
        #
        #         #. **space_id**: Guid of the space to which the model needs to be cloned. (optional)
        #
        #            **type**: str\n
        #
        #         #. **action**: Action specifying "copy" or "move". (optional)
        #
        #            **type**: str\n
        #
        #         #. **rev_id**: Revision ID of the artifact. (optional)
        #
        #            **type**: str\n
        #
        #     **Output**
        #
        #     .. important::
        #
        #             **returns**: Metadata of the model cloned.\n
        #             **return type**: dict\n
        #
        #     **Example**
        #
        #      >>> client.repository.clone(artifact_id=artifact_id,space_id=space_uid,action="copy")
        #
        #     .. note::
        #         * If revision id is not specified, all revisions of the artifact are cloned\n
        #
        #         * Default value of the parameter action is copy\n
        #
        #         * Space guid is mandatory for move action\n
        #
        #     """
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError('Cloning is not supported.')

        Models._validate_type(artifact_id, 'artifact_id', str, True)
        clone_meta = {}
        if space_id is not None:
            clone_meta["space"] = {"href": API_VERSION + SPACES + "/" + space_id}
        if action is not None:
            clone_meta["action"] = action
        if rev_id is not None:
            clone_meta["rev"] = rev_id
        res = self._check_artifact_type(artifact_id)

        url = ""
        type = ""
        if res['model'] is True:
            url = self._client.service_instance._href_definitions.get_published_model_href(artifact_id)
            type = "model"
        elif res['library'] is True:
            url = self._client.service_instance._href_definitions.get_custom_library_href(artifact_id)
            type = "library"
        elif res['runtime'] is True:
            url = self._client.service_instance._href_definitions.get_runtime_href(artifact_id)
            type = "runtime"
        elif res['function'] is True:
            url = self._client.service_instance._href_definitions.get_function_href(artifact_id)
            type = "function"
        elif res['pipeline'] is True:
            url = self._client.service_instance._href_definitions.get_pipeline_href(artifact_id)
            type = "pipeline"
        elif res['experiment'] is True:
            url = self._client.service_instance._href_definitions.get_experiment_href(artifact_id)
            type = "experiment"


        if type == "":
            raise WMLClientError('Unsupported artifact type. Supported artifact types are models, libraries, runtimes, experiments, pipelines and functions')

        response_post = requests.post(url, json=clone_meta,
                                      headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=200, operationName=u'cloning '+ type,
                                        response=response_post)

        return details

    @inherited_docstring(Functions.store)
    def store_function(self, function, meta_props):
        return self._client._functions.store(function, meta_props)

    @inherited_docstring(Models.create_revision)
    def create_model_revision(self, model_uid):
        return self._client._models.create_revision(model_uid=model_uid)

    @inherited_docstring(Pipelines.create_revision)
    def create_pipeline_revision(self, pipeline_uid):
        return self._client.pipelines.create_revision(pipeline_uid=pipeline_uid)

    @inherited_docstring(Functions.create_revision)
    def create_function_revision(self, function_uid):
        return self._client._functions.create_revision(function_uid=function_uid)

    @inherited_docstring(Experiments.create_revision, {'experiment_id': 'experiment_uid'})
    def create_experiment_revision(self, experiment_uid):
        return self._client.experiments.create_revision(experiment_id=experiment_uid)

    @inherited_docstring(Models.update, {'meta_props': 'updated_meta_props'})
    def update_model(self, model_uid, updated_meta_props=None, update_model=None):
        return self._client._models.update(model_uid, updated_meta_props, update_model)

    @inherited_docstring(Experiments.update)
    def update_experiment(self, experiment_uid, changes):
        if self._client.WSD:
            raise WMLClientError('Experiments APIs are not supported in IBM Watson Studio Desktop.')

        return self._client.experiments.update(experiment_uid, changes)

    @inherited_docstring(Functions.update)
    def update_function(self, function_uid, changes, update_function=None):
        return self._client._functions.update(function_uid, changes, update_function)

    @inherited_docstring(Pipelines.update)
    def update_pipeline(self, pipeline_uid, changes):
        return self._client.pipelines.update(pipeline_uid, changes)

    @inherited_docstring(Spaces.update)
    def update_space(self, space_uid, changes):
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        return self._client.spaces.update(space_uid, changes)

    @inherited_docstring(Models.load)
    def load(self, artifact_uid):
        return self._client._models.load(artifact_uid)

    def download(self, artifact_uid, filename='downloaded_artifact.tar.gz', rev_uid=None, format=None):
        """Downloads configuration file for artifact with specified uid.

        :param artifact_uid: Unique Id of model, function, runtime or library
        :type artifact_uid: str
        :param filename: name of the file to which the artifact content has to be downloaded
        :type filename: str, optional

        :return: path to the downloaded artifact content
        :rtype: str

        **Examples**

        .. code-block:: python

            client.repository.download(model_uid, 'my_model.tar.gz')
            client.repository.download(model_uid, 'my_model.json') # if original model was saved as json, works only for xgboost 1.3
        """
        self._validate_type(artifact_uid, 'artifact_uid', str, True)
        self._validate_type(filename, 'filename', str, True)

        res = self._check_artifact_type(artifact_uid)

        if res['model'] is True:
            return self._client._models.download(artifact_uid, filename, rev_uid,format)
        elif res['function'] is True:
            return self._client._functions.download(artifact_uid, filename, rev_uid)
        elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_35 and not self._client.ICP_40  \
                and not self._client.ICP_45 and not self._client.ICP_46 and not self._client.ICP_47 and res['library'] is True:
            return self._client.runtimes.download_library(artifact_uid, filename)
        elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_35 and not self._client.ICP_40 \
                and not self._client.ICP_45 and not self._client.ICP_46 and not self._client.ICP_47 and res['runtime'] is True:
            return self._client.runtimes.download_configuration(artifact_uid, filename)
        else:
            raise WMLClientError('Unexpected type of artifact to download or Artifact with artifact_uid: \'{}\' does not exist.'.format(artifact_uid) )

    def delete(self, artifact_uid):
        """Delete model, experiment, pipeline, space, runtime, library or function from repository.

        :param artifact_uid: Unique id of stored model, experiment, function, pipeline, space, library or runtime
        :type artifact_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.repository.delete(artifact_uid)
        """
        Repository._validate_type(artifact_uid, u'artifact_uid', str, True)
        if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and self._if_deployment_exist_for_asset(artifact_uid):
            raise WMLClientError(
                u'Cannot delete artifact that has existing deployments. Please delete all associated deployments and try again')
        params = self._client._params()
        if Repository.cloud_platform_spaces or self._client.ICP_PLATFORM_SPACES:
            # ideally purge_on_delete=true query param has to be provided for deletion of cams assets
            # This doesn't seem to be done for CP4D 3.0.1 and before. We should do this for CP4D 3.5
            params.update({'purge_on_delete': 'true'})

        response = requests.delete(self._client.service_instance._href_definitions.get_asset_href(artifact_uid),
                                       params=params,
                                       headers=self._client._get_headers())

        if response.status_code == 200 or response.status_code == 204:
            if response.status_code == 200:
                response = self._handle_response(200, u'delete assets', response)
                return response
            else:
                response = self._handle_response(204, u'delete assets', response)
                return response
        else:
            if Repository.cloud_platform_spaces or self._client.ICP_PLATFORM_SPACES:
                # Since we are using /v2/assets for deletion, don't need all the logic
                # in the following else block. The else block is applicable only for cloud beta
                # and has to be kept till then. For 3.5, move logic to same as cloud convergence
                # for deletion
                if response.status_code == 404:
                    raise WMLClientError(u'Artifact with artifact_uid: \'{}\' does not exist.'.format(artifact_uid))
                else:
                    raise WMLClientError("Deletion error for the given id : ", response.text)
            else:
                artifact_type = self._check_artifact_type(artifact_uid)
                self._logger.debug(u'Attempting deletion of artifact with type: \'{}\''.format(str(artifact_type)))
                if self._client.WSD:
                    if artifact_type[u'model'] is True:
                        return self._client._models.delete(artifact_uid)
                    elif artifact_type[u'pipeline'] is True:
                        return self._client.pipelines.delete(artifact_uid)
                    elif artifact_type[u'function'] is True:
                        return self._client._functions.delete(artifact_uid)
                    else:
                        raise WMLClientError(u'Artifact with artifact_uid: \'{}\' does not exist.'.format(artifact_uid))
                else:
                    if artifact_type[u'model'] is True:
                        return self._client._models.delete(artifact_uid)
                    elif artifact_type[u'experiment'] is True:
                        return self._client.experiments.delete(artifact_uid)
                    elif artifact_type[u'pipeline'] is True:
                        return self._client.pipelines.delete(artifact_uid)
                    elif artifact_type[u'function'] is True:
                        return self._client._functions.delete(artifact_uid)
                    elif artifact_type[u'space'] is True:
                        return self._client.spaces.delete(artifact_uid)
                    elif artifact_type[u'runtime'] is True:
                        return self._client.runtimes.delete(artifact_uid)
                    elif artifact_type[u'library'] is True:
                        return self._client.runtimes.delete_library(artifact_uid)
                    else:
                        raise WMLClientError(u'Artifact with artifact_uid: \'{}\' does not exist.'.format(artifact_uid))

    def get_details(self, artifact_uid=None, spec_state=None):
        """Get metadata of stored artifacts. If `artifact_uid` is not specified returns all models, experiments,
        functions, pipelines, spaces, libraries and runtimes metadata.

        :param artifact_uid: Unique Id of stored model, experiment, function, pipeline, space, library or runtime
        :type artifact_uid: str, optional

        :param spec_state: software specification state, can be used only when `artifact_uid` is None
        :type spec_state: SpecStates, optional

        :return: stored artifact(s) metadata
        :rtype: dict (if artifact_uid is not None) or {"resources": [dict]} (if artifact_uid is None)

        **Examples**

        .. code-block:: python

            details = client.repository.get_details(artifact_uid)
            details = client.repository.get_details()

        Example of getting all repository assets with deprecated software specifications:

        .. code-block:: python

            from ibm_watson_machine_learning.lifecycle import SpecStates

            details = client.repository.get_details(spec_state=SpecStates.DEPRECATED)
        """
        Repository._validate_type(artifact_uid, u'artifact_uid', str, False)

        if artifact_uid is None and self._client.WSD is None:
            model_details = self._client._models.get_details(spec_state=spec_state)
            experiment_details = self.get_experiment_details() if not spec_state else {'resources': []}
            pipeline_details = self.get_pipeline_details() if not spec_state else {'resources': []}
            function_details = self._client._functions.get_details(spec_state=spec_state)

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                space_details = self._client.spaces.get_details() if not spec_state else {'resources': []}
                library_details = self._client.runtimes.get_library_details() if not spec_state else {'resources': []}
                runtime_details = self._client.runtimes.get_details() if not spec_state else {'resources': []}
                details = {
                    u'models': model_details,
                    u'experiments': experiment_details,
                    u'pipeline': pipeline_details,
                    u'runtimes': runtime_details,
                    u'libraries': library_details,
                    u'spaces': space_details,
                    u'functions': function_details
                }
            else:
                details = {
                    u'models': model_details,
                    u'experiments': experiment_details,
                    u'pipeline': pipeline_details,
                    u'functions': function_details
                }
        else:
            if self._client.WSD and artifact_uid is None:
                raise WMLClientError(
                        u' artifiact_uid is mandatory for get_details() in IBM Watson Studio Desktop.')
            uid_type = self._check_artifact_type(artifact_uid)
            if self._client.WSD:
                if uid_type[u'model'] is True:
                    details = self._client._models.get_details(artifact_uid)
                elif uid_type[u'pipeline'] is True:
                    details = self.get_pipeline_details(artifact_uid)
                elif uid_type[u'function'] is True:
                    details = self._client._functions.get_details(artifact_uid)
                else:
                    raise WMLClientError(
                        u'Getting artifact details failed. Artifact uid: \'{}\' not found.'.format(artifact_uid))
            else:
                if uid_type[u'model'] is True:
                    details = self._client._models.get_details(artifact_uid)
                elif uid_type[u'experiment'] is True:
                    details = self.get_experiment_details(artifact_uid)
                elif uid_type[u'pipeline'] is True:
                    details = self.get_pipeline_details(artifact_uid)
                elif uid_type[u'function'] is True:
                    details = self._client._functions.get_details(artifact_uid)
                elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES and uid_type[u'runtime'] is True:
                    details = self._client.runtimes.get_details(artifact_uid)
                elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES and uid_type[u'library'] is True:
                    details = self._client.runtimes.get_library_details(artifact_uid)
                elif not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES and uid_type[u'space'] is True:
                    details = self._client.spaces.get_details(artifact_uid)
                else:
                    raise WMLClientError(u'Getting artifact details failed. Artifact uid: \'{}\' not found.'.format(artifact_uid))

        return details

    @inherited_docstring(Models.get_details)
    def get_model_details(self, model_uid=None, limit=None, asynchronous=False, get_all=False, spec_state=None):
        return self._client._models.get_details(model_uid, limit,
                                                asynchronous=asynchronous, get_all=get_all, spec_state=spec_state)

    @inherited_docstring(Models.get_revision_details)
    def get_model_revision_details(self, model_uid, rev_uid):
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError('Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data 3.0 and above.')
        return self._client._models.get_revision_details(model_uid, rev_uid)

    @inherited_docstring(Experiments.get_details)
    def get_experiment_details(self, experiment_uid=None, limit=None, asynchronous=False, get_all=False):
        if self._client.WSD:
            raise WMLClientError('Experiment APIs are not supported in IBM Watson Studio Desktop.')

        Repository._validate_type(experiment_uid, u'experiment_uid', str, False)
        Repository._validate_type(limit, u'limit', int, False)
        Repository._validate_type(asynchronous, u'asynchronous', bool, False)
        Repository._validate_type(get_all, u'get_all', bool, False)

        return self._client.experiments.get_details(experiment_uid, limit, asynchronous, get_all)

    @inherited_docstring(Experiments.get_revision_details, {'rev_uid': 'rev_id'})
    def get_experiment_revision_details(self, experiment_uid, rev_id):
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                'Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data 3.0 and above.')

        return self._client.experiments.get_revision_details(experiment_uid, rev_id)

    @inherited_docstring(Functions.get_details)
    def get_function_details(self, function_uid=None, limit=None, asynchronous=False, get_all=False, spec_state=None):
        Repository._validate_type(function_uid, u'function_uid', str, False)
        Repository._validate_type(limit, u'limit', int, False)
        Repository._validate_type(asynchronous, u'asynchronous', bool, False)
        Repository._validate_type(get_all, u'get_all', bool, False)
        Repository._validate_type(spec_state, u'spec_state', object, False)
        return self._client._functions.get_details(function_uid, limit, asynchronous, get_all, spec_state)

    @inherited_docstring(Functions.get_revision_details, {'rev_uid': 'rev_id'})
    def get_function_revision_details(self, function_uid, rev_id):
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError('Not supported in this release')
        return self._client._functions.get_revision_details(function_uid, rev_id)

    @inherited_docstring(Pipelines.get_details)
    def get_pipeline_details(self, pipeline_uid=None, limit=None, asynchronous=False, get_all=False):
        Repository._validate_type(pipeline_uid, u'pipeline_uid', str, False)
        Repository._validate_type(limit, u'limit', int, False)
        Repository._validate_type(asynchronous, u'asynchronous', bool, False)
        Repository._validate_type(get_all, u'get_all', bool, False)
        return self._client.pipelines.get_details(pipeline_uid, limit, asynchronous, get_all)

    @inherited_docstring(Pipelines.get_revision_details, {'rev_uid': 'rev_id'})
    def get_pipeline_revision_details(self, pipeline_uid, rev_id):
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                'Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data 3.0 and above.')
        return self._client.pipelines.get_revision_details(pipeline_uid, rev_id)

    @inherited_docstring(Spaces.get_details)
    def get_space_details(self, space_uid=None, limit=None):
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        Repository._validate_type(space_uid, u'space_uid', str, False)
        Repository._validate_type(limit, u'limit', int, False)
        return self._client.spaces.get_details(space_uid, limit)

    @inherited_docstring(Spaces.get_members_details)
    def get_members_details(self, space_uid, member_id=None, limit=None):
        if self._client.WSD or self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")
        return self._client.spaces.get_members_details(space_uid, member_id, limit)

    @staticmethod
    @inherited_docstring(Models.get_href)
    def get_model_href(model_details):
        return Models.get_href(model_details)

    @staticmethod
    def get_model_uid(model_details):
        """
            This method is deprecated, please use ``get_id()`` instead."
        """
        warn("This method is deprecated, please use get_model_id()")
        print("This method is deprecated, please use get_model_id()")

        return Models.get_id(model_details)

    @staticmethod
    @inherited_docstring(Models.get_id)
    def get_model_id(model_details):
        return Models.get_id(model_details)

    @staticmethod
    @inherited_docstring(Experiments.get_uid, {'experiments.get_details': 'repository.get_experiment_details'})
    def get_experiment_uid(experiment_details):
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Experiment APIs are not supported for Watson Studio Desktop.')
        return Experiments.get_uid(experiment_details)

    @staticmethod
    @inherited_docstring(Experiments.get_id, {'experiments.get_details': 'repository.get_experiment_details'})
    def get_experiment_id(experiment_details):
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Experiment APIs are not supported for Watson Studio Desktop.')
        return Experiments.get_id(experiment_details)

    @staticmethod
    @inherited_docstring(Experiments.get_href, {'experiments.get_details': 'repository.get_experiment_details'})
    def get_experiment_href(experiment_details):
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Experiment APIs are not supported for Watson Studio Desktop.')
        return Experiments.get_href(experiment_details)

    @staticmethod
    @inherited_docstring(Functions.get_id)
    def get_function_id(function_details):
        return Functions.get_id(function_details)

    @staticmethod
    @inherited_docstring(Functions.get_uid)
    def get_function_uid(function_details):
        return Functions.get_uid(function_details)

    @staticmethod
    @inherited_docstring(Pipelines.get_uid)
    def get_pipeline_uid(pipeline_details):
        return Pipelines.get_uid(pipeline_details)

    @staticmethod
    @inherited_docstring(Functions.get_href)
    def get_function_href(function_details):
        return Functions.get_href(function_details)

    @staticmethod
    @inherited_docstring(Pipelines.get_href, {'pipelines.get_details': 'repository.get_pipeline_details'})
    def get_pipeline_href(pipeline_details):
        return Pipelines.get_href(pipeline_details)

    @staticmethod
    @inherited_docstring(Pipelines.get_id)
    def get_pipeline_id(pipeline_details):
        return Pipelines.get_id(pipeline_details)

    @staticmethod
    @inherited_docstring(Spaces.get_uid, {'spaces.get_details': 'repository.get_space_details'})
    def get_space_uid(space_details):
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Spaces APIs are not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return Spaces.get_uid(space_details)

    @staticmethod
    @inherited_docstring(Spaces.get_member_uid, {'spaces.get_member_details': 'repository.get_member_details'})
    def get_member_uid(member_details):
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Spaces APIs are not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return Spaces.get_member_uid(member_details)

    @staticmethod
    @inherited_docstring(Spaces.get_href, {'spaces.get_details': 'repository.get_space_details'})
    def get_space_href(space_details):
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Spaces APIs are not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return Spaces.get_href(space_details)

    @staticmethod
    @inherited_docstring(Spaces.get_member_href, {'spaces.get_member_details': 'repository.get_member_details'})
    def get_member_href(member_details):
        if 'WSD_PLATFORM' in os.environ and os.environ['WSD_PLATFORM'] == 'True':
            raise WMLClientError(u'Spaces APIs are not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return Spaces.get_member_href(member_details)

    def list(self):
        """Print stored models, pipelines, runtimes, libraries, functions, spaces and experiments in a table format.
        If limit is set to None there will be only first 50 records shown.

        **Example**
        
        .. code-block:: python

            client.repository.list()
        """

        from tabulate import tabulate

        headers = self._client._get_headers()
        params = self._client._params()
        params.update({u'limit': 1000})
        #params = {u'limit': 1000} # TODO - should be unlimited, if results not sorted

        pool = Pool(processes=4)
        isIcp = self._ICP
        if self._client.WSD:
            raise WMLClientError(
                u'list() - Listing all artifact is not supported for IBM Watson Studio Desktop. '
                u'Use list method of specific artifact.')


        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            endpoints = {
                u'model': self._client.service_instance._href_definitions.get_published_models_href(),
                u'experiment': self._client.service_instance._href_definitions.get_experiments_href(),
                u'pipeline': self._client.service_instance._href_definitions.get_pipelines_href(),
                u'function': self._client.service_instance._href_definitions.get_functions_href()
            }
        else:
            endpoints = {
                u'model': self._client.service_instance._href_definitions.get_published_models_href(),
                u'experiment': self._client.service_instance._href_definitions.get_experiments_href(),
                u'pipeline': self._client.service_instance._href_definitions.get_pipelines_href(),
                u'function': self._client.service_instance._href_definitions.get_functions_href(),
                u'runtime': self._client.service_instance._href_definitions.get_runtimes_href(),
                u'library': self._client.service_instance._href_definitions.get_custom_libraries_href()
            }

        artifact_get = {}
        for artifact in endpoints:
            if (artifact=="library" or artifact=="runtime" or artifact=="space"):
                params = None
            else:
                params = self._client._params()
            artifact_get[artifact] = pool.apply_async(get_url,
                                                (endpoints[artifact], self._client._get_headers(), params, isIcp))

        # artifact_get = {artifact: pool.apply_async(get_url, (endpoints[artifact], headers, self._client._params(), isIcp)) for
        #           artifact in endpoints if (artifact != "library" or artifact != "runtime" or artifact != "space")}
        # artifact_no_space = {artifact: pool.apply_async(get_url, (endpoints[artifact], headers, None, isIcp)) for artifact
        #                    in endpoints if (artifact == "library" or artifact == "runtime")}
        # artifact_get.update(artifact_no_space)

        resources = {artifact: [] for artifact in endpoints}

        for artifact in endpoints:
            try:
                    response = artifact_get[artifact].get()
                    response_text = self._handle_response(200, u'getting all {}s'.format(artifact), response)
                    resources[artifact] = response_text[u'resources']
            except Exception as e:
                    self._logger.error(e)

        pool.close()

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            sw_spec_info = {s['id']: s
                            for s in self._client.software_specifications.get_details(state_info=True)['resources']}

            def get_spec_info(spec_id, prop):
                if spec_id and spec_id in sw_spec_info:
                    return sw_spec_info[spec_id].get(prop, '')
                else:
                    return ''

            values = []
            for t in endpoints.keys():
                values += [
                    (m['metadata']['id'],
                     m['metadata']['name'],
                     m['metadata']['created_at'],
                     m['entity']['type'] if t == 'model' else '-',
                     t if t != 'function' else m['entity']['type'] + ' function',
                     get_spec_info(m['entity'].get('software_spec', {}).get('id'), 'state'),
                     get_spec_info(m['entity'].get('software_spec', {}).get('id'), 'replacement'))
                    for m in resources[t]]

            values = sorted(sorted(list(set(values)), key=lambda x: x[2], reverse=True), key=lambda x: x[4])

            table = tabulate([['GUID', 'NAME', 'CREATED', 'FRAMEWORK', 'TYPE', 'SPEC_STATE', 'SPEC_REPLACEMENT']]
                             + values[:_DEFAULT_LIST_LENGTH])
        else:
            model_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], m[u'entity'][u'type'], u'model') for m in resources[u'model']]
            experiment_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'],m['metadata']['created_at'], u'-', u'experiment') for m in resources[u'experiment']]
            pipeline_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], u'-', u'pipeline')for m in self._client.pipelines.get_details()[u'resources']]
            function_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], u'-', m[u'entity'][u'type'] + u' function') for m in resources[u'function']]
            runtime_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], u'-', m[u'entity'][u'platform'][u'name'] + u' runtime') for m in resources[u'runtime']]
            library_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], u'-', m[u'entity'][u'platform'][u'name'] + u' library') for m in resources[u'library']]
            values = list(set(model_values + experiment_values + pipeline_values + function_values + runtime_values + library_values))
            values = sorted(sorted(values, key=lambda x: x[2], reverse=True), key=lambda x: x[4])

            table = tabulate([[u'GUID', u'NAME', u'CREATED', u'FRAMEWORK', u'TYPE']] + values[:_DEFAULT_LIST_LENGTH])

        print(table)
        if len(values) > _DEFAULT_LIST_LENGTH:
             print('Note: Only first {} records were displayed. To display more use more specific list functions.'.format(_DEFAULT_LIST_LENGTH))

    @inherited_docstring(Models.list)
    def list_models(self, limit=None, asynchronous=False, get_all=False, return_as_df=True):
        return self._client._models.list(limit=limit, asynchronous=asynchronous, get_all=get_all, return_as_df=return_as_df)

    @inherited_docstring(Experiments.list)
    def list_experiments(self, limit=None, return_as_df=True):
        if self._client.WSD:
            raise WMLClientError(u'Experiment APIs are not supported for Watson Studio Desktop.')
        return self._client.experiments.list(limit=limit,  return_as_df=return_as_df)

    @inherited_docstring(Spaces.list)
    def list_spaces(self, limit=None, return_as_df=True):
        if self._client.WSD:
            raise WMLClientError('list_spaces - Listing spaces is not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return self._client.spaces.list(limit=limit, return_as_df=return_as_df)

    @inherited_docstring(Functions.list)
    def list_functions(self, limit=None, return_as_df=True):
        return self._client._functions.list(limit=limit, return_as_df=return_as_df)

    @inherited_docstring(Pipelines.list)
    def list_pipelines(self, limit=None, return_as_df=True):
        return self._client.pipelines.list(limit=limit, return_as_df=return_as_df)

    @inherited_docstring(Spaces.list_members)
    def list_members(self, space_uid, limit=None, return_as_df=True):
        if self._client.WSD:
            raise WMLClientError('list_members - Listing members is not supported for Watson Studio Desktop.')

        if Repository.cloud_platform_spaces or Repository.icp_platform_spaces:
            raise WMLClientError(u"Not supported in this release. Use methods in 'client.spaces' instead")

        return self._client.spaces.list_members(space_uid=space_uid, limit=limit, return_as_df=return_as_df)

    def _check_artifact_type(self, artifact_uid):
        Repository._validate_type(artifact_uid, u'artifact_uid', str, True)

        def _artifact_exists(response):
            return (response is not None) and (u'status_code' in dir(response)) and (response.status_code == 200)

        pool = Pool(processes=4)
        #headers =

        isIcp=self._ICP

        if self._client.WSD:
            endpoint = self._client.service_instance._href_definitions.get_model_definition_assets_href() + "/" + artifact_uid
            response = requests.get(
                endpoint,
                params=self._client._params()
            )

           # requestsget_url, (endpoint, self._client._get_headers(), self._client._params(), True))
            response_get = _artifact_exists(response)

            artifact_type = artifact_uid.rsplit(".")[0]
            artifact_list = ['wml_model', 'wml_pipeline', 'wml_function']
            artifact_type_exists = {artifact.rsplit('_')[-1]: (response_get and artifact == artifact_type) for artifact in artifact_list}
            return artifact_type_exists

        else:
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                endpoints = {
                    u'model': self._client.service_instance._href_definitions.get_model_last_version_href(artifact_uid),
                    u'pipeline': self._client.service_instance._href_definitions.get_pipeline_href(artifact_uid),
                    u'experiment': self._client.service_instance._href_definitions.get_experiment_href(artifact_uid),
                    u'function': self._client.service_instance._href_definitions.get_function_href(artifact_uid)
                }
            else:
                endpoints = {
                        u'model': self._client.service_instance._href_definitions.get_model_last_version_href(artifact_uid),
                        u'pipeline': self._client.service_instance._href_definitions.get_pipeline_href(artifact_uid),
                        u'experiment': self._client.service_instance._href_definitions.get_experiment_href(artifact_uid),
                        u'function': self._client.service_instance._href_definitions.get_function_href(artifact_uid),
                        u'runtime': self._client.service_instance._href_definitions.get_runtime_href(artifact_uid),
                        u'library': self._client.service_instance._href_definitions.get_custom_library_href(artifact_uid),
                        u'space': self._client.service_instance._href_definitions.get_space_href(artifact_uid)
                    }
            future = {}
            for artifact in endpoints:
                if (artifact=="library" or artifact=="runtime" or artifact=="space"):
                    params = None
                else:
                    params = self._client._params()
                future[artifact] = pool.apply_async(get_url, (endpoints[artifact], self._client._get_headers(), params , isIcp))

            # future_no_space = {artifact: pool.apply_async(get_url, (endpoints[artifact], headers, None, isIcp)) for artifact in endpoints if (artifact=="library" or artifact=="runtime" or artifact=="space")}
            # future.update(future_no_space)
            response_get = {artifact: None for artifact in endpoints}

            for artifact in endpoints:
                    try:
                        response_get[artifact] = future[artifact].get(timeout=180)
                        self._logger.debug(u'Response({})[{}]: {}'.format(endpoints[artifact], response_get[artifact].status_code, response_get[artifact].text))

                    except Exception as e:
                        self._logger.debug(u'Error during checking artifact type: ' + str(e))

            pool.close()
            artifact_type = {artifact: _artifact_exists(response_get[artifact]) for artifact in response_get}

            return artifact_type

    def create_revision(self, artifact_uid):
        """Create revision for passed `artifact_uid`.

        :param artifact_uid: Unique id of stored model, experiment, function or pipelines
        :type artifact_uid: str

        :return: artifact new revision metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            details = client.repository.create_revision(artifact_uid)
        """
        Repository._validate_type(artifact_uid, u'artifact_uid', str, True)

        uid_type = self._check_artifact_type(artifact_uid)
        if uid_type[u'experiment'] is True:
            return self._client.experiments.create_revision(artifact_uid)
        if uid_type[u'pipeline'] is True:
            return self._client.pipelines.create_revision(artifact_uid)
        else:
           raise WMLClientError(u'Getting artifact details failed. Artifact uid: \'{}\' not found.'.format(artifact_uid))

        return details

    def _get_revision_details(self, artifact_uid):
        """Get metadata of stored artifacts revisions.

        :param artifact_uid:  unique id of stored model or experiment or function or pipelines
        :type artifact_uid: str

        :return: stored artifacts metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            details = client.repository.get_revision_details(artifact_uid)

        """
        Repository._validate_type(artifact_uid, u'artifact_uid', str, True)

        uid_type = self._check_artifact_type(artifact_uid)

        if uid_type[u'experiment'] is True:
            details = self._client.experiments.get_revision_details(artifact_uid)
        if uid_type[u'pipeline'] is True:
            details = self._client.pipelines.get_revisions(artifact_uid)

        else:
            raise WMLClientError(u'Getting artifact details failed. Artifact uid: \'{}\' not found.'.format(artifact_uid))
        return details

    @inherited_docstring(Models.list_revisions)
    def list_models_revisions(self, model_uid, limit=None, return_as_df=True):
        return self._client._models.list_revisions(model_uid, limit=limit, return_as_df=return_as_df)

    @inherited_docstring(Pipelines.list_revisions)
    def list_pipelines_revisions(self, pipeline_uid, limit=None, return_as_df=True):
        return self._client.pipelines.list_revisions(pipeline_uid, limit=limit, return_as_df=return_as_df)

    @inherited_docstring(Functions.list_revisions)
    def list_functions_revisions(self, function_uid, limit=None, return_as_df=True):
        return self._client._functions.list_revisions(function_uid, limit=limit, return_as_df=return_as_df)

    @inherited_docstring(Experiments.list_revisions)
    def list_experiments_revisions(self, experiment_uid, limit=None, return_as_df=True):
        return self._client.experiments.list_revisions(experiment_uid, limit=limit, return_as_df=return_as_df)

    @inherited_docstring(Models.promote)
    def promote_model(self, model_id: str, source_project_id: str, target_space_id: str):  # deprecated
        return self._client._models.promote(model_id, source_project_id, target_space_id)


