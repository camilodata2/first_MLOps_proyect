#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
from abc import ABC, abstractmethod
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, Union, Dict
from warnings import warn
from contextlib import redirect_stdout

from pandas import DataFrame
import ibm_watson_machine_learning._wrappers.requests as requests

from ..experiment import AutoAI
from ..utils import is_lale_pipeline
from ..utils.autoai.utils import (prepare_auto_ai_model_to_publish, prepare_auto_ai_model_to_publish_normal_scenario,
                                  remove_file, prepare_auto_ai_model_to_publish_notebook,
                                  prepare_auto_ai_model_to_publish_notebook_normal_scenario,
                                  download_wml_pipeline_details_from_file, get_sw_spec_and_type_based_on_sklearn,
                                  check_if_ts_pipeline_is_winner, convert_dataframe_to_fields_values_payload)
from ..utils.deployment.errors import (WrongDeploymnetType, ModelTypeNotSupported, NotAutoAIExperiment,
                                       DeploymentNotSupported, MissingSpace, ModelPromotionFailed,
                                       ServingNameNotAvailable)
from ..helpers import DataConnection
from ..wml_client_error import ApiRequestFailure
from ..workspace import WorkSpace

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from numpy import ndarray

__all__ = [
    "BaseDeployment"
]


class BaseDeployment(ABC):
    """Base abstract class for Deployment."""

    def __init__(self,
                 deployment_type: str,
                 source_wml_credentials: Union[dict, 'WorkSpace'] = None,
                 source_project_id: str = None,
                 source_space_id: str = None,
                 target_wml_credentials: Union[dict, 'WorkSpace'] = None,
                 target_project_id: str = None,
                 target_space_id: str = None,
                 wml_credentials: Union[dict, 'WorkSpace'] = None,
                 project_id: str = None,
                 space_id: str = None):

        if space_id is None and source_space_id is None and target_space_id is None:
            raise MissingSpace(reason="Any of the [space_id, source_space_id, target_space_id] is not specified.")

        # note: backward compatibility
        if wml_credentials is not None:
            source_wml_credentials = wml_credentials
            warn("\"wml_credentials\" parameter is deprecated, please use \"source_wml_credentials\"")
            print("\"wml_credentials\" parameter is deprecated, please use \"source_wml_credentials\"")

        if project_id is not None:
            source_project_id = project_id
            warn("\"project_id\" parameter is deprecated, please use \"source_project_id\"")
            print("\"project_id\" parameter is deprecated, please use \"source_project_id\"")

        if space_id is not None:
            source_space_id = space_id
            warn("\"space_id\" parameter is deprecated, please use \"source_space_id\"")
            print("\"space_id\" parameter is deprecated, please use \"source_space_id\"")
        # --- end note

        # note: as workspace is not clear enough to understand, there is a possibility to use pure
        # wml credentials with project and space IDs, but in addition we
        # leave a possibility to use a previous WorkSpace implementation, it could be passed as a first argument
        if isinstance(source_wml_credentials, WorkSpace):
            self._source_workspace = source_wml_credentials

        elif isinstance(source_wml_credentials, dict):
            self._source_workspace = WorkSpace(wml_credentials=source_wml_credentials.copy(),
                                               project_id=source_project_id,
                                               space_id=source_space_id)
        else:
            self._source_workspace = None

        if target_wml_credentials is None:
            if isinstance(source_wml_credentials, WorkSpace):
                self._target_workspace = WorkSpace(wml_credentials=source_wml_credentials.wml_credentials,
                                                   space_id=target_space_id if target_space_id is not None else source_wml_credentials.space_id)

            elif isinstance(source_wml_credentials, dict):
                self._target_workspace = WorkSpace(wml_credentials=source_wml_credentials.copy(),
                                                   space_id=target_space_id if target_space_id is not None else source_space_id)
            else:
                self._target_workspace = None

            if target_space_id or target_project_id:
                self._target_workspace.project_id = target_project_id
                self._target_workspace.space_id = target_space_id

        else:
            if isinstance(target_wml_credentials, WorkSpace):
                self._target_workspace = target_wml_credentials

            elif isinstance(target_wml_credentials, dict):
                self._target_workspace = WorkSpace(wml_credentials=target_wml_credentials.copy(),
                                                   project_id=target_project_id,
                                                   space_id=target_space_id)

                # note: only if user provides target WML information
                if self._source_workspace is None:
                    self._source_workspace = copy(self._target_workspace)

            else:
                self._target_workspace = None
        # --- end note

        if self._source_workspace is None and self._target_workspace is None:
            raise TypeError(f"{self.__class__.__name__} is missing one of the parameters: "
                            f"['source_wml_credentials', 'target_wml_credentials']")

        self.name = None
        self.id = None
        if deployment_type == 'online':
            self.scoring_url = None

        if deployment_type == 'batch':
            self._obm = False

    def __repr__(self):
        return f"name: {self.name}, id: {self.id}"

    def __str__(self):
        return f"name: {self.name}, id: {self.id}"

    @abstractmethod
    def create(self, **kwargs):
        """Create deployment from a model.

        :param model: AutoAI model name
        :type model: str

        :param deployment_name: name of the deployment
        :type deployment_name: str

        :param training_data: training data for the model
        :type training_data: pandas.DataFrame or numpy.ndarray, optional

        :param training_target: target/label data for the model
        :type training_target: pandas.DataFrame or numpy.ndarray, optional

        :param metadata: model meta properties
        :type metadata: dict, optional

        :param experiment_run_id: ID of a training/experiment (only applicable for AutoAI deployments)
        :type experiment_run_id: str, optional
        """

        if kwargs.get('serving_name'):
            status_code, response = self._target_workspace.wml_client.deployments._get_serving_name_info(kwargs.get('serving_name'))

            if status_code == 409:
                raise ServingNameNotAvailable(response['errors'][0])

        # note: This section is only for deployments with specified experiment_id
        if kwargs['experiment_run_id'] is not None:
            run_params = self._source_workspace.wml_client.training.get_details(
                training_uid=kwargs['experiment_run_id'], _internal=True)
            wml_pipeline_details = self._source_workspace.wml_client.pipelines.get_details(
                run_params['entity']['pipeline']['id'])

            if not ('autoai' in str(wml_pipeline_details) or 'auto_ai' in str(wml_pipeline_details)):
                raise NotAutoAIExperiment(
                    kwargs['experiment_run_id'], reason="Currently WebService class supports only AutoAI models.")

            if 'auto_ai.obm' in str(wml_pipeline_details):
                if hasattr(self, 'scoring_url'):
                    raise DeploymentNotSupported('WebService',
                                                 reason="AutoAI with DataJoin is not supported for WebService "
                                                        "deployment. Please use Batch deployment instead.")
                self._obm = True

            else:
                self._obm = False

            print("Preparing an AutoAI Deployment...")
            # TODO: remove part with model object depployment
            if not isinstance(kwargs['model'], str):
                warning_msg = ("Depreciation Warning: Passing an object will no longer be supported. "
                               "Please specify the AutoAI model name to deploy.")
                print(warning_msg)
                warn(warning_msg)

            if is_lale_pipeline(kwargs['model']):
                model = kwargs['model'].export_to_sklearn_pipeline()
            else:
                model = kwargs['model']

            # note: check if model is of lale type, if yes, convert it back to scikit
            if not isinstance(kwargs['model'], str):
                model_props = {
                    self._target_workspace.wml_client.repository.ModelMetaNames.NAME: f"{kwargs['deployment_name']} Model",
                    self._target_workspace.wml_client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                    self._target_workspace.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                        self._target_workspace.wml_client.software_specifications.get_uid_by_name("hybrid_0.1")
                }

                # fill additional needed OBM metadata
                if self._obm:
                    model_props[self._target_workspace.wml_client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES] = (
                        run_params['entity']['training_data_references']
                    )
                # --- end note

                schema, artifact_name = prepare_auto_ai_model_to_publish(
                    pipeline_model=model,
                    run_params=run_params,
                    run_id=kwargs['experiment_run_id'],
                    wml_client=self._source_workspace.wml_client)

                model_props[self._target_workspace.wml_client.repository.ModelMetaNames.INPUT_DATA_SCHEMA] = (
                    schema['schemas'] if self._obm else [schema])

            else:
                # raise an error when TS pipeline is discarded one
                check_if_ts_pipeline_is_winner(details=run_params, model_name=model)

                # Note: We need to fetch credentials when 'container' is the type
                if run_params['entity']['results_reference']['type'] == 'container':
                    data_connection = DataConnection._from_dict(_dict=run_params['entity']['results_reference'])
                    data_connection._wml_client = self._source_workspace.wml_client
                else:
                    data_connection = None
                # --- end note

                try:
                    auto_pipelines_parameters = wml_pipeline_details.get('entity', {}).get('document', {}).get('pipelines', [])[0].get('nodes', [])[0].get('parameters')
                except:
                    auto_pipelines_parameters = None

                artifact_name, model_props = prepare_auto_ai_model_to_publish_normal_scenario(
                    pipeline_model=model,
                    run_params=run_params,
                    run_id=kwargs['experiment_run_id'],
                    wml_client=self._source_workspace.wml_client,
                    space_id=self._target_workspace.space_id,
                    result_reference=data_connection,
                    auto_pipelines_parameters=auto_pipelines_parameters
                )

            deployment_details = self._deploy(
                pipeline_model=artifact_name,
                deployment_name=kwargs['deployment_name'],
                serving_name=kwargs.get('serving_name'),
                meta_props=model_props,
                hardware_spec=kwargs.get('hardware_spec')
            )

            remove_file(filename=artifact_name)

            self.name = kwargs['deployment_name']
            self.id = deployment_details.get('metadata').get('id')
            if kwargs['deployment_type'] == 'online':
                self.scoring_url = self._target_workspace.wml_client.deployments.get_scoring_href(deployment_details)
        # --- end note

        # note: This section is for deployments from auto-gen notebook with COS connection / WSD
        else:
            # note: only if we have COS connections from the notebook or for WSD
            if kwargs.get('metadata') is not None:
                print("Preparing an AutoAI Deployment...")

                # note: WMLS
                if self._source_workspace is not None and self._source_workspace.WMLS:
                    optimizer = AutoAI(
                        self._source_workspace.wml_credentials,
                        space_id=self._source_workspace.space_id
                    ).runs.get_optimizer(metadata=kwargs['metadata'])

                # note: CP4D 3.5
                elif self._source_workspace is not None and self._source_workspace.wml_client.ICP:
                    optimizer = AutoAI(self._source_workspace).runs.get_optimizer(metadata=kwargs['metadata'])

                # note: CLOUD
                else:
                    optimizer = AutoAI().runs.get_optimizer(metadata=kwargs['metadata'],
                                                            wml_client=self._source_workspace.wml_client)

                # note: check for obm step in pipeline details
                if hasattr(optimizer, '_result_client'):
                    wml_pipeline_details = download_wml_pipeline_details_from_file(optimizer._result_client)

                else:
                    run_id = optimizer._engine._current_run_id
                    pipeline_id = optimizer._workspace.wml_client.training.get_details(
                        run_id, _internal=True)['entity']['pipeline']['id']
                    wml_pipeline_details = optimizer._workspace.wml_client.pipelines.get_details(pipeline_id)

                if 'auto_ai.obm' in str(wml_pipeline_details):
                    if hasattr(self, 'scoring_url'):
                        raise DeploymentNotSupported('WebService',
                                                     reason="AutoAI with DataJoin is not supported for WebService "
                                                            "deployment. Please use Batch deployment instead.")
                    self._obm = True

                else:
                    self._obm = False
                # --- end note

                # note: only when user did not pass WMLS credentials during Service initialization
                if self._source_workspace is None:
                    self._source_workspace = copy(optimizer._workspace)

                if self._target_workspace is None:
                    self._target_workspace = copy(optimizer._workspace)
                # --- end note

                # TODO: remove part with model object depployment
                if not isinstance(kwargs['model'], str):
                    warning_msg = ("Depreciation Warning: Passing an object will no longer be supported. "
                                   "Please specify the AutoAI model name to deploy.")
                    print(warning_msg)
                    warn(warning_msg)

                if is_lale_pipeline(kwargs['model']):
                    model = kwargs['model'].export_to_sklearn_pipeline()
                else:
                    model = kwargs['model']

                if not isinstance(kwargs['model'], str):
                    model_props = {
                        self._target_workspace.wml_client.repository.ModelMetaNames.NAME: f"{kwargs['deployment_name']} Model",
                        self._target_workspace.wml_client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                        self._target_workspace.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                            self._target_workspace.wml_client.software_specifications.get_uid_by_name("hybrid_0.1")
                    }

                    # note: CLOUD
                    if not self._target_workspace.wml_client.ICP:
                        schema, artifact_name = prepare_auto_ai_model_to_publish_notebook(
                            pipeline_model=model,
                            result_connection=optimizer._result_client[0],
                            cos_client=optimizer._result_client[1],
                            obm=True if self._obm else False
                        )

                        model_props[self._target_workspace.wml_client.repository.ModelMetaNames.INPUT_DATA_SCHEMA] = (
                            schema['schemas'] if self._obm else [schema])

                        deployment_details = self._deploy(
                            pipeline_model=artifact_name,
                            deployment_name=kwargs['deployment_name'],
                            meta_props=model_props,
                            serving_name=kwargs.get('serving_name'),
                            result_client=optimizer._result_client
                        )

                    # note: WSD part
                    else:
                        training_result_reference = kwargs['metadata'].get('training_result_reference')
                        run_id = training_result_reference.location.path.split('/')[-3]
                        run_params = self._source_workspace.wml_client.training.get_details(training_uid=run_id, _internal=True)

                        schema, artifact_name = prepare_auto_ai_model_to_publish(
                            pipeline_model=model,
                            run_params=run_params,
                            run_id=kwargs['experiment_run_id'],
                            wml_client=self._source_workspace.wml_client)

                        model_props[self._target_workspace.wml_client.repository.ModelMetaNames.INPUT_DATA_SCHEMA] = (
                            schema['schemas'] if self._obm else [schema])

                        deployment_details = self._deploy(
                            pipeline_model=artifact_name,
                            deployment_name=kwargs['deployment_name'],
                            serving_name=kwargs.get('serving_name'),
                            meta_props=model_props
                        )
                    # --- end note

                else:
                    if not self._target_workspace.wml_client.ICP:

                        optimizer_2 = AutoAI(self._source_workspace).runs.get_optimizer(metadata=kwargs['metadata'])
                        run_details = optimizer_2._workspace.wml_client.training.get_details(
                            optimizer_2.get_params()['run_id'], _internal=True
                        )

                        artifact_name, model_props = prepare_auto_ai_model_to_publish_notebook_normal_scenario(
                            pipeline_model=model,
                            result_connection=optimizer._result_client[0],
                            cos_client=optimizer._result_client[1],
                            run_params=run_details,
                            space_id=self._target_workspace.space_id,
                            auto_pipelines_parameters=optimizer.get_params()
                        )

                        deployment_details = self._deploy(
                            pipeline_model=artifact_name,
                            deployment_name=kwargs['deployment_name'],
                            meta_props=model_props,
                            serving_name=kwargs.get('serving_name'),
                            result_client=optimizer._result_client
                        )

                    # note: WSD / CP4D 3.5 part
                    else:
                        training_result_reference = kwargs['metadata'].get('training_result_reference')
                        run_id = training_result_reference.location.path.split('/')[-3]
                        run_params = self._source_workspace.wml_client.training.get_details(training_uid=run_id, _internal=True)

                        artifact_name, model_props = prepare_auto_ai_model_to_publish_normal_scenario(
                            pipeline_model=model,
                            run_params=run_params,
                            run_id=run_id,
                            wml_client=self._source_workspace.wml_client,
                            space_id=self._target_workspace.space_id,
                            auto_pipelines_parameters=optimizer.get_params()
                        )

                        deployment_details = self._deploy(
                            pipeline_model=artifact_name,
                            deployment_name=kwargs['deployment_name'],
                            serving_name=kwargs.get('serving_name'),
                            meta_props=model_props
                        )
                    # --- end note

                remove_file(filename=artifact_name)

                self.name = kwargs['deployment_name']
                self.id = deployment_details.get('metadata').get('id')
                if kwargs['deployment_type'] == 'online':
                    self.scoring_url = self._target_workspace.wml_client.deployments.get_scoring_href(
                        deployment_details
                    )
            # --- end note

            else:
                raise ModelTypeNotSupported(type(kwargs['model']),
                                            reason="Currently WebService class supports only AutoAI models.")

    @abstractmethod
    def get_params(self):
        """Get deployment parameters."""
        return self._target_workspace.wml_client.deployments.get_details(self.id)

    @abstractmethod
    def score(self, **kwargs):
        """Scoring on WML. Payload is passed to the WML scoring endpoint where model have been deployed.

        :param payload: data to test the model
        :type payload: pandas.DataFrame
        """

        if isinstance(kwargs['payload'], DataFrame):
            payload = convert_dataframe_to_fields_values_payload(kwargs['payload'])
            input_data = [payload]

        elif isinstance(kwargs['payload'], dict):
            observations_df = kwargs['payload'].get('observations', DataFrame())
            supporting_features_df = kwargs['payload'].get('supporting_features')

            if isinstance(observations_df, DataFrame) and \
                    (isinstance(supporting_features_df, DataFrame) or supporting_features_df is None):
                observations_payload = convert_dataframe_to_fields_values_payload(observations_df)
                observations_payload['id'] = 'observations'
                input_data = [observations_payload]

                if supporting_features_df is not None:
                    supporting_features_payload = convert_dataframe_to_fields_values_payload(supporting_features_df)
                    supporting_features_payload['id'] = 'supporting_features'
                    input_data.append(supporting_features_payload)
            else:
                raise TypeError('Missing data observations in payload or observations'
                                 'or supporting_features are not pandas DataFrames.')

        else:
            raise TypeError(f"Scoring payload has invalid type: {type(kwargs['payload'])}. "
                            f"Supported types are: pandas.DataFrame and dictionary.")

        transaction_id = kwargs.get('transaction_id')

        scoring_payload = {self._target_workspace.wml_client.deployments.ScoringMetaNames.INPUT_DATA: input_data}

        score = self._target_workspace.wml_client.deployments.score(self.id, scoring_payload, transaction_id=transaction_id)

        return score

    @abstractmethod
    def delete(self, **kwargs):
        """Delete deployment on WML.

        :param deployment_id: ID of the deployment to delete, if empty, current deployment will be deleted
        :type deployment_id: str, optional

        :param deployment_type: type of the deployment: [online, batch]
        :type deployment_type: str
        """
        if kwargs['deployment_id'] is None:
            self._target_workspace.wml_client.deployments.delete(self.id)
            self.name = None
            self.scoring_url = None
            self.id = None

        else:
            deployment_details = self._target_workspace.wml_client.deployments.get_details(
                deployment_uid=kwargs['deployment_id'])
            if deployment_details.get('entity', {}).get(kwargs['deployment_type']) is not None:
                self._target_workspace.wml_client.deployments.delete(kwargs['deployment_id'])

            else:
                raise WrongDeploymnetType(
                    f"{kwargs['deployment_type']}",
                    reason=f"Deployment with ID: {kwargs['deployment_id']} is not of \"{kwargs['deployment_type']}\" type!")

    @abstractmethod
    def list(self, **kwargs):
        """List WML deployments.

        :param limit: set the limit of how many deployments to list,
            default is `None` (all deployments should be fetched)
        :type limit: int, optional

        :param deployment_type: type of the deployment: [online, batch]
        :type deployment_type: str
        """
        deployments = self._target_workspace.wml_client.deployments.get_details(limit=kwargs['limit'])
        columns = [
            'created_at',
            'modified_at',
            'id',
            'name',
            'status'
        ]

        data = [
            [deployment.get('metadata')['created_at'],
             deployment.get('metadata')['modified_at'],
             deployment.get('metadata')['id'],
             deployment.get('metadata')['name'],
             deployment.get('entity')['status']['state'],
             ] for deployment in deployments.get('resources', []) if
            isinstance(deployment.get('entity', {}).get(kwargs['deployment_type']), dict)
        ]

        deployments = DataFrame(data=data, columns=columns).sort_values(by=['created_at'], ascending=False)
        return deployments.head(n=kwargs['limit'])

    @abstractmethod
    def get(self, **kwargs):
        """Get WML deployment.

        :param deployment_id: ID of the deployment to work with
        :type deployment_id: str

        :param deployment_type: type of the deployment: [online, batch]
        :type deployment_type: str
        """
        deployment_details = self._target_workspace.wml_client.deployments.get_details(
            deployment_uid=kwargs['deployment_id'])
        if deployment_details.get('entity', {}).get(kwargs['deployment_type']) is not None:
            self.name = deployment_details.get('metadata').get('name')
            self.id = deployment_details.get('metadata').get('id')
            if kwargs['deployment_type'] == 'online':
                self.scoring_url = self._target_workspace.wml_client.deployments.get_scoring_href(deployment_details)

        else:
            raise WrongDeploymnetType(
                f"{kwargs['deployment_type']}",
                reason=f"Deployment with ID: {kwargs['deployment_id']} is not of \"{kwargs['deployment_type']}\" type!")

    @abstractmethod
    def _deploy(self, **kwargs):
        """Protected method to create a deployment."""
        pass

    def _publish_model(self,
                       pipeline_model: Union['Pipeline', str],
                       meta_props: Dict) -> str:
        """Publish model into WML.

        :param pipeline_model: model of the pipeline to publish
        :type pipeline_model: Pipeline or str

        :param meta_props: model meta properties
        :type meta_props: dict

        :return: asset id
        :rtype: str
        """

        # Note: publish model to project and then promote to space
        if self._source_workspace.project_id and self._source_workspace.project_id in str(meta_props):
            published_model_details = self._source_workspace.wml_client.repository.store_model(
                model=pipeline_model,
                meta_props=meta_props)

            project_asset_id = self._source_workspace.wml_client.repository.get_model_id(published_model_details)

            asset_id = self._source_workspace.wml_client.spaces.promote(asset_id=project_asset_id,
                                                                        source_project_id=self._source_workspace.project_id,
                                                                        target_space_id=self._target_workspace.space_id)

        else:
            published_model_details = self._target_workspace.wml_client.repository.store_model(
                model=pipeline_model,
                meta_props=meta_props)

            asset_id = self._target_workspace.wml_client.repository.get_model_id(published_model_details)

        print(f"Published model uid: {asset_id}")
        return asset_id

    @staticmethod
    def _project_to_space_to_project(method):
        @wraps(method)
        def _method(self, *method_args, **method_kwargs):
            with redirect_stdout(open(os.devnull, "w")):
                if self._target_workspace.space_id is not None:
                    self._target_workspace.wml_client.set.default_space(
                        self._target_workspace.space_id) if self._target_workspace.wml_client.ICP else None

            try:
                return method(self, *method_args, **method_kwargs)

            finally:
                if not self._target_workspace.WMLS and self._target_workspace.project_id:
                    with redirect_stdout(open(os.devnull, "w")):
                        if self._target_workspace.project_id is not None:
                            self._target_workspace.wml_client.set.default_project(
                                self._target_workspace.project_id) if self._target_workspace.wml_client.ICP else None

        return _method
