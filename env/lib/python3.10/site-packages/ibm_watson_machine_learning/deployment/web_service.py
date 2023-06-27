#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import TYPE_CHECKING, Any, Dict, Union, List, Optional

import pandas as pd
from pandas import DataFrame

from .base_deployment import BaseDeployment
from ..wml_client_error import WMLClientError

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from pandas import DataFrame
    from numpy import ndarray
    from ..workspace import WorkSpace

__all__ = [
    "WebService"
]


class WebService(BaseDeployment):
    """An Online Deployment class aka. WebService.
    With this class object you can manage any online (WebService) deployment.

    :param source_wml_credentials: credentials to Watson Machine Learning instance where training was performed
    :type source_wml_credentials: dict

    :param source_project_id: ID of the Watson Studio project where training was performed
    :type source_project_id: str, optional

    :param source_space_id: ID of the Watson Studio Space where training was performed
    :type source_space_id: str, optional

    :param target_wml_credentials: credentials to Watson Machine Learning instance where you want to deploy
    :type target_wml_credentials: dict

    :param target_project_id: ID of the Watson Studio project where you want to deploy
    :type target_project_id: str, optional

    :param target_space_id: ID of the Watson Studio Space where you want to deploy
    :type target_space_id: str, optional
    """

    def __init__(self,
                 source_wml_credentials: Union[dict, 'WorkSpace'] = None,
                 source_project_id: str = None,
                 source_space_id: str = None,
                 target_wml_credentials: Union[dict, 'WorkSpace'] = None,
                 target_project_id: str = None,
                 target_space_id: str = None,
                 wml_credentials: Union[dict, 'WorkSpace'] = None,
                 project_id: str = None,
                 space_id: str = None):

        super().__init__(
            deployment_type='online',
            source_wml_credentials=source_wml_credentials,
            source_project_id=source_project_id,
            source_space_id=source_space_id,
            target_wml_credentials=target_wml_credentials,
            target_project_id=target_project_id,
            target_space_id=target_space_id,
            wml_credentials=wml_credentials,
            project_id=project_id,
            space_id=space_id
        )

        self.name = None
        self.scoring_url = None
        self.id = None
        self.asset_id = None

    def __repr__(self):
        return f"name: {self.name}, id: {self.id}, scoring_url: {self.scoring_url}, asset_id: {self.asset_id}"

    def __str__(self):
        return f"name: {self.name}, id: {self.id}, scoring_url: {self.scoring_url}, asset_id: {self.asset_id}"

    def create(self,
               model: str,
               deployment_name: str,
               serving_name: str = None,
               metadata: Optional[Dict] = None,
               training_data: Optional[Union['DataFrame', 'ndarray']] = None,
               training_target: Optional[Union['DataFrame', 'ndarray']] = None,
               experiment_run_id: Optional[str] = None,
               hardware_spec: Optional[dict] = None) -> None:
        """Create deployment from a model.

        :param model: AutoAI model name
        :type model: str

        :param deployment_name: name of the deployment
        :type deployment_name: str

        :param training_data: training data for the model
        :type training_data: pandas.DataFrame or numpy.ndarray, optional

        :param training_target: target/label data for the model
        :type training_target: pandas.DataFrame or numpy.ndarray, optional

        :param serving_name: serving name of the deployment
        :type serving_name: str, optional

        :param metadata: model meta properties
        :type metadata: dict, optional

        :param experiment_run_id: ID of a training/experiment (only applicable for AutoAI deployments)
        :type experiment_run_id: str, optional

        :param hardware_spec: hardware specification for deployment
        :type hardware_spec: dict, optional

        **Example**

        .. code-block:: python

            from ibm_watson_machine_learning.deployment import WebService

            deployment = WebService(
                   wml_credentials={
                         "apikey": "...",
                         "iam_apikey_description": "...",
                         "iam_apikey_name": "...",
                         "iam_role_crn": "...",
                         "iam_serviceid_crn": "...",
                         "instance_id": "...",
                         "url": "https://us-south.ml.cloud.ibm.com"
                       },
                    project_id="...",
                    space_id="...")

            deployment.create(
                   experiment_run_id="...",
                   model=model,
                   deployment_name='My new deployment',
                   serving_name='my_new_deployment'
               )
        """
        return super().create(model=model,
                              deployment_name=deployment_name,
                              metadata=metadata,
                              serving_name=serving_name,
                              training_data=training_data,
                              training_target=training_target,
                              experiment_run_id=experiment_run_id,
                              deployment_type='online',
                              hardware_spec=hardware_spec)

    @BaseDeployment._project_to_space_to_project
    def get_params(self) -> Dict:
        """Get deployment parameters."""
        return super().get_params()

    @BaseDeployment._project_to_space_to_project
    def score(self, payload: Union[dict, 'DataFrame'] = pd.DataFrame(), transaction_id: str = None) -> Dict[str, List]:
        """Online scoring on WML. Payload is passed to the WML scoring endpoint where model have been deployed.

        :param payload: DataFrame with data to test the model or dictionary with keys `observations`
            and `supporting_features` and DataFrames with data for `observations` and `supporting_features`
            to score forecasting models
        :type payload: pandas.DataFrame or dict

        :param transaction_id: can be used to indicate under which id the records will be saved into payload table
            in IBM OpenScale
        :type transaction_id: str, optional

        :return: dictionary with list od model output/predicted targets
        :rtype: dict

        **Examples**

        .. code-block:: python

            predictions = web_service.score(payload=test_data)
            print(predictions)

            # Result:
            # {'predictions':
            #     [{
            #         'fields': ['prediction', 'probability'],
            #         'values': [['no', [0.9221385608558003, 0.07786143914419975]],
            #                   ['no', [0.9798324002736079, 0.020167599726392187]]
            #     }]}

            predictions = web_service.score(payload={'observations': new_observations_df})
            predictions = web_service.score(payload={'observations': new_observations_df, 'supporting_features': supporting_features_df}) # supporting features time series forecasting sceanrio
        """
        return super().score(payload=payload, transaction_id=transaction_id)

    @BaseDeployment._project_to_space_to_project
    def delete(self, deployment_id: str = None) -> None:
        """Delete deployment on WML.

        :param deployment_id: ID of the deployment to delete, if empty, current deployment will be deleted
        :type deployment_id: str, optional

        **Example**

        .. code-block:: python

            deployment = WebService(workspace=...)
            # Delete current deployment
            deployment.delete()
            # Or delete a specific deployment
            deployment.delete(deployment_id='...')
        """
        super().delete(deployment_id=deployment_id, deployment_type='online')

    @BaseDeployment._project_to_space_to_project
    def list(self, limit=None) -> 'DataFrame':
        """List WML deployments.

        :param limit: set the limit of how many deployments to list,
            default is `None` (all deployments should be fetched)
        :type limit: int, optional

        :return: Pandas DataFrame with information about deployments
        :rtype: pandas.DataFrame

        **Example**

        .. code-block:: python

            deployment = WebService(workspace=...)
            deployments_list = deployment.list()
            print(deployments_list)

            # Result:
            #                  created_at  ...  status
            # 0  2020-03-06T10:50:49.401Z  ...   ready
            # 1  2020-03-06T13:16:09.789Z  ...   ready
            # 4  2020-03-11T14:46:36.035Z  ...  failed
            # 3  2020-03-11T14:49:55.052Z  ...  failed
            # 2  2020-03-11T15:13:53.708Z  ...   ready
        """
        return super().list(limit=limit, deployment_type='online')

    @BaseDeployment._project_to_space_to_project
    def get(self, deployment_id: str) -> None:
        """Get WML deployment.

        :param deployment_id: ID of the deployment to work with
        :type deployment_id: str

        **Example**

        .. code-block:: python

            deployment = WebService(workspace=...)
            deployment.get(deployment_id="...")
        """
        super().get(deployment_id=deployment_id, deployment_type='online')

    @BaseDeployment._project_to_space_to_project
    def _deploy(self,
                pipeline_model: 'Pipeline',
                deployment_name: str,
                meta_props: Dict,
                serving_name=None,
                result_client=None,
                hardware_spec=None) -> Dict:
        """Deploy model into WML.

        :param pipeline_model: model of the pipeline to deploy
        :type pipeline_model: Pipeline or str

        :param deployment_name: name of the deployment
        :type deployment_name: str

        :param meta_props: model meta properties
        :type meta_props: dict

        :param serving_name: serving name of the deployment
        :type serving_name: str

        :param result_client: tuple with Result DataConnection object and initialized COS client
        :rtype: tuple[DataConnection, resource]

        :return: deployment details
        :rtype: dict
        """
        asset_uid = self._publish_model(pipeline_model=pipeline_model,
                                        meta_props=meta_props)

        self.asset_id = asset_uid

        conf_names = self._target_workspace.wml_client.deployments.ConfigurationMetaNames

        deployment_props = {
            conf_names.NAME: deployment_name,
            conf_names.ONLINE: {}
        }

        if hardware_spec:
            deployment_props[conf_names.HARDWARE_SPEC] = hardware_spec

        if serving_name:
            deployment_props[conf_names.ONLINE]["parameters"] = {conf_names.SERVING_NAME: serving_name}

        print("Deploying model {} using V4 client.".format(asset_uid))
        try:

            deployment_details = self._target_workspace.wml_client.deployments.create(
                artifact_uid=asset_uid,
                meta_props=deployment_props)
            self.deployment_id = self._target_workspace.wml_client.deployments.get_uid(deployment_details)

        except WMLClientError as e:
            raise e

        return deployment_details
