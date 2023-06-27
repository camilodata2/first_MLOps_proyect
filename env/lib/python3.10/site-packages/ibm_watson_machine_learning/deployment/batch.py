#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import io
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Union, List, Optional

import pandas as pd
from pandas import DataFrame, concat

from .base_deployment import BaseDeployment
from ..helpers import DataConnection, AssetLocation
from ..utils import StatusLogger, print_text_header_h1
from ..utils.autoai.connection import validate_source_data_connections, validate_deployment_output_connection
from ..utils.autoai.utils import init_cos_client, try_load_dataset, convert_dataframe_to_fields_values_payload
from ..utils.autoai.errors import NoneDataConnection
from ..utils.deployment.errors import BatchJobFailed, MissingScoringResults
from ..wml_client_error import WMLClientError
from ibm_watson_machine_learning.utils.autoai.enums import DataConnectionTypes

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from pandas import DataFrame
    from numpy import ndarray
    from ..workspace import WorkSpace

__all__ = [
    "Batch"
]


class Batch(BaseDeployment):
    """The Batch Deployment class.
    With this class object you can manage any batch deployment.

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
            deployment_type='batch',
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
        self.id = None
        self.asset_id = None

    def __repr__(self):
        return f"name: {self.name}, id: {self.id}, asset_id: {self.asset_id}"

    def __str__(self):
        return f"name: {self.name}, id: {self.id}, asset_id: {self.asset_id}"

    def score(self, **kwargs):
        raise NotImplementedError("Batch deployment supports only job runs.")

    def create(self,
               model: str,
               deployment_name: str,
               metadata: Optional[Dict] = None,
               training_data: Optional[Union['DataFrame', 'ndarray']] = None,
               training_target: Optional[Union['DataFrame', 'ndarray']] = None,
               experiment_run_id: Optional[str] = None) -> None:
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

        **Example**

        .. code-block:: python

            from ibm_watson_machine_learning.deployment import Batch

            deployment = Batch(
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
                   deployment_name='My new deployment'
               )
        """
        return super().create(model=model,
                              deployment_name=deployment_name,
                              metadata=metadata,
                              training_data=training_data,
                              training_target=training_target,
                              experiment_run_id=experiment_run_id,
                              deployment_type='batch')

    @BaseDeployment._project_to_space_to_project
    def get_params(self) -> Dict:
        """Get deployment parameters."""
        return super().get_params()

    @BaseDeployment._project_to_space_to_project
    def run_job(self,
                payload: Union[DataFrame, List[DataConnection], Dict[str, DataFrame], Dict[str, DataConnection]] = pd.DataFrame(),
                output_data_reference: 'DataConnection' = None,
                transaction_id: str = None,
                background_mode: 'bool' = True) -> Union[Dict, Dict[str, List], DataConnection]:
        """Batch scoring job on WML. Payload or Payload data reference is required.
        It is passed to the WML where model have been deployed.

        :param payload: DataFrame that contains data to test the model or data storage connection details
            that inform the model where payload data is stored
        :type payload: pandas.DataFrame or List[DataConnection] or Dict

        :param output_data_reference: DataConnection to the output COS for storing predictions,
             required only when DataConnections are used as a payload
        :type output_data_reference: DataConnection, optional

        :param transaction_id: can be used to indicate under which id the records will be saved into payload table
            in IBM OpenScale
        :type transaction_id: str, optional

        :param background_mode: indicator if score() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: scoring job details
        :rtype: dict

        **Examples**

        .. code-block:: python

            score_details = batch_service.run_job(payload=test_data)
            print(score_details['entity']['scoring'])

            # Result:
            # {'input_data': [{'fields': ['sepal_length',
            #               'sepal_width',
            #               'petal_length',
            #               'petal_width'],
            #              'values': [[4.9, 3.0, 1.4, 0.2]]}],
            # 'predictions': [{'fields': ['prediction', 'probability'],
            #               'values': [['setosa',
            #                 [0.9999320742502246,
            #                  5.1519823540224506e-05,
            #                  1.6405926235405522e-05]]]}]

            payload_reference = DataConnection(location=DSLocation(asset_id=asset_id))
            score_details = batch_service.run_job(payload=payload_reference, output_data_filename = "scoring_output.csv")
            score_details = batch_service.run_job(payload={'observations': payload_reference})
            score_details = batch_service.run_job(payload=[payload_reference])
            score_details = batch_service.run_job(payload={'observations': payload_reference, 'supporting_features': supporting_features_reference})  # supporting features time series forecasting sceanrio
        """

        if isinstance(payload, dict):
            observations = payload.get('observations', pd.DataFrame())
            supporting_features = payload.get('supporting_features')

            if isinstance(observations, DataFrame) and \
                    (isinstance(supporting_features, DataFrame) or supporting_features is None):
                observations_payload = convert_dataframe_to_fields_values_payload(observations, return_values_only=True)
                observations_payload['id'] = 'observations'
                input_data = [observations_payload]

                if supporting_features is not None:
                    supporting_features_payload = convert_dataframe_to_fields_values_payload(supporting_features, return_values_only=True)
                    supporting_features_payload['id'] = 'supporting_features'
                    input_data.append(supporting_features_payload)

                scoring_payload = {
                    self._target_workspace.wml_client.deployments.ScoringMetaNames.INPUT_DATA: input_data
                }

            elif isinstance(observations, DataConnection) and \
                    (isinstance(supporting_features, DataConnection) or supporting_features is None):

                observations.id = 'observations'
                input_data = [observations]
                if supporting_features is not None:
                    supporting_features.id = 'supporting_features'
                    input_data.append(supporting_features)

                for data_conn in input_data:
                    if hasattr(data_conn, 'location') and isinstance(data_conn.location, AssetLocation):
                        data_conn.location.wml_client = self._target_workspace.wml_client

                input_data = validate_source_data_connections(source_data_connections=input_data,
                                                              workspace=self._target_workspace,
                                                              deployment=True)
                input_data = [data_connection._to_dict() for data_connection in input_data]

                if output_data_reference is None:
                    raise ValueError("\"output_data_reference\" should be provided.")

                if isinstance(output_data_reference, DataConnection):

                    # wml_client sets correct href for Data Assets
                    if hasattr(output_data_reference, 'location') and isinstance(output_data_reference.location,
                                                                                 AssetLocation):
                        output_data_reference.location.wml_client = self._target_workspace.wml_client

                    output_data_reference = validate_deployment_output_connection(
                        results_data_connection=output_data_reference,
                        workspace=self._target_workspace,
                        source_data_connections=input_data)
                    output_data_reference = output_data_reference._to_dict()

                scoring_payload = {
                    self._target_workspace.wml_client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES: input_data,
                    self._target_workspace.wml_client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE:
                        output_data_reference}

            else:
                raise TypeError('Missing data observations in payload or observations '
                                 'or supporting_features are not pandas.DataFrames.')
        # note: support for DataFrame payload
        elif isinstance(payload, DataFrame):
            scoring_payload = {
                self._target_workspace.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{'values': payload}]
            }
        # note: support for DataConnections and dictionaries payload
        elif isinstance(payload, list):
            if isinstance(payload[0], DataConnection):
                if None in payload:
                    raise NoneDataConnection('payload')

                # wml_client sets correct href for Data Assets
                for data_conn in payload:
                    if hasattr(data_conn, 'location') and isinstance(data_conn.location, AssetLocation):
                        data_conn.location.wml_client = self._target_workspace.wml_client

                payload = [new_conn for conn in payload for new_conn in conn._subdivide_connection()]
                payload = validate_source_data_connections(source_data_connections=payload,
                                                           workspace=self._target_workspace,
                                                           deployment=True)
                payload = [data_connection._to_dict() for data_connection in payload]
            elif isinstance(payload[0], dict):
                pass
            else:
                raise ValueError(f"Current payload type: list of {type(payload[0])} is not supported.")

            if output_data_reference is None:
                raise ValueError("\"output_data_reference\" should be provided.")

            if isinstance(output_data_reference, DataConnection):

                # wml_client sets correct href for Data Assets
                if hasattr(output_data_reference, 'location') and isinstance(output_data_reference.location, AssetLocation):
                    output_data_reference.location.wml_client = self._target_workspace.wml_client

                output_data_reference = validate_deployment_output_connection(
                    results_data_connection=output_data_reference,
                    workspace=self._target_workspace,
                    source_data_connections=payload)
                output_data_reference = output_data_reference._to_dict()

            scoring_payload = {
                self._target_workspace.wml_client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES: payload,
                self._target_workspace.wml_client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE:
                    output_data_reference}

        else:
            raise ValueError(
                f"Incorrect payload type. Required: DataFrame or List[DataConnection], Passed: {type(payload)}")

        scoring_payload['hybrid_pipeline_hardware_specs'] = [
            {
                'node_runtime_id': 'auto_ai.kb',
                'hardware_spec': {
                    'name': 'M'
                }
            }
        ]

        if self._obm:
            scoring_payload['hybrid_pipeline_hardware_specs'].insert(
                0,
                {
                    "node_runtime_id": "auto_ai.obm",
                    "hardware_spec": {
                        "name": "M-Spark",
                        "num_nodes": 2
                    }
                }
            )

        job_details = self._target_workspace.wml_client.deployments.create_job(self.id,
                                                                               scoring_payload, _asset_id=self.asset_id)

        if background_mode:
            return job_details

        else:
            # note: monitor scoring job
            job_id = self._target_workspace.wml_client.deployments.get_job_uid(job_details)
            print_text_header_h1(u'Synchronous scoring for id: \'{}\' started'.format(job_id))

            status = self.get_job_status(job_id)['state']

            with StatusLogger(status) as status_logger:
                while status not in ['failed', 'error', 'completed', 'canceled']:
                    time.sleep(10)
                    status = self.get_job_status(job_id)['state']
                    status_logger.log_state(status)
            # --- end note

            if u'completed' in status:
                print(u'\nScoring job  \'{}\' finished successfully.'.format(job_id))
            else:
                raise BatchJobFailed(job_id, f"Scoring job failed with status: {self.get_job_status(job_id)}")

            return self.get_job_params(job_id)

    @BaseDeployment._project_to_space_to_project
    def rerun_job(self,
                  scoring_job_id: str,
                  background_mode: bool = True) -> Union[dict, 'DataFrame', 'DataConnection']:
        """Rerun scoring job with the same parameters as job described by `scoring_job_id`.

        :param scoring_job_id: Id described scoring job
        :type scoring_job_id: str

        :param background_mode: indicator if score_rerun() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: scoring job details
        :rtype: dict

        **Example**

        .. code-block:: python

            scoring_details = deployment.score_rerun(scoring_job_id)
        """
        scoring_params = self.get_job_params(scoring_job_id)['entity']['scoring']
        input_data_references = self._target_workspace.wml_client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES
        output_data_reference = self._target_workspace.wml_client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE

        if input_data_references in scoring_params:
            payload_ref = [input_ref for input_ref in scoring_params[input_data_references]]

            if 'href' in scoring_params[output_data_reference]['location']:
                del scoring_params[output_data_reference]['location']['href']

            return self.run_job(payload=payload_ref, output_data_reference=scoring_params['output_data_reference'],
                                background_mode=background_mode)
        else:
            raise NotImplementedError("'rerun_job' method supports only jobs with "
                                      "payload passed as a list of DataConnections. If you want to rerun job "
                                      "with payload passed directly, please use 'run_job' one more time.")

    @BaseDeployment._project_to_space_to_project
    def delete(self, deployment_id: str = None) -> None:
        """Delete deployment on WML.

        :param deployment_id: ID of the deployment to delete, if empty, current deployment will be deleted
        :type deployment_id: str, optional

        **Example**

        .. code-block:: python

            deployment = Batch(workspace=...)
            # Delete current deployment
            deployment.delete()
            # Or delete a specific deployment
            deployment.delete(deployment_id='...')
        """
        super().delete(deployment_id=deployment_id, deployment_type='batch')

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

            deployment = Batch(workspace=...)
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
        return super().list(limit=limit, deployment_type='batch')

    @BaseDeployment._project_to_space_to_project
    def get(self, deployment_id: str) -> None:
        """Get WML deployment.

        :param deployment_id: ID of the deployment to work with
        :type deployment_id: str

        **Example**

        .. code-block:: python

            deployment = Batch(workspace=...)
            deployment.get(deployment_id="...")
        """
        super().get(deployment_id=deployment_id, deployment_type='batch')

    @BaseDeployment._project_to_space_to_project
    def get_job_params(self, scoring_job_id: str = None) -> Dict:
        """Get batch deployment job parameters.

        :param scoring_job_id: Id of scoring job
        :type scoring_job_id: str

        :return: parameters of the scoring job
        :rtype: dict
        """
        return self._target_workspace.wml_client.deployments.get_job_details(scoring_job_id)

    @BaseDeployment._project_to_space_to_project
    def get_job_status(self, scoring_job_id: str) -> Dict:
        """Get status of scoring job.

        :param scoring_job_id: Id of scoring job
        :type scoring_job_id: str

        :return: dictionary with state of scoring job (one of: [completed, failed, starting, queued])
            and additional details if they exist
        :rtype: dict
        """
        return self._target_workspace.wml_client.deployments.get_job_status(scoring_job_id)

    @BaseDeployment._project_to_space_to_project
    def get_job_result(self, scoring_job_id: str) -> 'DataFrame':
        """Get batch deployment results of job with id `scoring_job_id`.

        :param scoring_job_id: Id of scoring job which results will be returned
        :type scoring_job_id: str

        :return: result
        :rtype: pandas.DataFrame

        :raises MissingScoringResults: in case of incompleted or failed job
            `MissingScoringResults` scoring exception is raised
        """
        scoring_params = self.get_job_params(scoring_job_id)['entity']['scoring']
        if scoring_params['status']['state'] == 'completed':
            if 'predictions' in scoring_params:
                data = DataFrame(scoring_params['predictions'][0]['values'], columns=scoring_params['predictions'][0]['fields'])
                return data
            else:
                conn = DataConnection._from_dict(scoring_params['output_data_reference'])
                conn._wml_client = self._target_workspace.wml_client

                return conn.read(raw=True) # if in future output may be excel file or with custom separator, here it should be recognized
        else:
            raise MissingScoringResults(scoring_job_id, reason="Scoring is not completed.")

    @BaseDeployment._project_to_space_to_project
    def get_job_id(self, batch_scoring_details):
        """Get id from batch scoring details."""
        return self._target_workspace.wml_client.deployments.get_job_uid(batch_scoring_details)

    @BaseDeployment._project_to_space_to_project
    def list_jobs(self):
        """Returns pandas DataFrame with list of deployment jobs"""

        resources = self._target_workspace.wml_client.deployments.get_job_details()['resources']
        columns = [u'job id', u'state', u'creted', u'deployment id']
        values = []
        for scoring_details in resources:
            if 'scoring' in scoring_details['entity']:
                state = scoring_details['entity']['scoring']['status']['state']
                score_values = (scoring_details[u'metadata'][u'id'], state,
                                scoring_details[u'metadata'][u'created_at'],
                                scoring_details['entity']['deployment']['id'])
                if self.id:
                    if self.id == scoring_details['entity']['deployment']['id']:
                        values.append(score_values)
                else:
                    values.append(score_values)

        return DataFrame(values, columns=columns)

    @BaseDeployment._project_to_space_to_project
    def _deploy(self,
                pipeline_model: 'Pipeline',
                deployment_name: str,
                meta_props: Dict,
                serving_name=None, # Not used, but added to match unified parameters for _deploy
                result_client=None,
                hardware_spec=None) -> Dict: # Not used, but added to match unified parameters for _deploy
        """Deploy model into WML.

        :param pipeline_model: model of the pipeline to deploy
        :type pipeline_model: Pipeline or str

        :param deployment_name: name of the deployment
        :type deployment_name: str

        :param meta_props: model meta properties
        :type meta_props: dict

        :param result_client: tuple with Result DataConnection object and initialized COS client
        :type result_client: tuple[DataConnection, resource]
        """
        deployment_details = {}
        asset_uid = self._publish_model(pipeline_model=pipeline_model,
                                        meta_props=meta_props)

        self.asset_id = asset_uid

        deployment_props = {
            self._target_workspace.wml_client.deployments.ConfigurationMetaNames.NAME: deployment_name,
            self._target_workspace.wml_client.deployments.ConfigurationMetaNames.BATCH: {}
        }

        deployment_props[self._target_workspace.wml_client.deployments.ConfigurationMetaNames.ASSET] = {
            "id": asset_uid
        }

        deployment_props[
            self._target_workspace.wml_client.deployments.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS] = [
            {
                'node_runtime_id': 'auto_ai.kb',
                'hardware_spec': {
                    'name': 'M'
                }
            }
        ]

        if self._obm:
            deployment_props[
                self._target_workspace.wml_client.deployments.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS
            ].insert(0, {
                "node_runtime_id": "auto_ai.obm",
                "hardware_spec": {
                    "name": "M-Spark",
                    "num_nodes": 2
                }
            })

        print("Deploying model {} using V4 client.".format(asset_uid))
        try:
            deployment_details = self._target_workspace.wml_client.deployments.create(
                artifact_uid=asset_uid,
                meta_props=deployment_props)

            self.deployment_id = self._target_workspace.wml_client.deployments.get_uid(deployment_details)

        except WMLClientError as e:
            raise e

        return deployment_details
