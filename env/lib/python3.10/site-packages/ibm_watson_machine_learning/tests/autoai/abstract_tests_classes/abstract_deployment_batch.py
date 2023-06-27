#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc
import os
from datetime import datetime
from os import environ

import unittest

from sklearn.pipeline import Pipeline

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.deployment import Batch
from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.helpers.connections import DataConnection, AssetLocation, \
    DeploymentOutputAssetLocation, S3Location
from ibm_watson_machine_learning.tests.utils import save_data_to_cos_bucket
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError

from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup, delete_model_deployment
from ibm_watson_machine_learning.utils.autoai.enums import TShirtSize, PredictionType


class AbstractTestBatch(abc.ABC):
    """
    The abstract tests which covers:
    - deployment with lale pipeline
    - deployment deletion
    In order to execute test connection definitions must be provided
    in inheriting classes.
    """

    # FLAGS
    SPACE_ONLY: bool  # if True, experiment was run on space.

    BATCH_DEPLOYMENT_WITH_DF = True  # batch input passed as Pandas.DataFrame
    BATCH_DEPLOYMENT_WITH_DA = True  # batch input passed as DataConnection type data-assets with csv files
    BATCH_DEPLOYMENT_WITH_CDA = True  # batch input passed as DataConnection type data-assets with connection_id(COS)
    BATCH_DEPLOYMENT_WITH_CA = True  # batch input passed as DataConnection type connected assets (COS)
    # end FLAGS

    DEPLOYMENT_NAME = "SDK tests Deployment"

    wml_client: 'APIClient'
    remote_auto_pipelines: 'RemoteAutoPipelines'
    wml_credentials: str
    service_batch: 'Batch'

    connection_id = None
    asset_id: str
    batch_payload_location: str
    batch_payload_cos_location: str
    batch_output_filename = f"batch_output_{datetime.utcnow().isoformat()}.csv"

    space_id: str
    target_space_id: str
    project_id: str

    data_assets_to_delete: set = set()

    X_df = None

    @abc.abstractmethod
    def test_00a_space_cleanup(self):
        pass

    ########################################################
    #  Batch deployment (possible test numbers are: 40-54) #
    ########################################################

    def test_40_deployment_target_space_setup(self):
        # note: if target_space_id is not set, use the space_id
        if self.target_space_id is None:
            AbstractTestBatch.target_space_id = self.space_id
        else:
            AbstractTestBatch.target_space_id = self.target_space_id
        # end note

        self.wml_client.set.default_space(AbstractTestBatch.target_space_id)

    def test_41_batch_deployment_setup_and_preparation(self):

        self.assertIsNotNone(AbstractTestBatch.target_space_id, "Test issue: target space not set.")

        if self.SPACE_ONLY:
            AbstractTestBatch.service_batch = Batch(source_wml_credentials=self.wml_credentials,
                                                    source_space_id=self.space_id,
                                                    target_wml_credentials=self.wml_credentials,
                                                    target_space_id=AbstractTestBatch.target_space_id)
        else:
            AbstractTestBatch.service_batch = Batch(source_wml_credentials=self.wml_credentials,
                                                    source_project_id=self.project_id,
                                                    target_wml_credentials=self.wml_credentials,
                                                    target_space_id=AbstractTestBatch.target_space_id)

        self.assertIsInstance(AbstractTestBatch.service_batch, Batch, msg="Deployment is not of Batch type.")
        self.assertIsInstance(AbstractTestBatch.service_batch._source_workspace, WorkSpace,
                              msg="Workspace set incorrectly.")
        self.assertEqual(AbstractTestBatch.service_batch.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(AbstractTestBatch.service_batch.name, None, msg="Deployment name initialized incorrectly")

    def test_42_deploy__batch_deploy_best_computed_pipeline_from_autoai_on_wml(self):
        AbstractTestBatch.service_batch.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model="Pipeline_2",
            deployment_name=self.DEPLOYMENT_NAME + ' BATCH')

        self.assertIsNotNone(AbstractTestBatch.service_batch.id, msg="Batch Deployment creation - missing id")
        self.assertIsNotNone(AbstractTestBatch.service_batch.id, msg="Batch Deployment creation - name not set")
        self.assertIsNotNone(AbstractTestBatch.service_batch.asset_id,
                             msg="Batch Deployment creation - model (asset) id missing, incorrect model storing")

    def test_43_list_batch_deployments(self):
        deployments = AbstractTestBatch.service_batch.list()
        print(deployments)
        self.assertIn('created_at', str(deployments).lower())
        self.assertIn('status', str(deployments).lower())

        params = AbstractTestBatch.service_batch.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_44_run_job__batch_deployed_model_with_data_frame(self):
        if self.BATCH_DEPLOYMENT_WITH_DF:
            scoring_params = AbstractTestBatch.service_batch.run_job(
                payload=self.X_df,
                background_mode=False)
            print(scoring_params)
            print(AbstractTestBatch.service_batch.get_job_result(scoring_params['metadata']['id']))
            self.assertIsNotNone(scoring_params)
            self.assertIn('predictions', str(scoring_params).lower())
        else:
            self.skipTest("Skip batch deployment run job with data frame")

    def test_45_run_job_batch_deployed_model_with_data_assets(self):
        data_connections_space_only = []
        # self.wml_client.set.default_space(self.target_space_id)

        test_case_batch_output_filename = "da_" + self.batch_output_filename

        results_reference = DataConnection(
            location=DeploymentOutputAssetLocation(name=test_case_batch_output_filename))

        if not self.BATCH_DEPLOYMENT_WITH_DA:
            self.skipTest("Skip batch deployment run job with data asset type")
        else:
            self.assertIsNotNone(self.batch_payload_location,
                                 "Test configuration failure: Batch payload location is missing")

            asset_details = self.wml_client.data_assets.create(
                name=self.batch_payload_location.split('/')[-1],
                file_path=self.batch_payload_location)
            asset_id = self.wml_client.data_assets.get_id(asset_details)
            AbstractTestBatch.data_assets_to_delete.add(asset_id)
            data_connections_space_only = [DataConnection(location=AssetLocation(asset_id=asset_id))]

        scoring_params = AbstractTestBatch.service_batch.run_job(
            payload=data_connections_space_only,
            output_data_reference=results_reference,
            background_mode=False)

        print(scoring_params)
        self.assertIsNotNone(scoring_params)

        deployment_job_id = self.wml_client.deployments.get_job_uid(scoring_params)

        self.wml_client.data_assets.list()

        data_asset_details = self.wml_client.data_assets.get_details()
        self.assertIn(test_case_batch_output_filename, str(data_asset_details),
                      f"Batch output file: {test_case_batch_output_filename} missed in data assets")

        predictions = AbstractTestBatch.service_batch.get_job_result(deployment_job_id)
        print(predictions)

        self.assertIsNotNone(predictions)

    def test_46_run_job_batch_deployed_model_with_data_assets_with_cos_connection(self):
        if not self.BATCH_DEPLOYMENT_WITH_CDA:
            self.skipTest("Skip batch deployment run job with data asset with cos connection type")

        data_connections_space_only = []
        # self.wml_client.set.default_space(self.target_space_id)

        test_case_batch_output_filename = "cos_da_" + self.batch_output_filename

        results_reference = DataConnection(
            location=DeploymentOutputAssetLocation(name=test_case_batch_output_filename))

        AbstractTestBatch.bucket_name = save_data_to_cos_bucket(self.batch_payload_location,
                                                                self.batch_payload_cos_location,
                                                                access_key_id=self.cos_credentials['cos_hmac_keys'][
                                                                    'access_key_id'],
                                                                secret_access_key=
                                                                self.cos_credentials['cos_hmac_keys'][
                                                                    'secret_access_key'],
                                                                cos_endpoint=self.cos_endpoint,
                                                                bucket_name=self.bucket_name)

        # prepare connection
        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(
                'bluemixcloudobjectstorage'),
            'name': 'Connection to COS for tests',
            'properties': {
                'bucket': self.bucket_name,
                'access_key': self.cos_credentials['cos_hmac_keys']['access_key_id'],
                'secret_key': self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                'iam_url': self.wml_client.service_instance._href_definitions.get_iam_token_url(),
                'url': self.cos_endpoint
            }
        })

        AbstractTestBatch.connection_id = self.wml_client.connections.get_uid(connection_details)

        self.assertIsNotNone(self.batch_payload_location,
                             "Test configuration failure: Batch payload location is missing")


        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: AbstractTestBatch.connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: "Batch deployment asset",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: f"{self.bucket_name}/{self.batch_payload_cos_location}"
        })

        asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(asset_id, str)
        AbstractTestBatch.data_assets_to_delete.add(asset_id)

        data_connections_space_only.append(
            DataConnection(data_asset_id=asset_id))

        self.assertEqual(len(data_connections_space_only), 1)

        scoring_params = AbstractTestBatch.service_batch.run_job(
            payload=data_connections_space_only,
            output_data_reference=results_reference,
            background_mode=False)

        print(scoring_params)
        self.assertIsNotNone(scoring_params)

        deployment_job_id = self.wml_client.deployments.get_job_uid(scoring_params)

        self.wml_client.data_assets.list()

        data_asset_details = self.wml_client.data_assets.get_details()
        self.assertIn(test_case_batch_output_filename, str(data_asset_details),
                      f"Batch output file: {test_case_batch_output_filename} missed in data assets")

        predictions = AbstractTestBatch.service_batch.get_job_result(deployment_job_id)
        print(predictions)

        self.assertIsNotNone(predictions)

    def test_47_run_job_batch_deployed_model_with_connected_data_asset(self):

        if not self.BATCH_DEPLOYMENT_WITH_CA:
            self.skipTest("Skip batch deployment run job with connected asset with cos connection type")
        data_connections_space_only = []

        self.assertIsNotNone(self.wml_client.default_space_id, "TEST Error: default space was not set correctly")

        test_case_batch_output_filename = "cos_ca_" + self.batch_output_filename

        results_reference = DataConnection(
            location=DeploymentOutputAssetLocation(name=test_case_batch_output_filename))

        # prepare connection
        if not AbstractTestBatch.connection_id:
            connection_details = self.wml_client.connections.create({
                'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(
                    'bluemixcloudobjectstorage'),
                'name': 'Connection to COS for tests',
                'properties': {
                    'bucket': self.bucket_name,
                    'access_key': self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    'secret_key': self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                    'iam_url': self.wml_client.service_instance._href_definitions.get_iam_token_url(),
                    'url': self.cos_endpoint
                }
            })

            AbstractTestBatch.connection_id = self.wml_client.connections.get_uid(connection_details)


        self.assertIsNotNone(self.batch_payload_location,
                             "Test configuration failure: Batch payload location is missing")

        AbstractTestBatch.bucket_name = save_data_to_cos_bucket(self.batch_payload_location,
                                                                self.batch_payload_cos_location,
                                                                access_key_id=self.cos_credentials['cos_hmac_keys'][
                                                                    'access_key_id'],
                                                                secret_access_key=
                                                                self.cos_credentials['cos_hmac_keys'][
                                                                    'secret_access_key'],
                                                                cos_endpoint=self.cos_endpoint,
                                                                bucket_name=self.bucket_name)
        conn_space = DataConnection(
            connection_asset_id=AbstractTestBatch.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.batch_payload_cos_location
            )
        )

        data_connections_space_only.append(conn_space)

        scoring_params = AbstractTestBatch.service_batch.run_job(
            payload=data_connections_space_only,
            output_data_reference=results_reference,
            background_mode=False)

        print(scoring_params)
        self.assertIsNotNone(scoring_params)

        deployment_job_id = self.wml_client.deployments.get_job_uid(scoring_params)

        self.wml_client.data_assets.list()

        data_asset_details = self.wml_client.data_assets.get_details()
        self.assertIn(test_case_batch_output_filename, str(data_asset_details),
                      f"Batch output file: {test_case_batch_output_filename} missed in data assets")

        predictions = AbstractTestBatch.service_batch.get_job_result(deployment_job_id)

        print(predictions)
        self.assertIsNotNone(predictions)

    def test_48_run_job_batch_deployed_model_with_data_connection_container(self):
        if self.wml_client.ICP or self.wml_client.WSD:
            self.skipTest("Batch Deployment with container data connection is available only for Cloud")
        else:
            self.skipTest("not ready")

    def test_49_delete_deployment_batch(self):
        print("Delete current deployment: {}".format(AbstractTestBatch.service_batch.deployment_id))
        AbstractTestBatch.service_batch.delete()
        self.wml_client.set.default_space(self.target_space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(AbstractTestBatch.service_batch.asset_id)

        try:
            print(f"Delete all created assets in Batch deployment tests. {AbstractTestBatch.data_assets_to_delete}")
            for asset_id in AbstractTestBatch.data_assets_to_delete:
                self.wml_client.data_assets.delete(asset_id)
        except WMLClientError:
            print("Not able to dele data assets")

        self.assertEqual(AbstractTestBatch.service_batch.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(AbstractTestBatch.service_batch.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(AbstractTestBatch.service_batch.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")
