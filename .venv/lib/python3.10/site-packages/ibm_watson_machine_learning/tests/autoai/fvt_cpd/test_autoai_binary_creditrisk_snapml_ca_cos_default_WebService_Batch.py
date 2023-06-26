#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import uuid

from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import bucket_exists, create_bucket, is_cp4d, create_connection_to_cos
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAIAsync, \
    AbstractTestWebservice, AbstractTestBatch

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms


class TestAutoAIRemote(AbstractTestAutoAIAsync, AbstractTestWebservice, AbstractTestBatch, unittest.TestCase):
    """
    The test can be run on CPD
    """

    cos_resource = None
    data_location = './autoai/data/credit_risk_training_500.parquet'

    data_cos_path = 'data/credit_risk_training_500.parquet'

    batch_payload_location = './autoai/data/scoring_payload/credit_risk_scoring_payload.csv'
    batch_payload_cos_location = 'scoring_payload/credit_risk_scoring_payload.csv'

    SPACE_ONLY = False

    OPTIMIZER_NAME = "Credit Risk test sdk"

    BATCH_DEPLOYMENT_WITH_DF = True
    BATCH_DEPLOYMENT_WITH_DA = False

    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc='test description',
        prediction_type=PredictionType.BINARY,
        prediction_column='Risk',
        scoring=Metrics.PRECISION_SCORE_MACRO,
        positive_label='No Risk',
        include_only_estimators=[ClassificationAlgorithms.SnapDT,
                                 ClassificationAlgorithms.SnapRF,
                                 ClassificationAlgorithms.SnapSVM,
                                 ClassificationAlgorithms.SnapLR],
        max_number_of_estimators=4,
        text_processing=False
    )

    def test_00b_prepare_COS_instance_and_connection(self):
        TestAutoAIRemote.connection_id, TestAutoAIRemote.bucket_name = create_connection_to_cos(
            wml_client=self.wml_client,
            cos_credentials=self.cos_credentials,
            cos_endpoint=self.cos_endpoint,
            bucket_name=self.bucket_name,
            save_data=True,
            data_path=self.data_location,
            data_cos_path=self.data_cos_path)

        self.assertIsInstance(self.connection_id, str)

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.data_cos_path
            )
        )
        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

        TestAutoAIRemote.results_connection = None

        self.assertIsNone(obj=TestAutoAIRemote.results_connection)

    def test_11b_check_snap(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(pipeline_params)

        pipeline_nodes = pipeline_params.get('pipeline_nodes')
        self.assertIn('Snap', str(pipeline_nodes), msg=f"{pipeline_nodes}")

    def test_47_run_job_batch_deployed_model_with_connected_asset_excel(self):
        from ibm_watson_machine_learning.helpers.connections import DeploymentOutputAssetLocation
        from ibm_watson_machine_learning.tests.utils import save_data_to_cos_bucket

        batch_payload_location = './autoai/data/scoring_payload/credit_risk_scoring_payload.xlsx'
        batch_payload_cos_location = 'scoring_payload/credit_risk_scoring_payload.xlsx'

        if not self.BATCH_DEPLOYMENT_WITH_CA:
            self.skipTest("Skip batch deployment run job with connected asset with cos connection type")
        data_connections_space_only = []

        self.assertIsNotNone(self.wml_client.default_space_id, "TEST Error: default space was not set correctly")

        test_case_batch_output_filename = "cos_ca_" + self.batch_output_filename.replace('.csv', '.xlsx')

        # results_reference = DataConnection(
        #     location=DeploymentOutputAssetLocation(name=test_case_batch_output_filename))


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

        self.assertIsNotNone(batch_payload_location,
                             "Test configuration failure: Batch payload location is missing")

        AbstractTestBatch.bucket_name = save_data_to_cos_bucket(batch_payload_location,
                                                                batch_payload_cos_location,
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
                path=batch_payload_cos_location,
                excel_sheet='scoring_payload'
            )
        )

        results_reference = DataConnection(
            connection_asset_id=AbstractTestBatch.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=test_case_batch_output_filename,
                excel_sheet='scoring_payload_output'
            ))

        data_connections_space_only.append(conn_space)

        scoring_params = AbstractTestBatch.service_batch.run_job(
            payload=data_connections_space_only,
            output_data_reference=results_reference,
            background_mode=False)

        print(scoring_params)
        self.assertIsNotNone(scoring_params)

        deployment_job_id = self.wml_client.deployments.get_job_uid(scoring_params)
        predictions = AbstractTestBatch.service_batch.get_job_result(deployment_job_id)

        print(predictions)
        self.assertIsNotNone(predictions)

    def test_99_delete_connection_and_connected_data_asset(self):
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
