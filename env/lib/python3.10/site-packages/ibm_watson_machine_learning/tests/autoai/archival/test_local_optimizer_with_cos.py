#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from functools import wraps

from ibm_watson_machine_learning.helpers import DataConnection, S3Connection, S3Location
from ibm_watson_machine_learning.tests.utils import get_cos_credentials, get_wml_credentials, print_test_separators
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.experiment.autoai.optimizers.local_auto_pipelines import LocalAutoPipelines
from lale.operators import TrainedPipeline
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.utils.autoai.enums import PipelineTypes
from sklearn.pipeline import Pipeline
import pandas as pd

"""
Format of COS credentials:

cos_credentials = {
                  "auth_endpoint": "https://iam.cloud.ibm.com/identity/token",
                  "endpoint_url": "https://s3.us.cloud-object-storage.appdomain.cloud",
                  "apikey": "...", 
                  "cos_hmac_keys": {
                    "access_key_id": "...",
                    "secret_access_key": "..."
                  }
                  }
                  
endpoint_url should be appropriate per bucket location, see on UI.
"""


class TestLocalOptimizerWithCOS(unittest.TestCase):
    cos_credentials = None
    bucket = 'nowyobm'
    path = 'bank.csv'
    pipeline_model_hmac = None
    pipeline_model_auth_endpoint = None
    metadata_hmac = None
    metadata_auth_endpoint = None
    pipeline_name = 'Pipeline_1'
    training_status = 'results_wml_autoai/07652672-f643-4b49-b0eb-449fde9d1a20/training-status.json'
    model_location = 'results_wml_autoai/07652672-f643-4b49-b0eb-449fde9d1a20/data/kb/global_output/Pipeline3/model.pickle'

    local_optimizer_hmac: 'LocalAutoPipelines' = None
    local_optimizer: 'LocalAutoPipelines' = None

    data_location = './autoai/data/bank.csv'
    data = None
    X = None
    y = None

    @classmethod
    def setUp(cls) -> None:
        cls.wml_credentials = get_wml_credentials()
        cls.cos_credentials = get_cos_credentials()
        cls.cos_credentials['endpoint_url'] = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
        cls.cos_credentials['auth_endpoint'] = "https://iam.bluemix.net/oidc/token/"
        cls.cos_credentials['api_key'] = '***'
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['y'], axis=1)
        cls.y = cls.data['y']

    @print_test_separators
    def test_01a_get_optimizer_with_COS_hmac_credentials(self):
        print("HMAC: Initializing DataConnections with COS credentials and paths...")

        training_data_reference = [DataConnection(
            connection=S3Connection(
                access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                endpoint_url=self.cos_credentials['endpoint_url']
            ),
            location=S3Location(
                bucket=self.bucket,
                path=self.path,
            )
        )]

        training_result_reference = DataConnection(
            connection=S3Connection(
                access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                endpoint_url=self.cos_credentials['endpoint_url']
            ),
            location=S3Location(
                bucket=self.bucket,
                path=self.path,
                model_location=self.model_location,
                training_status=self.training_status
            )
        )

        TestLocalOptimizerWithCOS.metadata_hmac = dict(
            training_data_reference=training_data_reference,
            training_result_reference=training_result_reference,
            prediction_type='classification',
            prediction_column='y',
            test_size=0.2,
            scoring='roc_auc',
            max_number_of_estimators=1,
        )

        print("HMAC: Initializing AutoAI local scenario with metadata...")

        TestLocalOptimizerWithCOS.local_optimizer_hmac = AutoAI().runs.get_optimizer(
            metadata=self.metadata_hmac
        )

    @print_test_separators
    def test_01b_get_optimizer_with_auth_endpoint_credentials(self):
        print("AUTH: Initializing DataConnections with COS credentials and paths...")

        training_data_reference = [DataConnection(
            connection=S3Connection(
                api_key=self.cos_credentials['apikey'],
                auth_endpoint=self.cos_credentials['auth_endpoint'],
                endpoint_url=self.cos_credentials['endpoint_url']
            ),
            location=S3Location(
                bucket=self.bucket,
                path=self.path,
            )
        )]

        training_result_reference = DataConnection(
            connection=S3Connection(
                api_key=self.cos_credentials['apikey'],
                auth_endpoint=self.cos_credentials['auth_endpoint'],
                endpoint_url=self.cos_credentials['endpoint_url']
            ),
            location=S3Location(
                bucket=self.bucket,
                path=self.path,
                model_location=self.model_location,
                training_status=self.training_status
            )
        )

        TestLocalOptimizerWithCOS.metadata_auth_endpoint = dict(
            training_data_reference=training_data_reference,
            training_result_reference=training_result_reference,
            prediction_type='classification',
            prediction_column='y',
            test_size=0.2,
            scoring='roc_auc',
            max_number_of_estimators=1,
        )

        print("AUTH: Initializing AutoAI local scenario with metadata...")
        TestLocalOptimizerWithCOS.local_optimizer = AutoAI().runs.get_optimizer(
            metadata=self.metadata_auth_endpoint
        )

    @print_test_separators
    def test_02a_get_pipeline_hmac(self):
        print("HMAC: Fetching and store pipeline by name, LALE...")

        pipeline_lale = self.local_optimizer_hmac.get_pipeline(
            pipeline_name=self.pipeline_name, persist=True)
        self.assertIsInstance(pipeline_lale, TrainedPipeline, msg="Loaded model should be of type TrainedPipeline")

        print("HMAC: Fetching pipeline by name, SKLEARN...")

        pipeline_skl = self.local_optimizer_hmac.get_pipeline(
            pipeline_name=self.pipeline_name, astype=PipelineTypes.SKLEARN)
        self.assertIsInstance(pipeline_skl, Pipeline, msg="Loaded model should be of type SKLEARN")

        print("HMAC: Fetching best pipeline, LALE...")

        try:
            TestLocalOptimizerWithCOS.pipeline_model_hmac = self.local_optimizer_hmac.get_pipeline()
        except Exception as e:
            self.assertIsInstance(self.pipeline_model_hmac, TrainedPipeline,
                                  msg=f"Cannot load best pipeline model, Error: {e}")

    @print_test_separators
    def test_02b_get_pipeline_auth_endpoint(self):
        print("AUTH: Fetching and store pipeline by name, LALE...")

        pipeline_lale = self.local_optimizer.get_pipeline(pipeline_name=self.pipeline_name, persist=True)
        self.assertIsInstance(pipeline_lale, TrainedPipeline, msg="Loaded model should be of type TrainedPipeline")

        print("AUTH: Fetching pipeline by name, SKLEARN...")

        pipeline_skl = self.local_optimizer.get_pipeline(pipeline_name=self.pipeline_name, astype=PipelineTypes.SKLEARN)
        self.assertIsInstance(pipeline_skl, Pipeline, msg="Loaded model should be of type SKLEARN")

        print("AUTH: Fetching best pipeline, LALE...")

        try:
            TestLocalOptimizerWithCOS.pipeline_model_auth_endpoint = self.local_optimizer.get_pipeline()
        except Exception as e:
            self.assertIsInstance(self.pipeline_model_auth_endpoint, TrainedPipeline,
                                  msg=f"Cannot load best pipeline model, Error: {e}")

    @print_test_separators
    def test_03a_get_params__fetch_dict_with_parameters_hmac(self):
        print("HMAC: Getting optimizer parameters...")

        params = self.local_optimizer_hmac.get_params()
        print(params)

    @print_test_separators
    def test_03b_get_params__fetch_dict_with_parameters_auth_endpoint(self):
        print("AUTH: Getting optimizer parameters...")

        params = self.local_optimizer.get_params()
        print(params)

    @print_test_separators
    def test_04a_fit__not_implemented_hmac(self):
        print("HMAC: Calling fit() method...")

        with self.assertRaises(NotImplementedError, msg="\"fit\" should raise NotImplementedError"):
            self.local_optimizer_hmac.fit(X=self.X, y=self.y)

    @print_test_separators
    def test_04b_fit__not_implemented_auth_endpoint(self):
        print("AUTH: Calling fit() method...")

        with self.assertRaises(NotImplementedError, msg="\"fit\" should raise NotImplementedError"):
            self.local_optimizer.fit(X=self.X, y=self.y)

    @print_test_separators
    def test_05a_summary__return_data_frame_with_summary_hmac(self):
        print("HMAC: Fetching summary of experiment...")

        summary = self.local_optimizer_hmac.summary()
        print(summary)

    @print_test_separators
    def test_05b_summary__return_data_frame_with_summary_auth_endpoint(self):
        print("AUTH: Fetching summary of experiment...")
        summary = self.local_optimizer.summary()
        print(summary)

    @print_test_separators
    def test_06a_get_pipeline_details_hmac(self):
        print("HMAC: Fetching best pipeline details...")

        best_pipeline_details = self.local_optimizer_hmac.get_pipeline_details()
        print(f"best pipeline details: {best_pipeline_details}")

        print("HMAC: Fetching pipeline details...")

        pipeline_details = self.local_optimizer_hmac.get_pipeline_details(pipeline_name=self.pipeline_name)
        print(f"pipeline details: {pipeline_details}")

    @print_test_separators
    def test_06b_get_pipeline_details_auth_endpoint(self):
        print("AUTH: Fetching best pipeline details...")

        best_pipeline_details = self.local_optimizer.get_pipeline_details()
        print(f"best pipeline details: {best_pipeline_details}")

        print("AUTH: Fetching pipeline details...")

        pipeline_details = self.local_optimizer.get_pipeline_details(pipeline_name=self.pipeline_name)
        print(f"pipeline details: {pipeline_details}")

    @print_test_separators
    def test_07a_predict__on_best_pipeline_hmac(self):
        print("HMAC: Calling predict on the best pipeline...")

        predictions = self.local_optimizer_hmac.predict(X=self.X)
        print(predictions)

    @print_test_separators
    def test_07b_predict__on_best_pipeline_auth_endpoint(self):
        print("AUTH: Calling predict on the best pipeline...")

        predictions = self.local_optimizer.predict(X=self.X)
        print(predictions)

    @print_test_separators
    def test_08a_get_data_connections__return_training_data_connection_and_recreate_holdout_split_hmac(self):
        print("HMAC: Reading training data with holdout split...")

        training_df, holdout_df = self.local_optimizer_hmac.get_data_connections()[0].read(with_holdout_split=True)
        print(f"Training data frame: {training_df}")
        print(f"Holdout data frame: {holdout_df}")

    @print_test_separators
    def test_08b_get_data_connections__return_training_data_connection_and_recreate_holdout_split_auth_endpoint(self):
        print("AUTH: Reading training data with holdout split...")

        training_df, holdout_df = self.local_optimizer.get_data_connections()[0].read(with_holdout_split=True)
        print(f"Training data frame: {training_df}")
        print(f"Holdout data frame: {holdout_df}")

    # @print_test_separators
    # def test_09a_deploy_model_from_object_hmac(self):
    #     print("HMAC: Deploying model from object...")
    #
    #     # service = WebService(self.wml_credentials, space_id='4e0541b8-174b-4d04-ae21-14c15475d68a')
    #     service = Batch(self.wml_credentials, space_id='4e0541b8-174b-4d04-ae21-14c15475d68a')
    #     service.create(
    #         model=self.pipeline_model_hmac,
    #         metadata=self.metadata_hmac,
    #         deployment_name="Test deployment from auto-gen notebook and COS"
    #     )
    #
    #     print(service)
    #     print(service.get_params())
    #     predictions = service.score(payload=self.X.iloc[:10])
    #     print(predictions)
    #     print(service.list())
    #     print(service.delete())
    #     print(service.list())
    #
    # @print_test_separators
    # def test_09b_deploy_model_from_object_auth_endpoint(self):
    #     print("AUTH: Deploying model from object...")
    #
    #     # service = WebService(self.wml_credentials, space_id='4e0541b8-174b-4d04-ae21-14c15475d68a')
    #     service = Batch(self.wml_credentials, space_id='4e0541b8-174b-4d04-ae21-14c15475d68a')
    #     service.create(
    #         model=self.pipeline_model_auth_endpoint,
    #         metadata=self.metadata_auth_endpoint,
    #         deployment_name="Test deployment from auto-gen notebook and COS"
    #     )
    #
    #     print(service)
    #     print(service.get_params())
    #     predictions = service.score(payload=self.X.iloc[:10])
    #     print(predictions)
    #     print(service.list())
    #     print(service.delete())
    #     print(service.list())

    @print_test_separators
    def test_10a_deploy_model_from_pipeline_name_hmac(self):
        print("HMAC: Deploying model from pipeline name...")

        # service = WebService(self.wml_credentials, space_id='4e0541b8-174b-4d04-ae21-14c15475d68a')
        service = Batch(self.wml_credentials, space_id='4e0541b8-174b-4d04-ae21-14c15475d68a')
        service.create(
            model=self.pipeline_name,
            metadata=self.metadata_hmac,
            deployment_name="Test deployment from auto-gen notebook and COS"
        )

        print(service)
        print(service.get_params())
        predictions = service.score(payload=self.X.iloc[:10])
        print(predictions)
        print(service.list())
        print(service.delete())
        print(service.list())

    @print_test_separators
    def test_10b_deploy_model_from_pipeline_name_auth_endpoint(self):
        print("AUTH: Deploying model from pipeline name...")

        # service = WebService(self.wml_credentials, space_id='4e0541b8-174b-4d04-ae21-14c15475d68a')
        service = Batch(self.wml_credentials, space_id='4e0541b8-174b-4d04-ae21-14c15475d68a')
        service.create(
            model=self.pipeline_name,
            metadata=self.metadata_auth_endpoint,
            deployment_name="Test deployment from auto-gen notebook and COS"
        )

        print(service)
        print(service.get_params())
        predictions = service.score(payload=self.X.iloc[:10])
        print(predictions)
        print(service.list())
        print(service.delete())
        print(service.list())


if __name__ == '__main__':
    unittest.main()
