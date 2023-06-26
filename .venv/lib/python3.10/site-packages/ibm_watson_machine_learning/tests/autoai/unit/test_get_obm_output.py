#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning.helpers import DataConnection, S3Connection, S3Location
from ibm_watson_machine_learning.tests.utils import get_cos_credentials, get_wml_credentials
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.experiment.autoai.optimizers.local_auto_pipelines import LocalAutoPipelines
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


class TestOBMOutput(unittest.TestCase):
    cos_credentials = None
    bucket = 'test-donotdelete-pr-7w72prz2oteazs'
    path = 'bank.csv'
    pipeline_model_hmac = None
    pipeline_model_auth_endpoint = None
    metadata_hmac = None
    metadata_auth_endpoint = None
    pipeline_name = 'Pipeline_1'
    training_status = 'auto_ml/5e04fc63-906f-46d1-af47-c4d76ba5c42a/wml_data/c8ed6196-6187-4e80-997f-378a1b9b24e2/training-status.json'
    model_location = 'auto_ml/5e04fc63-906f-46d1-af47-c4d76ba5c42a/wml_data/c8ed6196-6187-4e80-997f-378a1b9b24e2/data/kb/global_output/'
    run_id = 'c8ed6196-6187-4e80-997f-378a1b9b24e2'

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
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['y'], axis=1)
        cls.y = cls.data['y']

    def test_01_get_preprocessed_data_connection_with_holdout_split_notebook_COS_version(self):
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

        TestOBMOutput.metadata_hmac = dict(
            training_data_reference=training_data_reference,
            training_result_reference=training_result_reference,
            prediction_type='regression',
            prediction_column='next_purchase',
            test_size=0.1,
            scoring='roc_auc',
            max_number_of_estimators=1,
        )

        print("HMAC: Initializing AutoAI local scenario with metadata...")

        TestOBMOutput.local_optimizer_hmac = AutoAI().runs.get_optimizer(
            metadata=self.metadata_hmac
        )

        training_df, holdout_df = self.local_optimizer_hmac.get_preprocessed_data_connection().read(
            with_holdout_split=True)

        print(f"Training data frame: {training_df}")
        print(f"Holdout data frame: {holdout_df}")

    def test_02_get_preprocessed_data_connection_with_holdout_historical_scenario(self):
        experiment = AutoAI(self.wml_credentials)
        optimizer = experiment.runs.get_optimizer(run_id=self.run_id)

        training_df, holdout_df = optimizer.get_preprocessed_data_connection().read(
            with_holdout_split=True)

        print(f"Training data frame: {training_df}")
        print(f"Holdout data frame: {holdout_df}")


if __name__ == '__main__':
    unittest.main()
