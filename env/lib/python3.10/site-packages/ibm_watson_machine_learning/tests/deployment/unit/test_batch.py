#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.deployment import Batch
from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, AssetLocation
from tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d
from sklearn.pipeline import Pipeline
import pandas as pd


class TestBatch(unittest.TestCase):
    pipeline: 'Pipeline' = None
    deployment: 'Batch' = None

    scoring_id: str = None
    # deployment_scikit: 'WebService' = None

    # deployment_id = '11fe2388-7cc8-481a-9db4-c2fb8b14739a'
    # deployment_name = 'autoAI deployment'
    # deployment_scoring_url = 'https://us-south.ml.cloud.ibm.com/v4/deployments/11fe2388-7cc8-481a-9db4-c2fb8b14739a/predictions'

    # #cpd svt17
    # space_id = 'd651b5f4-1fa0-40de-9e80-cebd5098b57a'
    # project_id = '94a6074d-48db-4279-bacb-90cd6f3358c7'

    # cpd qa41
    space_id = 'aa81ee79-795a-4a23-8265-e87966dfe437'
    project_id = '3d177338-3f8b-444d-b385-9ff929fd124f'
    autoai_run_id = '0987e342-2ef7-4cb0-aac2-108b81367db7'

    deployment_id = None

    deployment_name = "Unit-Bank_Batch-Deployment_v2"

    data_location = './autoai/data/bank.csv'

    bucket_name = "wml-autoai-tests"
    data_cos_path = 'data/bank.csv'

    prediction_column = 'y'

    data = None
    X = None

    @classmethod
    def setUp(cls) -> None:
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop([cls.prediction_column], axis=1)
        cls.wml_credentials = get_wml_credentials()
        if not is_cp4d():
            cls.cos_credentials = get_cos_credentials()


    def test__01__initialize__Batch_initialization__object_initialized(self):
        TestBatch.deployment = Batch(self.wml_credentials.copy(),
                                          project_id=self.project_id,
                                          space_id=self.space_id)

        self.assertIsInstance(self.deployment, Batch, msg="Deployment is not of Batch type.")
        self.assertIsInstance(self.deployment._workspace, WorkSpace, msg="Workspace set incorrectly.")
        self.assertEqual(self.deployment.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(self.deployment.name, None, msg="Deployment name initialized incorrectly")

    def test__02__list__listing_deployments__deployments_listed_as_pandas_data_frame(self):
        deployments = self.deployment.list()
        print(deployments)

        self.assertIsInstance(deployments, pd.DataFrame, msg="Deployments are not of pandas DaraFrame type.")

    # def test__03__list__listing_deployments_with_limit__one_deployment_listed_as_pandas_data_frame(self):
    #     deployments = self.deployment.list(limit=1)
    #     print(deployments)
    #
    #     self.assertEqual(len(deployments), 1, msg="There is a different number of deployments than 1.")

    # def test__04__get__fetching_deployment__Batch_object_correctly_populated(self):
    #     self.deployment.get(deployment_id=self.deployment_id)
    #
    #     self.assertEqual(self.deployment.id, self.deployment_id, msg="Deployment ID initialized incorrectly")
    #     self.assertEqual(self.deployment.name, self.deployment_name, msg="Deployment name initialized incorrectly")
    #     # self.assertEqual(self.deployment.scoring_url, self.deployment_scoring_url,
    #     #                  msg="Deployment scoring_url initialized incorrectly")

    def test__05__get_params__fetching_deployment_details__details_fetched(self):
        details = self.deployment.get_params()
        print(details)

        self.assertIsInstance(details, dict, msg="Deployment parameters are not of dict type.")

    def test__06__create__creating_a_deployment__deployment_created(self):
        if not self.deployment_id:
            TestBatch.deployment.create(experiment_run_id=self.autoai_run_id,
                                        model='Pipeline_1',
                                        deployment_name=self.deployment_name)
            self.assertNotEqual(self.deployment.id, None, msg="Deployment ID initialized incorrectly")
            self.assertNotEqual(self.deployment.name, None, msg="Deployment name initialized incorrectly")
        else:
            print('Using existing deployment, id = ', self.deployment_id)
            TestBatch.deployment.get(deployment_id=self.deployment_id)

    # def test__07__score_DataFrame(self):
    #     payload_df = self.X[:5]
    #     scoring_results = self.deployment.score(payload_df, background_mode=False)
    #     print(scoring_results)

    def test__08__score_DataConnection(self):
        payload_ref = [DataConnection(
            connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                    access_key_id=self.cos_credentials['access_key_id'],
                                    secret_access_key=self.cos_credentials['secret_access_key']),
            location=S3Location(bucket=self.bucket_name,
                                path=self.data_cos_path))]

        out_ref = "unit_test_bank_out"

        scoring_results = self.deployment.score(payload=payload_ref, output_data_filename=out_ref)
        print(scoring_results)

        TestBatch.scoring_id = scoring_results['metadata']['id']

    def test__09__get_scoring_params(self):
        job_details = self.deployment.get_scoring_params(self.scoring_id)
        print(job_details)

        self.assertIsInstance(job_details, dict)

    def test__10__get_scoring_status(self):
        job_status = self.deployment.get_scoring_status(self.scoring_id)
        print(job_status)

        self.assertIsInstance(job_status, dict)

    def test__11__get_scoring_result(self):
        result = self.deployment.get_scoring_result(self.scoring_id)
        print(result)
        self.assertIsNone(result)

    def test__12__score_rerun(self):
        scoring_details = self.deployment.score_rerun(self.scoring_id, background_mode=True)

        self.assertIsInstance(scoring_details, dict)

        print(self.deployment.get_scoring_status(scoring_details['metadata']['id']))

    def test__13__list_scoring_jobs(self):
        scoring_jobs_df = self.deployment.list_scorings()
        print(scoring_jobs_df)
        self.assertGreater(len(scoring_jobs_df), 0)

    def test__14__delete__deleting_deployment__deployment_deleted(self):
        print(f"Deleting deployment: {self.deployment}")
        self.deployment.delete()

        self.assertEqual(self.deployment.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(self.deployment.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(self.deployment.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")


if __name__ == '__main__':
    unittest.main()
