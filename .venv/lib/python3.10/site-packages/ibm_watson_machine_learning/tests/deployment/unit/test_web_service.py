#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning.deployment import WebService
from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.tests.utils import get_wml_credentials
from sklearn.pipeline import Pipeline
import pandas as pd


class TestWebService(unittest.TestCase):
    pipeline: 'Pipeline' = None
    ws: 'WorkSpace' = None
    deployment: 'WebService' = None
    deployment_scikit: 'WebService' = None

    deployment_id = '11fe2388-7cc8-481a-9db4-c2fb8b14739a'
    deployment_name = 'autoAI deployment'
    deployment_scoring_url = 'https://us-south.ml.cloud.ibm.com/v4/deployments/11fe2388-7cc8-481a-9db4-c2fb8b14739a/predictions'

    space_id = None
    project_id = None

    data_location = './autoai/data/bank.csv'
    data = None
    X = None

    @classmethod
    def setUp(cls) -> None:
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['y'], axis=1)
        cls.wml_credentials = get_wml_credentials()
        cls.ws = WorkSpace(wml_credentials=cls.wml_credentials,
                           project_id=cls.project_id,
                           space_id=cls.space_id)

    def test_01__initialize__WebService_initialization__object_initialized(self):
        TestWebService.deployment = WebService(workspace=self.ws)

        self.assertIsInstance(self.deployment, WebService, msg="Deployment is not of WebService type.")
        self.assertIsInstance(self.deployment._workspace, WorkSpace, msg="Workspace set incorrectly.")
        self.assertEqual(self.deployment.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(self.deployment.name, None, msg="Deployment name initialized incorrectly")
        self.assertEqual(self.deployment.scoring_url, None, msg="Deployment scoring_url initialized incorrectly")

    def test__02__list__listing_deployments__deployments_listed_as_pandas_data_frame(self):
        deployments = self.deployment.list()
        print(deployments)

        self.assertIsInstance(deployments, pd.DataFrame, msg="Deployments are not of pandas DaraFrame type.")

    def test__03__list__listing_deployments_with_limit__one_deployment_listed_as_pandas_data_frame(self):
        deployments = self.deployment.list(limit=1)
        print(deployments)

        self.assertEqual(len(deployments), 1, msg="There is a different number of deployments than 1.")

    def test__03__get__fetching_deployment__WebService_object_correctly_populated(self):
        self.deployment.get(deployment_id=self.deployment_id)

        self.assertEqual(self.deployment.id, self.deployment_id, msg="Deployment ID initialized incorrectly")
        self.assertEqual(self.deployment.name, self.deployment_name, msg="Deployment name initialized incorrectly")
        self.assertEqual(self.deployment.scoring_url, self.deployment_scoring_url,
                         msg="Deployment scoring_url initialized incorrectly")

    def test__04__score__make_online_prediction__predictions_returned(self):
        predictions = self.deployment.score(payload=self.X)
        print(predictions)

        self.assertIsInstance(predictions, dict, msg="Predictions are not of List type.")

    def test__05__get_params__fetching_deployment_details__details_fetched(self):
        details = self.deployment.get_params()
        print(details)

        self.assertIsInstance(details, dict, msg="Deployment parameters are not of dict type.")

    def test__06__create__creating_a_deployment__deployment_created(self):
        from sklearn import datasets
        from sklearn import preprocessing
        from sklearn import svm

        model_data = datasets.load_digits()
        scaler = preprocessing.StandardScaler()
        clf = svm.SVC(kernel='rbf', probability=True)
        pipeline = Pipeline([('scaler', scaler), ('svc', clf)])
        model = pipeline.fit(model_data.data, model_data.target)

        model_props = {
            self.deployment._workspace.wml_client.repository.ModelMetaNames.NAME: "test deployment",
            self.deployment._workspace.wml_client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.deployment._workspace.wml_client.repository.ModelMetaNames.RUNTIME_UID: "scikit-learn_0.20-py3.6"
        }

        TestWebService.deployment_scikit = WebService(workspace=self.ws)
        self.deployment_scikit.create(model=model, metadata=model_props, deployment_name="test deployment with scikit",
                                      training_data=model_data.data, training_target=model_data.target)

        self.assertNotEqual(self.deployment_scikit.id, None, msg="Deployment ID initialized incorrectly")
        self.assertNotEqual(self.deployment_scikit.name, None, msg="Deployment name initialized incorrectly")
        self.assertNotEqual(self.deployment_scikit.scoring_url, None,
                            msg="Deployment scoring_url initialized incorrectly")

    def test__07__delete__deleting_deployment__deployment_deleted(self):
        print(f"Deleting deployment: {self.deployment_scikit}")
        self.deployment_scikit.delete()

        self.assertEqual(self.deployment_scikit.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(self.deployment_scikit.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(self.deployment_scikit.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")


if __name__ == '__main__':
    unittest.main()
