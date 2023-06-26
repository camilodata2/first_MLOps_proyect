#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning.helpers import DataConnection, CP4DAssetLocation, FSLocation, ConnectionAsset, ConnectionAssetLocation
from ibm_watson_machine_learning.tests.utils import get_cos_credentials, get_wml_credentials, print_test_separators
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.experiment.autoai.optimizers.local_auto_pipelines import LocalAutoPipelines
from lale.operators import TrainedPipeline
from ibm_watson_machine_learning.deployment import WebService
from ibm_watson_machine_learning.utils.autoai.enums import PipelineTypes
from sklearn.pipeline import Pipeline
import pandas as pd


class TestLocalOptimizerCPD(unittest.TestCase):
    path = 'bank.csv'
    pipeline_model_auth_endpoint = None
    metadata = None
    metadata_auth_endpoint = None
    pipeline_name = 'Pipeline_1'
    # training_status = 'auto_ml/4f4da88c-3a16-4450-9d3c-7bb3017164ae/wml_data/cf660105-7d43-480c-8ea1-b3d532386ab5/training-status.json'
    training_status = 'auto_ml/auto_ml.4392c2a1-4270-4b01-84f1-80a9b598e5a3/wml_data/00727c93-9fd3-4b5f-bc1a-8223ef4ac1ac/training-status.json'
    # model_location = 'auto_ml/4f4da88c-3a16-4450-9d3c-7bb3017164ae/wml_data/cf660105-7d43-480c-8ea1-b3d532386ab5/data/automl/global_output/Pipeline0/model.pickle'
    model_location = 'auto_ml/auto_ml.4392c2a1-4270-4b01-84f1-80a9b598e5a3/wml_data/00727c93-9fd3-4b5f-bc1a-8223ef4ac1ac/data/automl/global_output/Pipeline0/model.pickle'

    local_optimizer: 'LocalAutoPipelines' = None
    connection_id = '66d1d6c9-aa49-4583-b020-f1d7eaf416c1'
    project_id = 'acdee0a5-104d-497e-b6bd-d37601e072f4'
    space_id = 'c6f02cdc-ca64-4285-b248-b5ad788be456'

    data_location = './autoai/data/bank.csv'
    data = None
    X = None
    y = None

    @classmethod
    def setUp(cls) -> None:
        cls.wml_credentials = get_wml_credentials()
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['y'], axis=1)
        cls.y = cls.data['y']

    @print_test_separators
    def test_01a_get_optimizer(self):
        print("Initializing DataConnections...")

        # training_data_reference = [DataConnection(
        #     location=CP4DAssetLocation(asset_id=self.asset_id)
        # )]

        training_data_reference = [DataConnection(
            connection=ConnectionAsset(connection_id='66d1d6c9-aa49-4583-b020-f1d7eaf416c1'),
            location=ConnectionAssetLocation(bucket='somebucket2', file_name='bank.csv'))]

        training_result_reference = DataConnection(
            location=FSLocation(
                # path='projects/4ae1998c-87be-49df-9baf-c79ce3edbd4b/assets/auto_ml/auto_ml.190f5e37-59f3-4d95-9749-a49b6caf6da8/wml_data/4667839f-ae61-419d-a42c-adb6b70df380/data/automl',
                path='/projects/acdee0a5-104d-497e-b6bd-d37601e072f4/assets/auto_ml/auto_ml.4392c2a1-4270-4b01-84f1-80a9b598e5a3/wml_data/00727c93-9fd3-4b5f-bc1a-8223ef4ac1ac/data/automl',
            ))

        TestLocalOptimizerCPD.metadata = dict(
            training_data_reference=training_data_reference,
            training_result_reference=training_result_reference,
            prediction_type='classification',
            prediction_column='y',
            test_size=0.2,
            scoring='roc_auc',
            max_number_of_estimators=1,
        )

        print("Initializing AutoAI local scenario with metadata...")

        TestLocalOptimizerCPD.local_optimizer = AutoAI(
            wml_credentials=self.wml_credentials,
            project_id=self.project_id
        ).runs.get_optimizer(
            metadata=self.metadata
        )

    @print_test_separators
    def test_02_get_pipeline(self):
        print("AUTH: Fetching and store pipeline by name, LALE...")

        pipeline_lale = self.local_optimizer.get_pipeline(pipeline_name=self.pipeline_name, persist=True)
        self.assertIsInstance(pipeline_lale, TrainedPipeline, msg="Loaded model should be of type TrainedPipeline")

        print("AUTH: Fetching pipeline by name, SKLEARN...")

        pipeline_skl = self.local_optimizer.get_pipeline(pipeline_name=self.pipeline_name, astype=PipelineTypes.SKLEARN)
        self.assertIsInstance(pipeline_skl, Pipeline, msg="Loaded model should be of type SKLEARN")

        print("AUTH: Fetching best pipeline, LALE...")

        try:
            TestLocalOptimizerCPD.pipeline_model_auth_endpoint = self.local_optimizer.get_pipeline()
        except Exception as e:
            self.assertIsInstance(self.pipeline_model_auth_endpoint, TrainedPipeline,
                                  msg=f"Cannot load best pipeline model, Error: {e}")

    @print_test_separators
    def test_03_get_params__fetch_dict_with_parameters(self):
        print("AUTH: Getting optimizer parameters...")

        params = self.local_optimizer.get_params()
        print(params)

    @print_test_separators
    def test_04_fit__not_implemented(self):
        print("AUTH: Calling fit() method...")

        with self.assertRaises(NotImplementedError, msg="\"fit\" should raise NotImplementedError"):
            self.local_optimizer.fit(X=self.X, y=self.y)

    @print_test_separators
    def test_05_summary__return_data_frame_with_summary(self):
        print("AUTH: Fetching summary of experiment...")
        summary = self.local_optimizer.summary()
        print(summary)

    @print_test_separators
    def test_06_get_pipeline_details(self):
        print("AUTH: Fetching best pipeline details...")

        best_pipeline_details = self.local_optimizer.get_pipeline_details()
        print(f"best pipeline details: {best_pipeline_details}")

        print("AUTH: Fetching pipeline details...")

        pipeline_details = self.local_optimizer.get_pipeline_details(pipeline_name=self.pipeline_name)
        print(f"pipeline details: {pipeline_details}")

    @print_test_separators
    def test_07_predict__on_best_pipeline(self):
        print("AUTH: Calling predict on the best pipeline...")

        predictions = self.local_optimizer.predict(X=self.X)
        print(predictions)

    @print_test_separators
    def test_08_get_data_connections__return_training_data_connection_and_recreate_holdout_split(self):
        print("AUTH: Reading training data with holdout split...")

        training_df, holdout_df = self.local_optimizer.get_data_connections()[0].read(with_holdout_split=True)
        print(f"Training data frame: {training_df}")
        print(f"Holdout data frame: {holdout_df}")

    @print_test_separators
    def test_09_deploy_model_from_object(self):
        print("AUTH: Deploying model from object...")

        service = WebService(
            source_wml_credentials=self.wml_credentials,
            source_project_id=self.project_id,
            target_wml_credentials=self.wml_credentials,
            target_space_id=self.space_id)

        service.create(
            model=self.pipeline_model_auth_endpoint,
            metadata=self.metadata,
            deployment_name="Test deployment from auto-gen notebook"
        )

        print(service)
        print(service.get_params())
        predictions = service.score(payload=self.X.iloc[:10])
        print(predictions)
        print(service.list())
        print(service.delete())
        print(service.list())

    @print_test_separators
    def test_10_deploy_model_from_pipeline_name(self):
        print("AUTH: Deploying model from pipeline name...")

        service = WebService(
            source_wml_credentials=self.wml_credentials,
            source_project_id=self.project_id,
            target_wml_credentials=self.wml_credentials,
            target_space_id=self.space_id)

        service.create(
            model=self.pipeline_name,
            metadata=self.metadata,
            deployment_name="Test deployment from auto-gen notebookS"
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
