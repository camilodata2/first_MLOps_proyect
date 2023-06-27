#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from pprint import pprint

import pandas as pd

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.deployment import WebService
from ibm_watson_machine_learning.experiment.autoai.runs import AutoPipelinesRuns
from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.tests.utils import get_wml_credentials
from sklearn.pipeline import Pipeline


class TestAutoPipelinesRuns(unittest.TestCase):
    experiment: 'AutoAI' = None
    optimizer_runs: 'AutoPipelinesRuns' = None
    optimizer: 'RemoteAutoPipelines' = None
    pipeline: 'Pipeline' = None
    deployed_pipeline = None
    ws: 'WorkSpace' = None
    service: 'WebService' = None

    space_id = None
    project_id = None

    run_id = '08754ca1-e0fd-4d69-9319-eb3dbbc28757'
    data_location = './autoai/data/bank.csv'
    data = None
    X = None
    y = None

    @classmethod
    def setUp(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['y'], axis=1)
        cls.y = cls.data['y']
        cls.wml_credentials = get_wml_credentials()

        cls.ws = WorkSpace(wml_credentials=cls.wml_credentials,
                           project_id=cls.project_id,
                           space_id=cls.space_id)

    def test_01__initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        TestAutoPipelinesRuns.experiment = AutoAI(self.ws)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02__check_AutoPipelinesRuns__historical_runs_in_AutoAI__runs_initialized(self):
        self.assertIsInstance(self.experiment.runs, AutoPipelinesRuns,
                              msg="Experiment not initialized AutoPipelinesRuns (historical runs) correctly.")

    def test_03__runs_list__list_runs__runs_listed(self):
        runs = self.experiment.runs.list()
        print(runs)

        self.assertIsInstance(runs, pd.DataFrame, msg="Listed runs are not in the format of pandas DataFrame")

    def test_04__runs_list__list_filtered_runs__runs_listed(self):
        runs = self.experiment.runs(filter="test name").list()
        print(runs)

        self.assertIsInstance(runs, pd.DataFrame, msg="Listed runs are not in the format of pandas DataFrame")

    def test_05__get_params__get_configuration_parameters_of_remote_auto_pipeline__parameters_listed(self):
        parameters = self.experiment.runs.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_06__get_run_details__fetching_run_parameters__run_parameters_fetched(self):
        parameters = self.experiment.runs.get_run_details()
        pprint(parameters)

    def test_07__get_run_details__fetching_run_parameters_for_run_id__run_parameters_fetched(self):
        parameters = self.experiment.runs.get_run_details(run_id=self.run_id)
        pprint(parameters)

    def test_08__get_data_connections__return_list_of_data_connection_objects_from_run(self):
        data_connections = self.experiment.runs.get_data_connections(run_id=self.run_id)
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    def test_09__get_optimizer__get_RemoteAutoPipelines_object__new_object_returned(self):
        TestAutoPipelinesRuns.optimizer = self.experiment.runs.get_optimizer(run_id=self.run_id)

        print(type(self.optimizer))
        self.assertIsInstance(self.optimizer, RemoteAutoPipelines)

    def test_10__run_status__get_status_of_a_run__status_fetched(self):
        status = self.optimizer.get_run_status()
        print(status)

    def test_11__results__listing_all_pipelines_from_wml__all_pipelines_listed(self):
        pipelines_details = self.optimizer.summary()
        print(pipelines_details)

    def test_12__get_pipeline_params__fetch_specific_pipeline_parameters__parameters_fetched_as_dict(self):
        pipeline_params = self.optimizer.get_pipeline_details(pipeline_name='Pipeline_1')
        print(pipeline_params)

    def test_13__get_pipeline_params__fetch_best_pipeline_parameters__parameters_fetched_as_dict(self):
        best_pipeline_params = self.optimizer.get_pipeline_details()
        print(best_pipeline_params)

    def test_14__get_pipeline__load_specified_pipeline__pipeline_loaded(self):
        TestAutoPipelinesRuns.pipeline = self.optimizer.get_pipeline(pipeline_name='Pipeline_4',
                                                                     astype=AutoAI.PipelineTypes.SKLEARN)
        print(f"Fetched pipeline type: {type(self.pipeline)}")

        self.assertIsInstance(self.pipeline, Pipeline, msg="Fetched pipeline is not of sklearn type.")

    def test_15__get_pipeline_and_persist_it(self):
        try:
            from sklearn.externals import joblib
        except ImportError:
            import joblib

        pipeline = self.optimizer.get_pipeline(pipeline_name="Pipeline_1",
                                               astype=AutoAI.PipelineTypes.SKLEARN,
                                               persist=True)

        exception = None
        try:
            model = joblib.load('./Pipeline_1.pickle')

        except Exception as exception:
            self.assertIsNone(exception, msg=f"Cannot load saved pipeline, Error: {exception}")

    def test_16__deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
        TestAutoPipelinesRuns.service = WebService(self.ws)

        self.service.create(
            experiment_run_id=self.run_id,
            model=self.pipeline,
            deployment_name='autoAI deployment')

    def test_17__predict__score_deployment(self):
        results = self.service.score(payload=self.X.iloc[:10])
        print(results)


if __name__ == '__main__':
    unittest.main()
