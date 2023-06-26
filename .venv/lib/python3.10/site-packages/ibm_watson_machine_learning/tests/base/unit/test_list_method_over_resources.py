#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import abc
from sys import getsizeof
from os import environ
import pandas as pd
import pprint

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, is_cp4d


class TestListMethod(unittest.TestCase):
    wml_client: 'APIClient' = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

        cls.project_id = cls.wml_credentials.get('project_id')
        cls.wml_client.set.default_project(cls.project_id)

    def general_test(self, list_method, **kwargs):
        output = list_method(**kwargs)

        print(f"Output: {output}")
        self.assertIsInstance(output, pd.DataFrame, msg=f"DataFrame was not returned for method: {list_method}")

        output = list_method(return_as_df=False, **kwargs)

        print(f"Output: {output}")
        self.assertEqual(output, None, msg=f"None was not returned for method: {list_method}")

    def test_connections(self):
        self.general_test(self.wml_client.connections.list)
        self.general_test(self.wml_client.connections.list_datasource_types)
        if is_cp4d():
            self.general_test(self.wml_client.connections.list_uploaded_db_drivers)

    def test_deployments(self):
        space_details = self.wml_client.spaces.get_details(limit=1)
        space_id = self.wml_client.spaces.get_uid(space_details['resources'][0])
        self.wml_client.set.default_space(space_id)
        self.general_test(self.wml_client.deployments.list)

    def test_experiments(self):
        self.wml_client.set.default_project(self.project_id)
        self.general_test(self.wml_client.experiments.list)
        experiment_id = self.wml_client.experiments.get_id(
            self.wml_client.experiments.get_details(limit=1)['resources'][0])
        self.general_test(self.wml_client.experiments.list_revisions, experiment_uid=experiment_id)

    def test_export_assets(self):
        self.general_test(self.wml_client.export_assets.list, project_id=self.project_id)

    @unittest.skipIf(is_cp4d(), "Not supported on CP4D")
    def test_factsheets(self):
        output = self.wml_client.factsheets.list_model_entries()
        print(f"Output: {output}")
        self.assertIsInstance(output, dict)

    def test_hardware_specifications(self):
        self.general_test(self.wml_client.hardware_specifications.list)

    def test_import_assets(self):
        self.general_test(self.wml_client.import_assets.list, project_id=self.project_id)

    def test_model_definitions(self):
        self.general_test(self.wml_client.model_definitions.list)

    def test_pipelines(self):
        self.general_test(self.wml_client.pipelines.list)

    def test_repository_models(self):
        self.general_test(self.wml_client.repository.list_models)
        model_id = self.wml_client.repository.get_model_id(
            self.wml_client.repository.get_model_details(limit=1)['resources'][0])
        self.general_test(self.wml_client.repository.list_models_revisions, model_uid=model_id)

    def test_repository_functions(self):
        self.general_test(self.wml_client.repository.list_functions)
        function_id = self.wml_client.repository.get_function_id(
            self.wml_client.repository.get_function_details(limit=1)['resources'][0])
        self.general_test(self.wml_client.repository.list_functions_revisions, function_uid=function_id)

    def test_repository_experiments(self):
        self.general_test(self.wml_client.repository.list_experiments)
        experiment_id = self.wml_client.experiments.get_id(
            self.wml_client.repository.get_experiment_details(limit=1)['resources'][0])
        self.general_test(self.wml_client.repository.list_experiments_revisions, experiment_uid=experiment_id)

    def test_repository_pipelines(self):
        self.general_test(self.wml_client.repository.list_pipelines)
        pipeline_id = self.wml_client.repository.get_pipeline_id(
            self.wml_client.repository.get_pipeline_details(limit=1)['resources'][0])
        self.general_test(self.wml_client.repository.list_pipelines_revisions, pipeline_uid=pipeline_id)

    # def test_runtimes(self):
    #     self.general_test(self.wml_client.runtimes.list)
    #     self.general_test(self.wml_client.runtimes.list_libraries)
    #     runtime_id = self.wml_client.runtimes.get_details()
    #     self.general_test(self.wml_client.runtimes.list_libraries, {"runtime_uid"})

    def test_script(self):
        self.general_test(self.wml_client.script.list)
        print("space",  self.wml_client.default_space_id)
        print("project",  self.wml_client.default_project_id)
        script_id = self.wml_client.script.get_id(self.wml_client.script.get_details()['resources'][0])
        self.general_test(self.wml_client.script.list_revisions, script_uid=script_id)

    @unittest.skipIf(not is_cp4d(), "Not supported on Cloud")
    def test_shiny(self):
        self.general_test(self.wml_client.shiny.list)

    def test_spaces(self):
        self.general_test(self.wml_client.spaces.list)

    def test_software_specifications(self):
        self.general_test(self.wml_client.software_specifications.list)

    @unittest.skipIf(is_cp4d(), "Not supported on CP4D")
    def test_task_credentials(self):
        self.general_test(self.wml_client.task_credentials.list)

    def test_training(self):
        self.general_test(self.wml_client.training.list)

    @unittest.skipIf(not is_cp4d(), "Not supported on Cloud")
    def test_volumes(self):
        self.general_test(self.wml_client.volumes.list)
