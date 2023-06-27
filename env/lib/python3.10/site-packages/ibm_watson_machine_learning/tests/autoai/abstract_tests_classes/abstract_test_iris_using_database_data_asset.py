#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc

import pandas as pd
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.tests.utils import get_db_credentials
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes.abstract_test_iris_wml_autoai_multiclass_connections import (
                                                AbstractTestAutoAIRemote)


class AbstractTestAutoAIConnectedAsset(AbstractTestAutoAIRemote, abc.ABC):
    """
    The test can be run on CLOUD, and CPD
    The test covers:
    - DataBase connection set-up.
    - downloading training data using created connection.
    - downloading all generated pipelines to lale pipeline.
    - deployment with lale pipeline.
    - deployment deletion.
    Connection used in test:
    - input: Data asset connected to data base table.
    """
    database_name = "db_name"
    schema_name = "tests_sdk"
    table_name = "iris"

    def test_00c_prepare_connection_to_DATABASE(self):
        AbstractTestAutoAIConnectedAsset.db_credentials = get_db_credentials(self.database_name)
        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(self.database_name),
            'name': f'Connection to DB for python API tests - {self.database_name}',
            'properties': self.db_credentials
        })

        AbstractTestAutoAIConnectedAsset.connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.connection_id, str)

    def test_00d_prepare_connected_data_asset(self):
        asset_details = self.wml_client.data_assets.store({
            "connection_id": self.connection_id,
            "name": "Data asset for tests",
            "connectionPath": f"/{self.schema_name}/{self.table_name}",
            "data_content_name": f"/{self.schema_name}/{self.table_name}"
        })

        AbstractTestAutoAIConnectedAsset.asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.asset_id, str)

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        AbstractTestAutoAIConnectedAsset.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                             project_id=self.project_id,
                                                             space_id=self.space_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_DataConnection_setup(self):
        AbstractTestAutoAIConnectedAsset.data_connection = DataConnection(
            data_asset_id=self.asset_id
        )
        AbstractTestAutoAIConnectedAsset.results_connection = None

        self.assertIsNotNone(obj=AbstractTestAutoAIConnectedAsset.data_connection)

    def test_02a_read_saved_remote_data_before_fit(self):
        self.data_connection.set_client(self.wml_client)
        data = self.data_connection.read(raw=True)
        print("data sample:")
        print(data.head())
        self.assertGreater(len(data), 0)

    def test_29_delete_connection_and_connected_data_asset(self):
        self.wml_client.connections.delete(self.connection_id)
        self.wml_client.data_assets.delete(self.asset_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)
            self.wml_client.data_assets.get_details(self.asset_id)