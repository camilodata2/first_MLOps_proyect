#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc

import pandas as pd
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import DataConnection, DatabaseLocation
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes.abstract_test_iris_wml_autoai_multiclass_connections import (
    AbstractTestAutoAIRemote)
from ibm_watson_machine_learning.tests.utils import get_db_credentials
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError


class AbstractTestAutoAIDatabaseConnection(AbstractTestAutoAIRemote, abc.ABC):
    """
    The test can be run on CLOUD, and CPD
    The test covers:
    - DataBase connection set-up.
    - Saving data to a database.
    - downloading training data using created connection.
    - downloading all generated pipelines to lale pipeline.
    - deployment with lale pipeline.
    - deployment deletion.
    Connection used in test:
    - input: Connection to database.
    """
    database_name = "db_name"
    schema_name = "tests_sdk"
    table_name = "iris"

    data_location = './autoai/data/iris_dataset_train_09.csv'
    test_data_location = './autoai/data/iris_dataset_test_01.csv'
    TEST_DATA = False
    test_table_name = "iris_holdout"
    data = None

    def test_00c_prepare_connection_to_DATABASE(self):
        # cleaning variables reused between tests
        AbstractTestAutoAIDatabaseConnection.data_connection = None
        AbstractTestAutoAIDatabaseConnection.test_data_connection = None
        AbstractTestAutoAIDatabaseConnection.results_connection = None

        AbstractTestAutoAIDatabaseConnection.db_credentials = get_db_credentials(self.database_name)
        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(self.database_name),
            'name': f'Connection to DB for python API tests - {self.database_name}',
            'properties': self.db_credentials
        })

        AbstractTestAutoAIDatabaseConnection.connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.connection_id, str)

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        AbstractTestAutoAIDatabaseConnection.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                                 project_id=self.project_id,
                                                                 space_id=self.space_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_DataConnection_setup(self):
        AbstractTestAutoAIDatabaseConnection.data_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=DatabaseLocation(
                schema_name=self.schema_name,
                table_name=self.table_name
            )
        )
        AbstractTestAutoAIDatabaseConnection.results_connection = None

        self.assertIsNotNone(obj=AbstractTestAutoAIDatabaseConnection.data_connection)

        self.data_connection.set_client(self.wml_client)

        try:
            AbstractTestAutoAIDatabaseConnection.data = self.data_connection.read()
            print("Data sample:")
            print(self.data.head())
            self.assertGreater(len(self.data), 0)

        except Exception as e:
            print(e)
            print("Writing data to Database")
            data_df = pd.read_csv(self.data_location)
            self.data_connection.write(data_df)

    def test_02a_test_data_connection_setup(self):
        if self.TEST_DATA:
            AbstractTestAutoAIDatabaseConnection.test_data_connection = DataConnection(
                connection_asset_id=self.connection_id,
                location=DatabaseLocation(
                    schema_name=self.schema_name,
                    table_name=self.test_table_name
                )
            )

            self.assertIsNotNone(obj=AbstractTestAutoAIDatabaseConnection.test_data_connection)

            self.test_data_connection.set_client(self.wml_client)

            try:
                AbstractTestAutoAIDatabaseConnection.test_data = self.test_data_connection.read()
                print("Data sample:")
                print(self.test_data.head())
                self.assertGreater(len(self.test_data), 0)

            except Exception as e:
                print(e)
                print("Writing data to Database")
                data_df = pd.read_csv(self.test_data_location)
                self.data_connection.write(data_df)
        else:
            AbstractTestAutoAIDatabaseConnection.test_data_connection = None
            self.skipTest("Skipping test data setup.")

    def test_02b_read_saved_remote_data_before_fit(self):
        if self.data is None:
            AbstractTestAutoAIDatabaseConnection.data = self.data_connection.read()
            print("Data sample:")
            print(self.data.head())
            self.assertGreater(len(self.data), 0)
        else:
            print("Data have been downloaded in test 02")
            pass

        self.assertGreater(len(self.data), 100) #100 is minimum lenght of the dataset to start autoai experiment


    def test_29_delete_connection_and_connected_data_asset(self):
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)
