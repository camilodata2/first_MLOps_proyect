#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import json
from configparser import ConfigParser
from os import getenv

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.helpers.connections import DataConnection, DatabaseLocation
from ibm_watson_machine_learning.tests.utils import is_cp4d, get_env
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import (
    AbstractTestAutoAIRemote)

from ibm_watson_machine_learning.wml_client_error import WMLClientError

configDir = "./config.ini"

config = ConfigParser()
config.read(configDir)


@unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "Postgres not supported on FIPS clusters")
@unittest.skipIf(not is_cp4d(), "JDBC is not tested on cloud yet.")
class TestAutoAIRemote(AbstractTestAutoAIRemote, unittest.TestCase):
    """
    The test can be run on CPD
    The test covers:
    - JDBC Postgres connection set-up
    - downloading training data from connection
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline
    - deployment deletion
    Connection used in test:
     - input: Database connection pointing to Postgres.
     - output: null
    """
    prediction_column = 'species'
    connection_asset_details = json.loads(config.get(get_env(), 'jdbc_postgres_connection_asset_details'))

    def test_00c_prepare_connection_to_postgres(self):
        driver_file_path = self.connection_asset_details['driver_file_path']
        driver_file_name = driver_file_path.split('/')[-1]

        try:  # some users cannot upload db drivers
            self.wml_client.connections.upload_db_driver(driver_file_path)
        except:
            pass
        self.wml_client.connections.list_uploaded_db_drivers()
        try:  # TODO only works with new api
            print(self.wml_client.connections.get_uploaded_db_drivers())
        except:
            pass

        properties = self.connection_asset_details['properties']
        # deprecated
        properties['jar_uris'] = self.wml_client.connections.sign_db_driver_url(driver_file_name)
        try:  # TODO only works with new api
            properties['jar_uris'] = self.wml_client.connections.get_db_driver_url(driver_file_name)
        except:
            pass
        print(properties)

        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(
                self.connection_asset_details['datasource_type_name']
            ),
            'name': 'Connection to Postgres for tests',
            'properties': properties
        })

        TestAutoAIRemote.connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.connection_id, str)

    def test_02_DataConnection_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=DatabaseLocation(
                schema_name=self.connection_asset_details['schema_name'],
                table_name=self.connection_asset_details['table_name']
            )
        )
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        #self.assertIsNotNone(obj=TestAutoAIRemote.results_connection)

    def test_02a_read_saved_remote_data_before_fit(self):
        TestAutoAIRemote.data = self.data_connection.read()
        print("Data sample:")
        print(self.data.head())
        self.assertGreater(len(self.data), 0)

    def test_03_initialize_optimizer(self):
        AbstractTestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name='Iris - AutoAI',
            prediction_type=self.experiment.PredictionType.MULTICLASS,
            prediction_column=self.prediction_column,
            scoring=self.experiment.Metrics.ACCURACY_SCORE,
            drop_duplicates=False,
            enable_all_data_sources=True,

        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_29_delete_connection_and_connected_data_asset(self):
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
