#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from os import getenv
from ibm_watson_machine_learning.tests.utils import is_cp4d
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import \
    AbstractTestAutoAIDatabaseConnection


@unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "SQL Server not supported on FIPS clusters")
@unittest.skipIf(not is_cp4d(), "Not supported on Cloud")
class TestAutoAIMSSQLServer(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "sqlserver"
    schema_name = "connections"
    max_connection_nb = None


@unittest.skipIf(not is_cp4d(), "Not supported on Cloud")
class TestAutoAIDB2(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "db2cloud"
    schema_name = "LWH10123"
    table_name = "IRIS"
    test_table_name = "IRIS_HOLDOUT"
    TEST_DATA = True
    prediction_column = "SPECIES"
    max_connection_nb = 2


@unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "PostgresSQL not supported on FIPS clusters")
@unittest.skipIf(not is_cp4d(), "Not supported on Cloud")
class TestAutoAIPostgresSQL(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "postgresql"
    schema_name = "public"
    max_connection_nb = None


@unittest.skipIf(not is_cp4d(), "Not supported on Cloud")
@unittest.skip("The writing of training data is broken for now.")
class TestAutoAIMySQL(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "mysql"
    schema_name = "mysql"
    max_connection_nb = 15


@unittest.skipIf(not is_cp4d(), "Not supported on Cloud")
class TestAutoAIExasol(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "exasol"
    schema_name = "EXA_SVT"
    table_name = "IRIS"
    prediction_column = "species"
    max_connection_nb = 4

    def test_00c_prepare_connection_to_DATABASE(self):
        from tests.utils import get_db_credentials
        import os

        TestAutoAIExasol.db_credentials = get_db_credentials(self.database_name)

        driver_file_path = os.path.join(os.getcwd(), "autoai", "db_driver_jars", "exajdbc-7.1.4.jar")
        driver_file_name = driver_file_path.split('/')[-1]

        self.wml_client.connections.upload_db_driver(driver_file_path)
        self.wml_client.connections.list_uploaded_db_drivers()

        TestAutoAIExasol.db_credentials['jar_uris'] = self.wml_client.connections.sign_db_driver_url(driver_file_name)

        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(self.database_name),
            'name': 'Connection to DB for python API tests',
            'properties': self.db_credentials
        })

        TestAutoAIExasol.connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.connection_id, str)


@unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "DataStax not supported on FIPS clusters")
@unittest.skipIf(not is_cp4d(), "Not supported on Cloud")
@unittest.skip("Not ready - missing valid credentials details")
class TestAutoAIDataStax(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = 'datastax-ibmcloud'
    schema_name = "tm_wml_kb_rw"  # keyspace
    table_name = "IRIS4"
    prediction_column = "species"
    data_location = './autoai/data/iris_dataset_index.csv'
    max_connection_nb = 2


@unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "BigQuery not supported on FIPS clusters")
@unittest.skip("Not ready - missing credentials details")
class TestAutoAIBigQuery(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "bigquery"
    schema_name = "autoai"
    max_connection_nb = None


@unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "Oracle not supported on FIPS clusters")
class TestAutoAIOracle(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "oracle"
    schema_name = "TESTUSER"
    table_name = "IRIS"
    max_connection_nb = None


@unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "Teradata not supported on FIPS clusters")
class TestAutoAITeradata(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "teradata"
    schema_name = "conndb"
    table_name = "IRIS"
    max_connection_nb = None


class TestAutoAIGenericS3(AbstractTestAutoAIDatabaseConnection, unittest.TestCase):
    database_name = "generics3"
    bucket_name = 'test-data-connections-2022-02-22-bpbuiaqq'
    data_cos_path = 'iris_dataset_train.csv'

    max_connection_nb = None

    def test_02_DataConnection_setup(self):
        from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location
        import pandas as pd

        AbstractTestAutoAIDatabaseConnection.data_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.data_cos_path
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


if __name__ == "__main__":
    unittest.main()
