#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import pandas as pd

from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment.autoai.engines import WMLEngine
from ibm_watson_machine_learning.helpers import S3Connection, S3Location, DataConnection
from ibm_watson_machine_learning.utils.autoai.errors import HoldoutSplitNotSupported
from ibm_watson_machine_learning.experiment.autoai.runs import AutoPipelinesRuns


class TestDataConnection(unittest.TestCase):
    data_location = './autoai/data/bank.csv'
    data = None
    bank_connection: 'DataConnection' = None
    wml_credentials_cp4d = None
    csv_with_different_delimeter_location = './autoai/data/datasets_different_delimeter/iris_dataset_different_delimeter.csv'

    @classmethod
    def setUp(cls) -> None:
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['y'], axis=1)
        cls.y = cls.data['y']
        cls.wml_credentials = get_wml_credentials('CLOUD_PROD_AM')
        cls.wml_credentials_cp4d = get_wml_credentials('CLOUD_DEV_AM')
        cls.cos_credentials = get_cos_credentials('CLOUD_PROD_AM')

    def test_01_initialize_DataConnection__object_initialized(self):
        TestDataConnection.bank_connection = DataConnection(
            connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                    access_key_id=self.cos_credentials['access_key_id'],
                                    secret_access_key=self.cos_credentials['secret_access_key']),
            location=S3Location(bucket='automlnew',
                                path='bank.csv')
        )
        self.assertEqual(self.bank_connection.connection.endpoint_url, self.cos_credentials['endpoint_url'],
                         msg="Wrong endpoint_url stored.")
        self.assertEqual(self.bank_connection.connection.access_key_id, self.cos_credentials['access_key_id'],
                         msg="Wrong access_key_id stored.")
        self.assertEqual(self.bank_connection.connection.secret_access_key, self.cos_credentials['secret_access_key'],
                         msg="Wrong secret_access_key stored.")
        self.assertEqual(self.bank_connection.location.bucket, 'automlnew', msg="Wrong bucket stored.")
        self.assertEqual(self.bank_connection.location.path, 'bank.csv', msg="Wrong path stored.")

        print(self.bank_connection)

    def test_02_read__download_dataset_and_load_from_COS(self):
        data = self.bank_connection.read()
        print(data)

    def test_03_read_logs__download_logs_from_COS(self):
        wml_client = APIClient(self.wml_credentials)
        historical_opt = AutoPipelinesRuns(WMLEngine(wml_client=wml_client))
        data_connection = historical_opt.get_data_connections(run_id='b1308028-8bfd-44af-88d8-693697fea4bb')[0]
        data = data_connection.read_logs()
        print(data)

        self.assertIsInstance(data, str)

    def test_04_read_logs__download_logs_from_filesystem(self):
        wml_client = APIClient(self.wml_credentials_cp4d,
                                                    project_id='dd829201-9d59-4f5a-b0e0-6ea3a88ae66b')
        historical_opt = AutoPipelinesRuns(WMLEngine(wml_client=wml_client))
        data_connection = historical_opt.get_data_connections(run_id='165ff73b-77cb-49b4-aee4-4d4b87dcad37')[0]
        data = data_connection.read_logs()
        print(data)

        self.assertIsInstance(data, str)

    def test_05_read__download_dataset_stored_on_COS_with_holdout_split_recreation(self):
        auto_pipelines_params = {
            'name': 'test name',
            'desc': 'test description',
            'prediction_type': 'classification',
            'prediction_column': 'y',
            'scoring': 'roc_auc',
            'test_size': 0.1,
            'max_num_daub_ensembles': 1
        }
        self.bank_connection.auto_pipeline_params = auto_pipelines_params
        train_data, holdout_data = self.bank_connection.read(with_holdout_split=True)
        print(holdout_data)

    def test_06__write__upload_dataset_to_COS(self):
        self.bank_connection.write(data=self.data_location, remote_name='bank_test.csv')
        bank_test_connection = DataConnection(
            connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                    access_key_id=self.cos_credentials['access_key_id'],
                                    secret_access_key=self.cos_credentials['secret_access_key']),
            location=S3Location(bucket='automlnew',
                                path='bank_test.csv')
        )
        test_data = bank_test_connection.read()
        print(test_data)

    def test_07__write__upload_dataset_to_COS_from_pandas_DataFrame(self):
        self.bank_connection.write(data=self.data, remote_name='test_data.csv')

        bank_test_connection = DataConnection(
            connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                    access_key_id=self.cos_credentials['access_key_id'],
                                    secret_access_key=self.cos_credentials['secret_access_key']),
            location=S3Location(bucket='automlnew',
                                path='test_data.csv')
        )
        test_data = bank_test_connection.read()
        print(test_data)

    def test_08_read__download_xlsx_dataset_stored_on_COS(self):
        test_connection = DataConnection(
            connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                    access_key_id=self.cos_credentials['access_key_id'],
                                    secret_access_key=self.cos_credentials['secret_access_key']),
            location=S3Location(bucket='automlnew',
                                path='test_data.xlsx')
        )

        train_data = test_connection.read()
        print(train_data)

    def test_09_read__download_xlsx_dataset_stored_on_COS_with_holdout_split(self):
        test_connection = DataConnection(
            connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                    access_key_id=self.cos_credentials['access_key_id'],
                                    secret_access_key=self.cos_credentials['secret_access_key']),
            location=S3Location(bucket='automlnew',
                                path='test_data.xlsx')
        )
        auto_pipelines_params = {
            'name': 'test name',
            'desc': 'test description',
            'prediction_type': 'classification',
            'prediction_column': 'y',
            'scoring': 'roc_auc',
            'test_size': 0.1,
            'max_num_daub_ensembles': 1
        }
        test_connection.auto_pipeline_params = auto_pipelines_params

        with self.assertRaises(HoldoutSplitNotSupported):
            train_data = test_connection.read(with_holdout_split=True)

    def test_10__read__download_csv_file_with_not_standard_delimeter__return_as_pandas_DataFrame(self):
        test_connection = DataConnection(
            connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                    access_key_id=self.cos_credentials['access_key_id'],
                                    secret_access_key=self.cos_credentials['secret_access_key']),
            location=S3Location(bucket='automlnew',
                                path='iris_dataset_different_delimeter.csv')
        )

        test_connection.write(data=self.csv_with_different_delimeter_location,
                              remote_name='iris_dataset_different_delimeter.csv')

        test_data = test_connection.read()
        print(test_data)

        self.assertGreater(len(test_data.columns), 2,
                           msg="Wrong columns number, data separator probably is wrongly interpreted.")


if __name__ == '__main__':
    unittest.main()
