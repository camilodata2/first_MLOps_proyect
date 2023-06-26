#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import sys
import unittest
from os import environ, getenv

from os.path import join

import pandas

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers.connections import DataConnection, ContainerLocation
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import create_connection_to_cos, get_wml_credentials, get_cos_credentials, \
    get_space_id
from ibm_watson_machine_learning.data_loaders.experiment import ExperimentDataLoader

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, SamplingTypes


class TestDataRead(unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    # note: base class vars
    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa-lukasz")
    pod_version = environ.get('KB_VERSION', None)
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    results_cos_path = 'results_wml_autoai'

    database_name = "sqlserver"
    schema_name = 'connections'
    table_name = 'blank'

    # to be set in every child class:
    OPTIMIZER_NAME = "AutoAI regression test"

    SPACE_ONLY = True
    HISTORICAL_RUNS_CHECK = True
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    service: 'WebService' = None
    service_batch: 'Batch' = None

    cos_resource_instance_id = None
    experiment_info: dict = None

    trained_pipeline_details = None
    run_id = None
    prev_run_id = None
    data_connection = None
    results_connection = None
    train_data = None

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None
    X_df = None
    X_values = None
    y_values = None

    project_id = None
    space_id = None

    asset_id = None
    connection_id = None
    db_connection_id = None
    db_data_asset_id = None
    db_data_connections = []
    # end note

    cos_resource = None
    file_names = ['autoai_exp_cfpb_Small.csv', 'height.xlsx', 'japanese_gb18030.csv', 'cashflows_descend_sort.csv',
                  'AUTOAI-marvel-wikia-data-characters.csv']
    data_locations = [join('./autoai/data/read_issues', name) for name in file_names]
    data_cos_paths = [join('data', name) for name in file_names]

    SPACE_ONLY = True
    OPTIMIZER_NAME = "read issues"
    target_space_id = None
    connections_ids = []
    assets_ids = []
    data_connections = []
    results_connections = []

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())

        cls.cos_credentials = get_cos_credentials()
        cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
        cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.project_id = cls.wml_credentials.get('project_id')

    def test_00a_space_cleanup(self):
        space_checked = False
        while not space_checked:
            space_cleanup(self.wml_client,
                          get_space_id(self.wml_client, self.space_name,
                                       cos_resource_instance_id=self.cos_resource_instance_id),
                          days_old=7)
            space_id = get_space_id(self.wml_client, self.space_name,
                                    cos_resource_instance_id=self.cos_resource_instance_id)
            try:
                self.assertIsNotNone(space_id, msg="space_id is None")
                space_checked = True
            except AssertionError:
                space_checked = False

        TestDataRead.space_id = space_id

        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

    def test_00b_prepare_connection_to_COS(self):
        for location, cos_path in zip(self.data_locations, self.data_cos_paths):
            TestDataRead.connection_id, TestDataRead.bucket_name = create_connection_to_cos(
                wml_client=self.wml_client,
                cos_credentials=self.cos_credentials,
                cos_endpoint=self.cos_endpoint,
                bucket_name=self.bucket_name,
                save_data=True,
                data_path=location,
                data_cos_path=cos_path)

            self.connections_ids.append(TestDataRead.connection_id)

        self.assertIsInstance(self.connection_id, str)

    @unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "SQL Server not supported on FIPS clusters")
    def test_00c_prepare_connection_to_SQL_Server(self):
        from ibm_watson_machine_learning.tests.utils import get_db_credentials
        db_credentials = get_db_credentials(self.database_name)
        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name(self.database_name),
            'name': f'Connection to DB - {self.database_name}',
            'properties': db_credentials
        })
        TestDataRead.db_connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(TestDataRead.db_connection_id, str)

    def test_00d_prepare_connected_data_asset(self):
        for connection_id, cos_path in zip(self.connections_ids, self.data_cos_paths):
            asset_details = self.wml_client.data_assets.store({
                self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: connection_id,
                self.wml_client.data_assets.ConfigurationMetaNames.NAME: "training asset",
                self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: join(self.bucket_name,
                                                                                           cos_path)
            })

            self.assets_ids.append(self.wml_client.data_assets.get_id(asset_details))

        self.assertEqual(len(self.assets_ids), 5)

    @unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "SQL Server not supported on FIPS clusters")
    def test_00e_prepare_db_DataConnections(self):
        from ibm_watson_machine_learning.helpers.connections import DatabaseLocation
        db_connected_asset = DataConnection(
            connection_asset_id=self.db_connection_id,
            location=DatabaseLocation(
                schema_name=self.schema_name,
                table_name=self.table_name
            )
        )

        from os.path import join

        asset_details = self.wml_client.data_assets.store({
            self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: self.db_connection_id,
            self.wml_client.data_assets.ConfigurationMetaNames.NAME: f"BLANK TABLE {self.database_name}",
            self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: join(self.schema_name,
                                                                                       self.table_name)
        })
        TestDataRead.db_data_asset_id = self.wml_client.data_assets.get_id(asset_details)
        db_data_asset = DataConnection(data_asset_id=TestDataRead.db_data_asset_id)

        TestDataRead.db_data_connections = [db_connected_asset, db_data_asset]

        print(self.db_data_connections)
        print(TestDataRead.db_data_connections)

    def test_02_data_reference_setup(self):
        for asset_id in self.assets_ids:
            self.data_connections.append(DataConnection(data_asset_id=asset_id))
            self.results_connections.append(DataConnection(
                location=ContainerLocation()
            ))

        self.assertEqual(len(self.data_connections), 5)
        self.assertEqual(len(self.results_connections), 5)

    def test_03a_get_train_data_autoai_exp_cfpb_Small(self):  # test data
        self.data_connections[0].set_client(self.wml_client)
        experiment_info = dict(name='OPTIMIZER_NAME',
                               desc='test description',
                               prediction_type=PredictionType.MULTICLASS,
                               prediction_column='Consumer Loan')
        test_df = self.data_connections[0].read(use_flight=True, enable_sampling=False)
        print(test_df.head())
        print(test_df.shape)
        print(test_df[experiment_info['prediction_column']].value_counts())
        self.assertEqual(test_df.shape, (5000, 2))

    @unittest.skip("Extra columns added by Flight issue https://github.ibm.com/wdp-gov/tracker/issues/86059")
    def test_03a_get_train_data_autoai_exp_cfpb_Small_Flight(self):  # test data
        self.data_connections[0].set_client(self.wml_client)
        test_df = self.data_connections[0].read(use_flight=True)
        print(test_df.head())
        print(test_df.shape)
        self.assertEqual(test_df.shape, (5000, 2))

    def test_03b_get_train_data_height(self):  # test data
        self.data_connections[1].set_client(self.wml_client)
        test_df = self.data_connections[1].read(use_flight=False)
        print(test_df.head())
        print(test_df.shape)
        self.assertEqual(test_df.shape, (199, 3))

    def test_03b_get_train_data_height_Flight(self):  # test data
        self.data_connections[1].set_client(self.wml_client)
        test_df = self.data_connections[1].read(use_flight=True, excel_sheet='height')
        print(test_df.head())
        print(test_df.shape)
        self.assertEqual(test_df.shape, (199, 3))

    def test_03c_get_train_data_japanese_gb18030(self):  # test data
        self.data_connections[2].set_client(self.wml_client)
        test_df = self.data_connections[2].read(use_flight=False, encoding='gb18030')
        print(test_df.head())
        target = u'温度'
        self.assertTrue(target in str(test_df.columns))

    def test_03d_get_train_data_japanese_gb18030_Flight(self):  # test data
        self.data_connections[2].set_client(self.wml_client)
        test_df = self.data_connections[2].read(use_flight=True)
        print(test_df.head())
        target = u'温度'
        self.assertTrue(target in str(test_df.columns))

    @unittest.skipIf(getenv('FIPS', 'false').lower() == 'true', "SQL Server not supported on FIPS clusters")
    def test_03e_get_empty_sql_table(self):
        from ibm_watson_machine_learning.wml_client_error import EmptyDataSource

        self.assertGreater(len(self.db_data_connections), 0)

        for db_data_conn in self.db_data_connections:
            db_data_conn.set_client(self.wml_client)
            with self.assertRaises(EmptyDataSource) as error:
                db_data_conn.read()

            self.assertIsInstance(db_data_conn, DataConnection)

    def test_04a_as_iterator(self):
        self.data_connections[0].set_client(self.wml_client)
        iterator = self.data_connections[0].read(use_flight=True, return_data_as_iterator=True)
        self.assertTrue(isinstance(iterator, ExperimentDataLoader))
        for df in iterator:
            self.assertTrue(isinstance(df, pandas.DataFrame))

    def test_04b_sampling_first_n_records_params(self):
        self.data_connections[0].set_client(self.wml_client)
        # one batch as sample with size 10kb
        test_df_first_n = self.data_connections[0].read(use_flight=True, sampling_type=SamplingTypes.FIRST_VALUES,
                                                        sample_size_limit=10000)
        print(test_df_first_n.head())
        print(test_df_first_n.shape)
        # self.assertEqual(test_df_first_n.shape, (5000, 2))

    def test_04c_sampling_random_params(self):
        experiment_info = dict(name='OPTIMIZER_NAME',
                               desc='test description',
                               prediction_type=PredictionType.REGRESSION,
                               prediction_column='height')

        self.data_connections[0].set_client(self.wml_client)
        # one batch with random sample of size 100kb
        test_df_rand = self.data_connections[1].read(use_flight=True, sampling_type=SamplingTypes.RANDOM,
                                                     experiment_metadata=experiment_info,
                                                     sample_size_limit=100000)
        print(test_df_rand.head())
        print(test_df_rand.shape)
        self.assertTrue(abs(test_df_rand.shape[0] - 108) < 10)
        self.assertEqual(test_df_rand.shape[1], 3)

    def test_04d_sampling_stratified_params(self):
        experiment_info = dict(name='OPTIMIZER_NAME',
                               desc='test description',
                               prediction_type=PredictionType.MULTICLASS,
                               prediction_column='Consumer Loan')

        self.data_connections[0].set_client(self.wml_client)
        # one batch with stratified sample of size 100kb
        sample_size_limit = 100 * 1024
        test_df_strat = self.data_connections[0].read(use_flight=True, sampling_type=SamplingTypes.STRATIFIED,
                                                      experiment_metadata=experiment_info,
                                                      sample_size_limit=sample_size_limit)
        print(test_df_strat.head())
        print(test_df_strat.shape)
        print(sample_size_limit // (sys.getsizeof(test_df_strat) / len(test_df_strat)))
        self.assertAlmostEqual(sys.getsizeof(test_df_strat), sample_size_limit, delta=20 * 1024)
        self.assertAlmostEqual(test_df_strat.shape[0], 84, delta=10)
        self.assertEqual(test_df_strat.shape[1], 2)
        print(test_df_strat[experiment_info['prediction_column']].value_counts())

    def test_04e_sampling_random_classification_params(self):
        experiment_info = dict(name='OPTIMIZER_NAME',
                               desc='test description',
                               prediction_type=PredictionType.MULTICLASS,
                               prediction_column='Consumer Loan')

        self.data_connections[0].set_client(self.wml_client)
        # one batch with stratified sample of size 100kb
        sample_size_limit = 100 * 1024
        test_df_strat = self.data_connections[0].read(use_flight=True, sampling_type=SamplingTypes.RANDOM,
                                                      experiment_metadata=experiment_info,
                                                      sample_size_limit=sample_size_limit)
        print(test_df_strat.head())
        print(test_df_strat.shape)
        self.assertAlmostEqual(sys.getsizeof(test_df_strat), sample_size_limit, delta=20 * 1024)
        self.assertAlmostEqual(test_df_strat.shape[0], 84, delta=10)
        self.assertEqual(test_df_strat.shape[1], 2)
        print(test_df_strat[experiment_info['prediction_column']].value_counts())

    def test_04f_sampling_without_size_stratified_params(self):
        experiment_info = dict(name='OPTIMIZER_NAME',
                               desc='test description',
                               prediction_type=PredictionType.MULTICLASS,
                               prediction_column='Consumer Loan')

        self.data_connections[0].set_client(self.wml_client)
        test_df_strat = self.data_connections[0].read(use_flight=True, sampling_type=SamplingTypes.STRATIFIED,
                                                      experiment_metadata=experiment_info)
        print(test_df_strat.head())
        print(test_df_strat.shape)
        self.assertTrue(abs(test_df_strat.shape[0] - 4969) < 10)
        self.assertEqual(test_df_strat.shape[1], 2)
        print(test_df_strat[experiment_info['prediction_column']].value_counts())

    def test_04g_sampling_random_classification_raw_params(self):
        experiment_info = dict(name='OPTIMIZER_NAME',
                               desc='test description',
                               prediction_type=PredictionType.MULTICLASS,
                               prediction_column='Consumer Loan', raw=True)

        self.data_connections[0].set_client(self.wml_client)
        # one batch with stratified sample of size 100kb
        sample_size_limit = 100 * 1024
        test_df_strat = self.data_connections[0].read(use_flight=True, sampling_type=SamplingTypes.RANDOM,
                                                      experiment_metadata=experiment_info,
                                                      sample_size_limit=sample_size_limit, raw=True)
        print(test_df_strat.head())
        print(test_df_strat.shape)
        self.assertAlmostEqual(sys.getsizeof(test_df_strat), sample_size_limit, delta=20 * 1024)
        self.assertAlmostEqual(test_df_strat.shape[0], 84, delta=10)
        self.assertEqual(test_df_strat.shape[1], 2)
        print(test_df_strat[experiment_info['prediction_column']].value_counts())

    def test_04h_sampling_no_type_params(self):
        experiment_info = dict(name='OPTIMIZER_NAME',
                               desc='test description',
                               prediction_type=PredictionType.MULTICLASS,
                               prediction_column='Consumer Loan', raw=True)

        self.data_connections[0].set_client(self.wml_client)
        # one batch with stratified sample of size 100kb
        sample_size_limit = 100 * 1024
        test_df_strat = self.data_connections[0].read(use_flight=True,
                                                      experiment_metadata=experiment_info,
                                                      sample_size_limit=sample_size_limit, raw=True)
        print(test_df_strat.head())
        print(test_df_strat.shape)
        self.assertAlmostEqual(sys.getsizeof(test_df_strat), sample_size_limit, delta=20 * 1024)
        self.assertAlmostEqual(test_df_strat.shape[0], 84, delta=10)
        self.assertEqual(test_df_strat.shape[1], 2)
        print(test_df_strat[experiment_info['prediction_column']].value_counts())

    def test_04i_sampling_random_classification_dict_params(self):
        sample_size_limit = 100 * 1024  # 100 kB
        sample_dict = {'sample_size_limit': sample_size_limit, 'sampling_type': 'random'}
        self.data_connections[0].set_client(self.wml_client)

        test_df_strat = self.data_connections[0].read(use_flight=True, raw=True, **sample_dict)

        print(test_df_strat.head())
        print(test_df_strat.shape)
        self.assertAlmostEqual(sys.getsizeof(test_df_strat), sample_size_limit, delta=35 * 1024)
        self.assertAlmostEqual(test_df_strat.shape[0], 84, delta=10)
        self.assertEqual(test_df_strat.shape[1], 2)

    def test_04j_sampling_stratified_missing_values_in_target_column_data_stats(self):
        data_conn = self.data_connections[4]
        data_conn.set_client(self.wml_client)

        df, data_stats = data_conn.read(use_flight=True, raw=True,
                                        experiment_metadata={
                                            "prediction_type": PredictionType.MULTICLASS,
                                            "prediction_column": 'ALIGN',  # kb
                                        },
                                        sampling_type='stratified',
                                        _return_subsampling_stats=True
                                        )

        self.assertEqual(data_stats.get('no_batches'), 1)
        self.assertEqual(data_stats.get('data_batch_nrows'), df.shape[0])
        self.assertEqual(data_stats.get('data_nrows'), data_stats.get('data_nrows'))
        self.assertEqual(df.shape[0], 13561)

    def test_04k_sampling_random_missing_values_in_target_column_data_stats(self):
        data_conn = self.data_connections[4]
        data_conn.set_client(self.wml_client)

        df, data_stats = data_conn.read(use_flight=True, raw=True,
                                        experiment_metadata={
                                            "prediction_type": PredictionType.MULTICLASS,
                                            "prediction_column": 'ALIGN',  # kb
                                        },
                                        sampling_type='random',
                                        _return_subsampling_stats=True
                                        )
        print(data_stats)
        self.assertEqual(data_stats.get('no_batches'), 1)
        self.assertEqual(data_stats.get('data_batch_nrows'), df.shape[0])
        self.assertEqual(data_stats.get('data_nrows'), data_stats.get('data_nrows'))
        self.assertEqual(df.shape[0], 13561)

    def test_05_a_batch_size(self):
        self.data_connections[0].set_client(self.wml_client)
        # one batch as sample with size 1000 rows
        batch_size = 120
        test_df = self.data_connections[0].read(use_flight=True, sample_rows_limit=batch_size)
        print(test_df.head())
        print(test_df.shape)
        self.assertTrue(abs(test_df.shape[0] - batch_size) < 10)
        self.assertEqual(test_df.shape[1], 2)

    def test_06a_sampling_truncate_with_timestamp(self):
        sample_size_limit = 50 * 1024  # 100 kB
        experiment_info = dict(name='TS experience',
                               desc='test description',
                               prediction_type=PredictionType.FORECASTING,
                               timestamp_column_name='DATE')

        self.data_connections[3].set_client(self.wml_client)
        test_df_strat = self.data_connections[3].read(use_flight=True, sampling_type=SamplingTypes.LAST_VALUES,
                                                      experiment_metadata=experiment_info,
                                                      sample_size_limit=sample_size_limit)
        print(test_df_strat)
        print(test_df_strat.shape)
        print((test_df_strat.iloc[-1, 0]))
        self.assertAlmostEqual(sys.getsizeof(test_df_strat), sample_size_limit, delta=35 * 1024)
        self.assertAlmostEqual(test_df_strat.shape[0], 197, delta=10)
        self.assertEqual(test_df_strat.shape[1], 25)
        self.assertEqual(str(test_df_strat.iloc[-1, 0]), '2002-01-31')

    def test_06b_sampling_truncate_without_timestamp(self):
        sample_size_limit = 50 * 1024  # 50 kB
        experiment_info = dict(name='TS experience',
                               desc='test description',
                               prediction_type=PredictionType.FORECASTING)

        self.data_connections[3].set_client(self.wml_client)
        test_df_strat = self.data_connections[3].read(use_flight=True, sampling_type=SamplingTypes.LAST_VALUES,
                                                      experiment_metadata=experiment_info,
                                                      sample_size_limit=sample_size_limit)
        print(test_df_strat.head())
        print(test_df_strat.shape)
        self.assertAlmostEqual(sys.getsizeof(test_df_strat), sample_size_limit, delta=35 * 1024)
        self.assertAlmostEqual(test_df_strat.shape[0], 197, delta=10)
        self.assertEqual(test_df_strat.shape[1], 25)
        self.assertEqual(str(test_df_strat.iloc[-1, 0]), '2001-01-02')

    def test_99_delete_connection_and_connected_data_asset(self):
        if self.db_data_asset_id and self.db_connection_id:
            self.assets_ids.append(self.db_data_asset_id)
            self.connections_ids.append(self.db_connection_id)
        for asset_id, connection_id in zip(self.assets_ids, self.connections_ids):
            self.wml_client.data_assets.delete(asset_id)
            self.wml_client.connections.delete(connection_id)

            with self.assertRaises(WMLClientError):
                self.wml_client.data_assets.get_details(asset_id)
                self.wml_client.connections.get_details(connection_id)


if __name__ == '__main__':
    unittest.main()
