#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import timeout_decorator
from os import environ
import sys

from os.path import join

import ibm_boto3
import pandas as pd
import logging

# logging.basicConfig()
# logger = logging.getLogger("automl")
# logger.setLevel(logging.DEBUG)

from ibm_watson_machine_learning.data_loaders.datasets.experiment import DEFAULT_SAMPLE_SIZE_LIMIT
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers.connections import DataConnection, ContainerLocation
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import create_connection_to_cos, get_wml_credentials, get_cos_credentials, \
    get_space_id
from ibm_watson_machine_learning.data_loaders.experiment import ExperimentDataLoader

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, SamplingTypes


class AbstractAutoAISubsamplingIteratorBatched:
    """
    The test can be run on CLOUD, and CPD
    """

    ## beginning of base class vars

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa-lukasz")
    pod_version = environ.get('KB_VERSION', None)
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space_14')

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    results_cos_path = 'results_wml_autoai'

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

    ## end of base class vars

    cos_resource = None
    file_names = ['regr_taxi_fare_20MB.csv']  # 'autoai_exp_cfpb_Small.csv', 'height.xlsx', 'japanese_gb18030.csv']
    data_locations = [join('./autoai/data/read_issues', name) for name in file_names]
    data_cos_paths = [join('data', name) for name in file_names]
    data_rows_count = [len(pd.read_csv(path)) for path in data_locations]
    label_columns = ['passenger_count']
    index_columns = ['Unnamed: 0']

    SPACE_ONLY = True
    OPTIMIZER_NAME = "read issues"
    target_space_id = None
    connections_ids = []
    assets_ids = []
    data_connections = []
    results_connections = []

    experiment_info = dict(name='OPTIMIZER_NAME',
                           desc='test description',
                           prediction_type=PredictionType.MULTICLASS,
                           prediction_column=label_columns[0])

    TIMEOUT = 200

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
        cos = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # for bucket in cos.buckets.all():
        #     print(bucket.name)

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

        AbstractAutoAISubsamplingIteratorBatched.space_id = space_id

        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

    def test_00b_prepare_connection_to_COS(self):
        for location, cos_path in zip(self.data_locations, self.data_cos_paths):
            AbstractAutoAISubsamplingIteratorBatched.connection_id, AbstractAutoAISubsamplingIteratorBatched.bucket_name = create_connection_to_cos(
                wml_client=self.wml_client,
                cos_credentials=self.cos_credentials,
                cos_endpoint=self.cos_endpoint,
                bucket_name=self.bucket_name,
                save_data=True,
                data_path=location,
                data_cos_path=cos_path)

            self.connections_ids.append(AbstractAutoAISubsamplingIteratorBatched.connection_id)

        self.assertIsInstance(self.connection_id, str)

    def test_00d_prepare_connected_data_asset(self):
        for connection_id, cos_path in zip(self.connections_ids, self.data_cos_paths):
            asset_details = self.wml_client.data_assets.store({
                self.wml_client.data_assets.ConfigurationMetaNames.CONNECTION_ID: connection_id,
                self.wml_client.data_assets.ConfigurationMetaNames.NAME: "training asset",
                self.wml_client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: join(self.bucket_name,
                                                                                           cos_path)
            })

            self.assets_ids.append(self.wml_client.data_assets.get_id(asset_details))

        self.assertEqual(len(self.assets_ids), len(self.file_names))

    def test_02_data_reference_setup(self):
        for asset_id in self.assets_ids:
            self.data_connections.append(DataConnection(data_asset_id=asset_id))
            self.results_connections.append(DataConnection(
                location=ContainerLocation()
            ))

        self.assertEqual(len(self.data_connections), len(self.file_names))
        self.assertEqual(len(self.results_connections), len(self.file_names))

    def initialize_data_set_read(self, return_data_as_iterator, enable_sampling, sample_size_limit, sampling_type,
                                 number_of_batch_rows, return_subsampling_stats, experiment_metadata, total_size_limit):
        raise NotImplemented()

    def iterate_through_dataset_and_validate(self, return_data_as_iterator, enable_sampling, sample_size_limit,
                                             sampling_type,
                                             number_of_batch_rows, return_subsampling_stats, experiment_metadata,
                                             total_size_limit, total_nrows_limit):
        res = self.initialize_data_set_read(return_data_as_iterator, enable_sampling, sample_size_limit, sampling_type,
                                            number_of_batch_rows, return_subsampling_stats, experiment_metadata,
                                            total_size_limit, total_nrows_limit)

        if return_data_as_iterator:
            return self.iterate_through_dataset(res, number_of_batch_rows, return_subsampling_stats)
        else:
            return self.validate_returned_data(res, return_subsampling_stats)

    def iterate_through_dataset(self, iterator, number_of_batch_rows, return_subsampling_stats):
        self.assertTrue(isinstance(iterator, ExperimentDataLoader))

        downloaded_data_nrows = 0
        downloaded_data_size = 0
        iteration = 0
        first_batch_size = 0
        indexes = []

        for iteration, batch_data in enumerate(iterator):
            if return_subsampling_stats:
                self.assertEqual(type(batch_data), tuple)
                batch_data = batch_data[0]

            self.assertTrue(isinstance(batch_data, pd.DataFrame))

            downloaded_data_nrows += len(batch_data)
            if number_of_batch_rows:
                self.assertTrue(len(batch_data) <= number_of_batch_rows,
                                f'Batch size bigger than number_of_batch_rows ({len(batch_data)} > {number_of_batch_rows})')
            indexes_in_batch = batch_data[self.index_columns[0]].tolist()
            indexes.extend(indexes_in_batch)

            if first_batch_size == 0:
                first_batch_size = sys.getsizeof(batch_data)

            downloaded_data_size += sys.getsizeof(batch_data)

        return {"downloaded_data_nrows": downloaded_data_nrows,
                "downloaded_data_size": downloaded_data_size,
                "first_batch_size": first_batch_size,
                "all_indexes_count": len(indexes),
                "unique_indexes_count": len(set(indexes)),
                "no_batches_downloaded": iteration + 1  # enumerate starts from 0 so +1 added.
                }

    def validate_returned_data(self, data, return_subsampling_stats):
        if return_subsampling_stats:
            self.assertEqual(type(data), tuple)
            data = data[0]

        self.assertTrue(isinstance(data, pd.DataFrame))

        indexes = data[self.index_columns[0]].tolist()

        return {"downloaded_data_nrows": len(data),
                "downloaded_data_size": sys.getsizeof(data),
                "first_batch_size": sys.getsizeof(data),
                "all_indexes_count": len(indexes),
                "unique_indexes_count": len(set(indexes)),
                "no_batches_downloaded": None  # we do not iterate here
                }

    def is_non_iterator_available(self):
        raise NotImplemented()

    @timeout_decorator.timeout(TIMEOUT)
    def _test_read_func(self, return_data_as_iterator, enable_sampling, sample_size_limit, sampling_type,
                        number_of_batch_rows,
                        return_subsampling_stats, experiment_metadata, total_size_limit, total_nrows_limit):
        print(f"\n\nIterator: {return_data_as_iterator}")
        print(
            f"Sampling: {enable_sampling} {sampling_type}({sample_size_limit / (1024 * 1024) if sample_size_limit else '-'} MB)")
        print(f"Batch: {number_of_batch_rows}")
        print(f"Sampling stats: {return_subsampling_stats}")
        print(f"Experiment metadata:", experiment_metadata)
        print(f"Total size limit: {total_size_limit}")
        print(f"Total no. rows limit: {total_nrows_limit}")

        try:
            self.data_connections[0].set_client(self.wml_client)

            iteration_statistics = self.iterate_through_dataset_and_validate(
                return_data_as_iterator, enable_sampling, sample_size_limit, sampling_type,
                number_of_batch_rows, return_subsampling_stats,
                experiment_metadata, total_size_limit, total_nrows_limit)

            print('Rows number: ', iteration_statistics['downloaded_data_nrows'], '/', self.data_rows_count[0])
            print('Dataset size: ', iteration_statistics['downloaded_data_size'] / (1024 * 1024), "MB")
            print("First batch size: ", iteration_statistics['first_batch_size'] / (1024 * 1024), "MB")
            print('Number of indexes (all/ unique):', iteration_statistics['all_indexes_count'], '/',
                  iteration_statistics['unique_indexes_count'])
            # print(data[self.index_columns[0]].tolist())

            if sample_size_limit:
                self.assertTrue(iteration_statistics['first_batch_size'] <= 1.1 * sample_size_limit,
                                f"Byte size much bigger than sample_size_limit ({iteration_statistics['first_batch_size']} > {sample_size_limit})")
            elif total_size_limit:
                self.assertTrue(iteration_statistics['downloaded_data_size'] <= 1.1 * total_size_limit,
                                f"Byte size much bigger than total_size_limit ({iteration_statistics['downloaded_data_size']} > {total_size_limit})")
            elif total_nrows_limit:
                self.assertAlmostEqual(iteration_statistics['downloaded_data_nrows'], total_nrows_limit, delta=5,
                                       msg=f"Incorrect amount data set was downloaded ({iteration_statistics['downloaded_data_nrows']} / {total_nrows_limit}")
            else:
                self.assertEqual(iteration_statistics['downloaded_data_nrows'], self.data_rows_count[0],
                                 f"Not complete data set was downloaded ({iteration_statistics['downloaded_data_nrows']} / {self.data_rows_count[0]})")

            print('\nSUCCESS')
        except Exception as e:
            # traceback.print_exc()
            print(f'\n{e.__repr__()}')
            raise e

    @timeout_decorator.timeout(TIMEOUT)
    def test_05_read_func_sample_size_limit_is_0(self):
        with self.assertRaises(Exception):  # clearly incorrect input
            self.data_connections[0].set_client(self.wml_client)
            self.read_from_api(False, 0, SamplingTypes.FIRST_VALUES, None, None, None)

    @timeout_decorator.timeout(TIMEOUT)
    def test_06_read_func_sample_size_limit_is_minus_1(self):
        with self.assertRaises(Exception):  # clearly incorrect input
            self.data_connections[0].set_client(self.wml_client)
            self.read_from_api(False, -1, SamplingTypes.FIRST_VALUES, None, None, None)

    @timeout_decorator.timeout(TIMEOUT)
    def test_07_read_func_sample_size_limit_is_1(self):
        with self.assertRaises(Exception):  # clearly incorrect input
            self.data_connections[0].set_client(self.wml_client)
            self.read_from_api(False, -1, SamplingTypes.FIRST_VALUES, None, None, None)

    @timeout_decorator.timeout(TIMEOUT)
    def test_08_read_func_experiment_metadata_is_empty_dict(self):
        with self.assertRaises(Exception):  # clearly incorrect input
            self.data_connections[0].set_client(self.wml_client)
            self.read_from_api(False, None, None, None, None, {})

    def test_99_delete_connection_and_connected_data_asset(self):
        for asset_id, connection_id in zip(self.assets_ids, self.connections_ids):
            self.wml_client.data_assets.delete(asset_id)
            self.wml_client.connections.delete(connection_id)

            with self.assertRaises(WMLClientError):
                self.wml_client.data_assets.get_details(asset_id)
                self.wml_client.connections.get_details(connection_id)

    def test_04_read(self):
        for func in [self._test_read_func]:
            for return_data_as_iterator in [True, False] if self.is_non_iterator_available() else [True]:
                for enable_sampling in [True, False]:
                    for sampling_type in [None, SamplingTypes.RANDOM,
                                          SamplingTypes.STRATIFIED] if enable_sampling or not return_data_as_iterator else [None]:
                        for total_size_limit in [None, 2 * 1024 * 1024, 25 * 1024 * 1024] if enable_sampling else [
                            None]:
                            for total_nrows_limit in [2000, None] if enable_sampling else [None]:
                                for number_of_batch_rows in [None] if total_size_limit or enable_sampling else [None, 1000]:
                                    for sample_size_limit in [None] if number_of_batch_rows else [None, 2 * 1024 * 1024]:
                                        for return_subsampling_stats in [True, False] if enable_sampling else [False]:
                                            experiment_meta = dict(name='OPTIMIZER_NAME',
                                                                   desc='test description',
                                                                   prediction_type=PredictionType.MULTICLASS,
                                                                   prediction_column=self.label_columns[0])

                                            for experiment_metadata in [experiment_meta]:
                                                with self.subTest(iterator=return_data_as_iterator,
                                                                  enable_sampling=enable_sampling,
                                                                  sampling=sampling_type,
                                                                  sample_size=sample_size_limit,
                                                                  batch_rows=number_of_batch_rows,
                                                                  sampling_stats=return_subsampling_stats,
                                                                  exp_meta=experiment_metadata,
                                                                  total_size_limit=total_size_limit,
                                                                  total_nrows_limit=total_nrows_limit):
                                                    func(return_data_as_iterator, enable_sampling, sample_size_limit,
                                                         sampling_type,
                                                         number_of_batch_rows, return_subsampling_stats,
                                                         experiment_metadata, total_size_limit, total_nrows_limit)



if __name__ == '__main__':
    unittest.main()
