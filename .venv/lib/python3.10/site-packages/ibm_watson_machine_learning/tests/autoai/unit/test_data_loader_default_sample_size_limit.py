#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import abc
from sys import getsizeof
from os import environ
import pandas as pd
import pprint
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestAutoAIAsync, \
    AbstractTestWebservice
from ibm_watson_machine_learning.data_loaders import experiment as data_loaders
from ibm_watson_machine_learning.data_loaders.datasets import experiment as datasets
from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms, \
    BatchedClassificationAlgorithms, SamplingTypes
from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id,
                                                     is_cp4d)
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id,
                                                     is_cp4d)
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, RunStateTypes
from ibm_watson_machine_learning.tests.utils.assertions import get_and_predict_all_pipelines_as_lale, \
    validate_autoai_experiment
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


class TestAutoAIRemote(unittest.TestCase):
    """
    The test can be run on CPD only
    """
    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    pod_version = environ.get('KB_VERSION', None)
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    results_cos_path = 'results_wml_autoai'

    cos_resource = None
    data_location = './autoai/data/Call_log.csv'
    data_cos_path = 'data/Call_log.csv'
    SPACE_ONLY = False
    OPTIMIZER_NAME = "breast_cancer test sdk"
    target_space_id = None
    df = None
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None

    cos_resource_instance_id = None
    experiment_info: dict = None

    trained_pipeline_details = None
    run_id = None
    data_connection = None
    train_data = None
    project_id = None
    space_id = None
    asset_id = None
    connection_id = None

    # stratified 10kb should be read
    experiment_1_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.MULTICLASS,
        prediction_column='Plan_ID',
        scoring=Metrics.AVERAGE_PRECISION_SCORE,
        max_number_of_estimators=1,
        autoai_pod_version='!>dev-incremental_learning-11',  # TODO to be removed
        include_only_estimators=[ClassificationAlgorithms.SnapRF, ClassificationAlgorithms.LGBM],
        include_batched_ensemble_estimators=[BatchedClassificationAlgorithms.SnapRF,
                                             BatchedClassificationAlgorithms.LGBM],
        use_flight=True
    )

    # first 10kb of data should be read
    # experiment_2_info = experiment_1_info.copy()
    # experiment_2_info['sampling_type'] = SamplingTypes.FIRST_N_RECORDS
    #
    # # defaults - the whole data set should be read
    # experiment_3_info = experiment_1_info.copy()
    # experiment_3_info.pop('number_of_batch_rows', None)
    # experiment_3_info.pop('sample_size_limit', None)
    # experiment_3_info.pop('sampling_type', None)

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
        cls.wml_client.set.default_project(cls.project_id)

    def test_00d_prepare_data_asset(self):
        TestAutoAIRemote.orig_dataset = pd.read_csv(self.data_location)
        print(f"Original dataset shape: {self.orig_dataset.shape}")
        print(f"Original dataset size: {getsizeof(self.orig_dataset)}")
        print(self.orig_dataset)
        asset_details = self.wml_client.data_assets.create(
            name=self.data_location.split('/')[-1],
            file_path=self.data_location)

        TestAutoAIRemote.asset_id = self.wml_client.data_assets.get_id(asset_details)
        self.assertIsInstance(self.asset_id, str)
        TestAutoAIRemote.data_connection = DataConnection(data_asset_id=TestAutoAIRemote.asset_id)

    def test_01_download_data_default(self):
        if environ.get('MEM', False):
            del environ['MEM']
        pprint.pprint(TestAutoAIRemote.experiment_1_info)
        dataset_1 = datasets.ExperimentIterableDataset(
            connection=TestAutoAIRemote.data_connection,
            enable_sampling=True,
            experiment_metadata=TestAutoAIRemote.experiment_1_info,
            _wml_client=self.wml_client
        )
        loader_1 = data_loaders.ExperimentDataLoader(dataset=dataset_1)
        self.assertEqual(dataset_1.sample_size_limit, datasets.DEFAULT_SAMPLE_SIZE_LIMIT)
        batch_df = None
        for batch_df in loader_1:
            # batch_df.to_csv('./autoai/data/breast_cancer_exp_1.csv', index=False)
            print(f"df shape: {batch_df.shape}, {len(batch_df)}")
            print(f"df size (MB): {getsizeof(batch_df) / (1024 * 1024)}")
            break

        self.assertEqual(len(batch_df), len(self.orig_dataset))
        print('end')

    def test_02_download_data_m_tshirt(self):
        environ['MEM'] = "16Gi"
        pprint.pprint(TestAutoAIRemote.experiment_1_info)
        dataset_2 = datasets.ExperimentIterableDataset(
            connection=TestAutoAIRemote.data_connection,
            enable_sampling=True,
            experiment_metadata=TestAutoAIRemote.experiment_1_info,
            _wml_client=self.wml_client
        )
        loader_2 = data_loaders.ExperimentDataLoader(dataset=dataset_2)
        self.assertEqual(dataset_2.sample_size_limit, datasets.DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT)

        batch_df = None
        for batch_df in loader_2:
            # batch_df.to_csv('./autoai/data/breast_cancer_exp_1.csv', index=False)
            print(f"df shape: {batch_df.shape}, {len(batch_df)}")
            print(f"df size (MB): {getsizeof(batch_df) / (1024 * 1024)}")
            break
        self.assertLess(abs(getsizeof(batch_df) - datasets.DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT), 5 * 1024 *1024) # 5MB range
        print('end')

    def test_03_download_data_xl_tshirt(self):
        environ['MEM'] = "64Gi"
        pprint.pprint(TestAutoAIRemote.experiment_1_info)
        dataset_3 = datasets.ExperimentIterableDataset(
            connection=TestAutoAIRemote.data_connection,
            enable_sampling=True,
            experiment_metadata=TestAutoAIRemote.experiment_1_info,
            _wml_client=self.wml_client
        )
        loader_3 = data_loaders.ExperimentDataLoader(dataset=dataset_3)

        self.assertEqual(dataset_3.sample_size_limit, datasets.DEFAULT_SAMPLE_SIZE_LIMIT)
        batch_df = None
        for batch_df in loader_3:
            # batch_df.to_csv('./autoai/data/breast_cancer_exp_1.csv', index=False)
            print(f"df shape: {batch_df.shape}, {len(batch_df)}")
            print(f"df size (MB): {getsizeof(batch_df) / (1024 * 1024)}")
            break

        self.assertEqual(len(batch_df), len(self.orig_dataset))
        print('end')

    def test_04_download_data_custom_size_limit(self):
        if environ.get('MEM', False):
            del environ['MEM']
        custom_size_limit = 30 * 1024 * 1024  # 30MB
        pprint.pprint(TestAutoAIRemote.experiment_1_info)
        dataset_4 = datasets.ExperimentIterableDataset(
            connection=TestAutoAIRemote.data_connection,
            enable_sampling=True,
            experiment_metadata=TestAutoAIRemote.experiment_1_info,
            sample_size_limit=custom_size_limit,
            _wml_client=self.wml_client
        )
        loader_4 = data_loaders.ExperimentDataLoader(dataset=dataset_4)

        self.assertEqual(dataset_4.sample_size_limit, custom_size_limit)
        batch_df = None
        for batch_df in loader_4:
            # batch_df.to_csv('./autoai/data/breast_cancer_exp_1.csv', index=False)
            print(f"df shape: {batch_df.shape}, {len(batch_df)}")
            print(f"df size (MB): {getsizeof(batch_df) / (1024 * 1024)}")
            break

        self.assertLess(abs(getsizeof(batch_df) - custom_size_limit), 5 * 1024 * 1024)  # 5MB range
        print('end')
