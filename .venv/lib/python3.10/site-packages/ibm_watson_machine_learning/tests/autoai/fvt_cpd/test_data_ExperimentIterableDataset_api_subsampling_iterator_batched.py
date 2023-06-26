#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning.data_loaders.datasets.experiment import ExperimentIterableDataset
from ibm_watson_machine_learning.data_loaders.experiment import ExperimentDataLoader

from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes.abstract_autoai_data_subsampling_iterator_batched import \
    AbstractAutoAISubsamplingIteratorBatched
from ibm_watson_machine_learning.tests.utils import is_cp4d


#@unittest.skipIf(not is_cp4d(), "Batched Tree Ensembles not supported yet on cloud")
class TestAutoAIRemote(AbstractAutoAISubsamplingIteratorBatched, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """
    def is_non_iterator_available(self):
        return False

    def initialize_data_set_read(self, return_data_as_iterator, enable_sampling, sample_size_limit, sampling_type,
                                 number_of_batch_rows, return_subsampling_stats, experiment_metadata,
                                 total_size_limit, total_nrows_limit):
        if not return_data_as_iterator:
            raise Exception(
                'For ExperimentIterableDataset api setting `return_data_as_iterator` to False makes no sense.')

        params = {}

        if sampling_type is not None:
            params['sampling_type'] = sampling_type

        if experiment_metadata is not None:
            params['experiment_metadata'] = experiment_metadata

        if sample_size_limit is not None:
            params['sample_size_limit'] = sample_size_limit

        if number_of_batch_rows is not None:
            params['number_of_batch_rows'] = number_of_batch_rows

        if return_subsampling_stats is not None:
            params['_return_subsampling_stats'] = return_subsampling_stats
        if total_size_limit is not None:
            params['total_size_limit'] = total_size_limit
        if total_nrows_limit is not None:
            params['total_nrows_limit'] = total_nrows_limit

        dataset = ExperimentIterableDataset(
            connection=self.data_connections[0],
            enable_sampling=enable_sampling,
            **params
        )

        return ExperimentDataLoader(dataset)


if __name__ == '__main__':
    unittest.main()
