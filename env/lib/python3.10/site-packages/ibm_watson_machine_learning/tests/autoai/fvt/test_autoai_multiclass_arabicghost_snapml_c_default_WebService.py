#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import ibm_boto3
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import DataConnection, ContainerLocation
from ibm_watson_machine_learning.tests.utils import is_cp4d, save_data_to_container
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestWebservice, \
    AbstractTestAutoAIAsync

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms


class TestAutoAIRemote(AbstractTestAutoAIAsync, AbstractTestWebservice, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = './autoai/data/arabicghosts_train.csv'

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'data/arabicghosts_train.csv'

    SPACE_ONLY = False

    OPTIMIZER_NAME = "Arabic Ghost test sdk"

    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        desc='test description',
        prediction_type=PredictionType.MULTICLASS,
        prediction_column='لون',
        scoring=Metrics.LOG_LOSS,
        holdout_size=0.2,
        include_only_estimators=[ClassificationAlgorithms.SnapRF, ClassificationAlgorithms.SnapDT,
                                 ClassificationAlgorithms.SnapSVM, ClassificationAlgorithms.SnapLR],
        max_number_of_estimators=4
    )

    def test_00b_write_data_to_container(self):
        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

        save_data_to_container(self.data_location, self.data_cos_path, self.wml_client)

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            location=ContainerLocation(path=self.data_cos_path
                                       ))
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)


if __name__ == '__main__':
    unittest.main()
