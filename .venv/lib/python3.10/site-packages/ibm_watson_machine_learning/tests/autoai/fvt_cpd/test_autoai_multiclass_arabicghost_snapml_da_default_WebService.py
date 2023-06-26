#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.helpers.connections import DataConnection,  AssetLocation
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestWebservice, \
    AbstractTestAutoAIAsync

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, ClassificationAlgorithms


class TestAutoAIRemote(AbstractTestAutoAIAsync, AbstractTestWebservice, unittest.TestCase):
    """
    The test can be run on CP4D (data asset with csv file)
    """

    cos_resource = None
    data_location = './autoai/data/arabicghosts_train.csv'

    cos_endpoint = "https://s3.us.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'data/arabicghosts_train.csv'

    SPACE_ONLY = False

    OPTIMIZER_NAME = "Arabic Ghost test sdk"

    target_space_id = None
    asset_id = None

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

    def test_00b_write_data_to_da_cpd(self):
        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

        asset_details = self.wml_client.data_assets.create(
            name=self.data_location.split('/')[-1],
            file_path=self.data_location)
        TestAutoAIRemote.asset_id = asset_details['metadata']['guid']

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            location=AssetLocation(asset_id=self.asset_id))
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNone(obj=TestAutoAIRemote.results_connection)

    def test_99_delete_data_asset(self):
        self.wml_client.data_assets.delete(self.asset_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.data_assets.get_details(self.asset_id)
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
