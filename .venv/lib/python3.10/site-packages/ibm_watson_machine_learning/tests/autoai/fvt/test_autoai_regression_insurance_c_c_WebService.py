#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import ibm_boto3
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import DataConnection, ContainerLocation
from ibm_watson_machine_learning.tests.utils import is_cp4d, save_data_to_container
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import BaseTestStoreModel, \
    AbstractTestAutoAIAsync

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics, RegressionAlgorithms


@unittest.skipIf(is_cp4d(), "Excel files not supported on CPD 4.0 an low")
class TestAutoAIRemote(AbstractTestAutoAIAsync,BaseTestStoreModel , unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = './autoai/data/xlsx/credit_risk_insurance__two_sheets.xlsx'
    sheet_name = 'insurance'
    sheet_number = 1

    data_cos_path = 'data/credit_risk_insurance__two_sheets.xlsx'

    SPACE_ONLY = False
    pipeline_to_deploy = "Pipeline_1"

    OPTIMIZER_NAME = "Insurance test sdk"
    SKIP_BATCH_DEPLOYMENT = True

    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.REGRESSION,
        prediction_column='charges',
        outliers_columns=['charges'],
        scoring=Metrics.R2_SCORE,
        include_only_estimators=[RegressionAlgorithms.LGBM, RegressionAlgorithms.XGB],
        max_number_of_estimators=2,
        excel_sheet=sheet_name
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
        TestAutoAIRemote.results_connection = DataConnection(
            location=ContainerLocation(path=self.results_cos_path
                                       ))

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNotNone(obj=TestAutoAIRemote.results_connection)


if __name__ == '__main__':
    unittest.main()
