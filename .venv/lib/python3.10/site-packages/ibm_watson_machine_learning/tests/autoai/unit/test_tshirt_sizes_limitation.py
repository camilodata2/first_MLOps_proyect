#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest


from ibm_watson_machine_learning import APIClient

from ibm_watson_machine_learning.experiment import AutoAI

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d
from ibm_watson_machine_learning.utils.autoai.errors import TShirtSizeNotSupported
from ibm_watson_machine_learning.utils.autoai.enums import TShirtSize


class TestAutoAIRemote(unittest.TestCase):
    wml_credentials = None

    DEV_FIELD_NAME = 'development'
    OPTIMIZER_NAME = 'Test t-shirt size'

    experiment: 'AutoAI' = None

    supported_tshirt_size = TShirtSize.L


    @classmethod
    def setUp(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()

        cls.supported_tshirt_size = TShirtSize.L

    def test_01_wml_credentials_development_False(self):
        if self.DEV_FIELD_NAME in self.wml_credentials.keys():
            self.wml_credentials.pop(self.DEV_FIELD_NAME)
        self.assertNotIn(self.DEV_FIELD_NAME, self.wml_credentials.keys())

    def test_02_initialize_AutoAI_experiment(self):
        TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_03_initialize_optimizer(self):
        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            prediction_type=AutoAI.PredictionType.MULTICLASS,
            prediction_column="column",
            scoring=AutoAI.Metrics.ACCURACY_SCORE,
            t_shirt_size=self.supported_tshirt_size
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_initialize_optimizer_set_unsupported_tshirt_size(self):
        tshirt_size =TShirtSize.M if self.supported_tshirt_size == TShirtSize.L else TShirtSize.L
        print(f"Unsupported tshirt size tested: {tshirt_size}")
        with self.assertRaises(TShirtSizeNotSupported):
            _ = self.experiment.optimizer(
                name=self.OPTIMIZER_NAME,
                prediction_type=AutoAI.PredictionType.MULTICLASS,
                prediction_column="column",
                scoring=AutoAI.Metrics.ACCURACY_SCORE,
                t_shirt_size=tshirt_size
            )

        print("TShirtSizeNotSupported error raised succesfully")

    def test_05_initialize_optimizer_set_unsupported_tshirt_size(self):
        with self.assertRaises(TShirtSizeNotSupported):
            _ = self.experiment.optimizer(
                name=self.OPTIMIZER_NAME,
                prediction_type=AutoAI.PredictionType.MULTICLASS,
                prediction_column="column",
                scoring=AutoAI.Metrics.ACCURACY_SCORE,
                t_shirt_size=TShirtSize.XL
            )

        print("TShirtSizeNotSupported error raised succesfully")

    # development credentials #

    def test_06_wml_credentials_add_development_True(self):

        self.wml_credentials[self.DEV_FIELD_NAME] = True
        self.assertIn(self.DEV_FIELD_NAME, self.wml_credentials.keys())

        if is_cp4d():
            TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials,
                                                 project_id=self.project_id,
                                                 space_id=self.space_id)
        else:
            TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_07_initialize_optimizer(self):
        for tshirt_size in [TShirtSize.S, TShirtSize.M, TShirtSize.L, TShirtSize.XL]:
            try:
                TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
                    name=self.OPTIMIZER_NAME,
                    prediction_type=AutoAI.PredictionType.MULTICLASS,
                    prediction_column="column",
                    scoring=AutoAI.Metrics.ACCURACY_SCORE,
                    t_shirt_size=tshirt_size
                )

                self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                                      msg="experiment.optimizer did not return RemoteAutoPipelines object")
            except TShirtSizeNotSupported as e:
                print("Initializing optimalizer fails with TShirtSizeNotSupported error.")
                self.assertTrue(False,
                                msg=f"Initializing optimalizer fails with TShirtSizeNotSupported error."
                                    f"\n tshirt_size = {tshirt_size}\n {e}")







