#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
import pandas as pd
import numpy as np
import random

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_space_id

logger = logging.getLogger("automl")


class TestDataTypes(unittest.TestCase):
    data_asset_id = "80d60d68-a310-4a92-aade-e25f17ed5aaa"
    wml_credentials = get_wml_credentials().copy()

    def test_01_assert_correct_data_types(self):
        client = APIClient(self.wml_credentials)
        client.set.default_space(get_space_id(client, "TRUE_FALSE_space"))
        teacher_data = DataConnection(data_asset_id=self.data_asset_id)
        teacher_data.set_client(client)
        df = teacher_data.read(use_flight=True)
        print(df.dtypes)

        self.assertEqual(bool, df['Resign?'].dtypes)

    # TODO Write test to check what happens when there are string-booleans and other metrics in one column
    # (for example: "TRUE", "FALSE", "Unknown") and we run AutoAI experiment on it
    def test_02_test_fairness_metrics_casting(self):
        size = 500
        values = ["TRUE", "FALSE", "Unknown"]
        target_column = []
        for _ in range(size):
            target_column.append(random.choice(values))

        data = {"x": np.random.randint(-100, 100, size), "y": target_column}
        df = pd.DataFrame(data)
        print("\n", df.head(10))