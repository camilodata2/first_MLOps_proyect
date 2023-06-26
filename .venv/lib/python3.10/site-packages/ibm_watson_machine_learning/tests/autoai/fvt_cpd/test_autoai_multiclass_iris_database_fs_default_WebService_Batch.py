#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from ibm_watson_machine_learning.helpers.connections import DataConnection, FSLocation
from ibm_watson_machine_learning.tests.utils import is_cp4d
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import (
    AbstractTestAutoAIRemote)


@unittest.skip("Not ready, training does not support connection_asset")
@unittest.skipIf(not (is_cp4d()), "Supported only on CP4D")
class TestAutoAIRemote(AbstractTestAutoAIRemote, unittest.TestCase):
    """
    The test can be run only on CPD
    The test covers:
    - File system connection set-up
    - downloading training data from filesystem
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline
    - deployment deletion
    Connection used in test:
     - input: File system.
     - output: [None] - default connection depending on environment.
    """

    def test_02_DataConnection_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            location=FSLocation(path=self.data_location)
        )
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    def test_02a_read_saved_remote_data_before_fit(self):
        TestAutoAIRemote.data = self.data_connection.read(csv_separator=self.custom_separator)
        print("Data sample:")
        print(self.data.head())
        self.assertGreater(len(self.data), 0)


if __name__ == '__main__':
    unittest.main()
