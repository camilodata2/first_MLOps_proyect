#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *

class TestHwSpec(unittest.TestCase):

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestHwSpec.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

    def test_01_list_hw_specs(self):

        hw_spec_details1 = self.client.hardware_specifications.list()
        print(hw_spec_details1)

        hw_spec_details2 = self.client.hardware_specifications.list(name='V100x2')
        print(hw_spec_details2)

    def test_02_get_details(self):

        hw_spec_id = self.client.hardware_specifications.get_uid_by_name('V100x2')

        details = self.client.hardware_specifications.get_details(hw_spec_id)
        print(details)
        self.assertTrue("V100x2" in str(details))

if __name__ == '__main__':
    unittest.main()
