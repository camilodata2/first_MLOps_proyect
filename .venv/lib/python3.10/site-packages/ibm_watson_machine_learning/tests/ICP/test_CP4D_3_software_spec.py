#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import time

import logging
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.ICP.models_preparation import *


class TestSwSpec(unittest.TestCase):
    runtime_uid = None
    deployment_uid = None
    function_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestSwSpec.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_TestSwSpecSw"})
        self.space_id = self.client.spaces.get_uid(self.space)
        self.client.set.default_space(self.space_id)

    # def test_01_service_instance_details(self):
    #     TestSwSpec.logger.info("Check client ...")
    #     self.assertTrue(self.client.__class__.__name__ == 'APIClient')
    #
    #     TestSwSpec.logger.info("Getting instance details ...")
    #     details = self.client.service_instance.get_details()
    #     TestSwSpec.logger.debug(details)
    #
    #     self.assertTrue("published_models" in str(details))
    #     self.assertEqual(type(details), dict)

    def test_01_create_sw_spec(self):

        self.client.software_specifications.ConfigurationMetaNames.show()
        base_sw_spec_uid = self.client.software_specifications.get_uid_by_name("ai-function_0.1-py3.6")

        meta_prop_sw_spec = {
            self.client.software_specifications.ConfigurationMetaNames.NAME: "sk_learn19_vshasha_new " + time.asctime(),
            self.client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for vshasha",
            self.client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_uid}
        }

        sw_spec_details = self.client.software_specifications.store(meta_props=meta_prop_sw_spec)

        TestSwSpec.sw_spec_uid = self.client.software_specifications.get_uid(sw_spec_details)
        sw_spec_url = self.client.software_specifications.get_href(sw_spec_details)
        TestSwSpec.logger.info("sw spec ID:" + str(TestSwSpec.sw_spec_uid))
        TestSwSpec.logger.info("w spec URL:" + str(sw_spec_url))
        self.assertIsNotNone(TestSwSpec.sw_spec_uid)
        self.assertIsNotNone(sw_spec_url)


    def test_02_get_details(self):

        details = self.client.software_specifications.get_details(TestSwSpec.sw_spec_uid)
        self.assertTrue("sk_learn19_vshasha_new" in str(details))

    def test_03_list(self):
        self.client.software_specifications.list()


    def test_04_delete_sw_spec(self):
        TestSwSpec.logger.info("Delete deployment")
        self.client.software_specifications.delete(TestSwSpec.sw_spec_uid)

    def test_05_delete_space(self):
        TestSwSpec.logger.info("Delete space")
        self.client.spaces.delete(TestSwSpec.space_id)



if __name__ == '__main__':
    unittest.main()
