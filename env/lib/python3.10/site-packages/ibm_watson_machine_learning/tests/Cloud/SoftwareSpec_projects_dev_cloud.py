#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import logging
from ibm_watson_machine_learning.tests.Cloud.preparation_and_cleaning import *

class TestSwSpec(unittest.TestCase):

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestSwSpec.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.project_id = get_project_id()

        self.client.set.default_project(self.project_id)

        # print(self.client.service_instance.get_details())

        print("project_id: ", self.project_id)

    def test_01_create_sw_spec(self):

        self.client.software_specifications.ConfigurationMetaNames.show()
        base_sw_spec_uid = self.client.software_specifications.get_uid_by_name("ai-function_0.1-py3.6")

        print(base_sw_spec_uid)

        meta_prop_sw_spec = {
            self.client.software_specifications.ConfigurationMetaNames.NAME: "test_sw_spec_" + str(uuid.uuid4()),
            self.client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for test",
            self.client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_uid}
        }

        sw_spec_details = self.client.software_specifications.store(meta_props=meta_prop_sw_spec)

        print(sw_spec_details)

        TestSwSpec.sw_spec_uid = self.client.software_specifications.get_uid(sw_spec_details)
        sw_spec_url = self.client.software_specifications.get_href(sw_spec_details)
        TestSwSpec.logger.info("sw spec ID:" + str(TestSwSpec.sw_spec_uid))
        TestSwSpec.logger.info("w spec URL:" + str(sw_spec_url))
        self.assertIsNotNone(TestSwSpec.sw_spec_uid)
        self.assertIsNotNone(sw_spec_url)

    # def test_02_create_sw_spec(self):
    #
    #     self.client.software_specifications.ConfigurationMetaNames.show()
    #     base_sw_spec_uid = self.client.software_specifications.get_uid_by_name("ai-function_0.1-py3.6")
    #
    #     meta_prop_sw_spec = {
    #         self.client.software_specifications.ConfigurationMetaNames.NAME: "test_sw_spec_software_configuration_" +
    #                                                                          str(uuid.uuid4()),
    #         self.client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for test",
    #         self.client.software_specifications.ConfigurationMetaNames.SOFTWARE_CONFIGURATION: {"platform": {
    #             "name": "python",
    #             "version": "3.6"
    #           }
    #         }
    #     }
    #
    #     sw_spec_details = self.client.software_specifications.store(meta_props=meta_prop_sw_spec)
    #
    #     TestSwSpec.sw_spec_uid1 = self.client.software_specifications.get_uid(sw_spec_details)
    #     sw_spec_url = self.client.software_specifications.get_href(sw_spec_details)
    #     TestSwSpec.logger.info("sw spec ID:" + str(TestSwSpec.sw_spec_uid1))
    #     TestSwSpec.logger.info("w spec URL:" + str(sw_spec_url))
    #     self.assertIsNotNone(TestSwSpec.sw_spec_uid1)
    #     self.assertIsNotNone(sw_spec_url)

    def test_03_get_details(self):

        details = self.client.software_specifications.get_details(TestSwSpec.sw_spec_uid)
        self.assertTrue("test_sw_spec_" in str(details))

        # details = self.client.software_specifications.get_details(TestSwSpec.sw_spec_uid1)
        # self.assertTrue("test_sw_spec_software_configuration_" in str(details))

    def test_04_list(self):
        self.client.software_specifications.list()

    def test_05_delete_sw_spec(self):
        TestSwSpec.logger.info("Delete Software spec")
        self.client.software_specifications.delete(TestSwSpec.sw_spec_uid)
        # self.client.software_specifications.delete(TestSwSpec.sw_spec_uid1)

if __name__ == '__main__':
    unittest.main()
