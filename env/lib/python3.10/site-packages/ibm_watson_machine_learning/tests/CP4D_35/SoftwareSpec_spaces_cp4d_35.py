#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *

class TestSwSpec(unittest.TestCase):

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestSwSpec.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        # self.space_id = get_project_id()
        # self.client.set.default_project(self.space_id)
        #
        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestSwSpec.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestSwSpec.space_id)
        self.client.set.default_space(TestSwSpec.space_id)
        sw_spec_details = self.client.software_specifications.list()
        print(sw_spec_details)

        sw_spec_id = self.client.software_specifications.get_id_by_name('ai-function_0.1-py3.6')

        details = self.client.software_specifications.get_details(sw_spec_id)
        print(details)

        self.client.set.default_space(TestSwSpec.space_id)
        #

    def test_01_create_sw_spec(self):

        self.client.software_specifications.ConfigurationMetaNames.show()
        base_sw_spec_id = self.client.software_specifications.get_id_by_name("ai-function_0.1-py3.6")

        print(base_sw_spec_id)

        meta_prop_sw_spec = {
            self.client.software_specifications.ConfigurationMetaNames.NAME: "test_sw_spec_" + str(uuid.uuid4()),
            self.client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for test",
            self.client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_id}
        }

        sw_spec_details = self.client.software_specifications.store(meta_props=meta_prop_sw_spec)

        print(sw_spec_details)

        TestSwSpec.sw_spec_id = self.client.software_specifications.get_id(sw_spec_details)
        sw_spec_url = self.client.software_specifications.get_href(sw_spec_details)
        TestSwSpec.logger.info("sw spec ID:" + str(TestSwSpec.sw_spec_id))
        TestSwSpec.logger.info("w spec URL:" + str(sw_spec_url))
        self.assertIsNotNone(TestSwSpec.sw_spec_id)
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

        details = self.client.software_specifications.get_details(TestSwSpec.sw_spec_id)
        self.assertTrue("test_sw_spec_" in str(details))

        # details = self.client.software_specifications.get_details(TestSwSpec.sw_spec_id1)
        # self.assertTrue("test_sw_spec_software_configuration_" in str(details))

    def test_04_list(self):
        self.client.software_specifications.list()

    def test_05_delete_sw_spec(self):
        TestSwSpec.logger.info("Delete Software spec")
        self.client.software_specifications.delete(TestSwSpec.sw_spec_id)
        # self.client.software_specifications.delete(TestSwSpec.sw_spec_id1)

    def test_06_delete_space(self):
        TestSwSpec.logger.info("Delete space")
        self.client.spaces.delete(TestSwSpec.space_id)

if __name__ == '__main__':
    unittest.main()
