#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import time

import logging
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.ICP.models_preparation import *


class TestPkgExtn(unittest.TestCase):
    runtime_uid = None
    deployment_uid = None
    function_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestPkgExtn.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_TestPkgExtnSw"})
        self.space_id = self.client.spaces.get_uid(self.space)
        self.client.set.default_space(self.space_id)

    # def test_01_service_instance_details(self):
    #     TestPkgExtn.logger.info("Check client ...")
    #     self.assertTrue(self.client.__class__.__name__ == 'APIClient')
    #
    #     TestPkgExtn.logger.info("Getting instance details ...")
    #     details = self.client.service_instance.get_details()
    #     TestPkgExtn.logger.debug(details)
    #
    #     self.assertTrue("published_models" in str(details))
    #     self.assertEqual(type(details), dict)


    def test_00_create_pkg_extn(self):


        meta_prop_pkg_extn = {
            self.client.package_extensions.ConfigurationMetaNames.NAME: "Pkg extension for conda",
            self.client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Pkg extension for conda",
            self.client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml",
        }

        pkg_extn_details = self.client.package_extensions.store(meta_props=meta_prop_pkg_extn,
                                                                file_path="artifacts/conda_regex.yaml")

        TestPkgExtn.pkg_extn_uid = self.client.package_extensions.get_uid(pkg_extn_details)
        pkg_extn_url = self.client.package_extensions.get_href(pkg_extn_details)
        TestPkgExtn.logger.info("sw spec ID:" + str(TestPkgExtn.pkg_extn_uid))
        TestPkgExtn.logger.info("w spec URL:" + str(pkg_extn_url))
        self.assertIsNotNone(TestPkgExtn.pkg_extn_uid)
        self.assertIsNotNone(pkg_extn_url)

    def test_01_get_details(self):

        details = self.client.package_extensions.get_details(TestPkgExtn.pkg_extn_uid)
        self.assertTrue("Pkg extension for conda" in str(details))


    def test_02_list_package_extensions(self):
        self.client.package_extensions.list()


    def test_03_create_sw_spec(self):

        self.client.software_specifications.ConfigurationMetaNames.show()
        base_sw_spec_uid = self.client.software_specifications.get_uid_by_name("ai-function_0.1-py3.6")

        meta_prop_sw_spec = {
            self.client.software_specifications.ConfigurationMetaNames.NAME: "sk_learn19_vshasha_new " + time.asctime(),
            self.client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for vshasha",
            self.client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_uid}
        }

        sw_spec_details = self.client.software_specifications.store(meta_props=meta_prop_sw_spec)

        TestPkgExtn.sw_spec_uid = self.client.software_specifications.get_uid(sw_spec_details)
        sw_spec_url = self.client.software_specifications.get_href(sw_spec_details)
        TestPkgExtn.logger.info("sw spec ID:" + str(TestPkgExtn.sw_spec_uid))
        TestPkgExtn.logger.info("sw spec URL:" + str(sw_spec_url))
        self.assertIsNotNone(TestPkgExtn.sw_spec_uid)
        self.assertIsNotNone(sw_spec_url)


    def test_04_add_pkg_extn_to_sw_spec(self):

        self.client.software_specifications.add_package_extension(TestPkgExtn.sw_spec_uid, TestPkgExtn.pkg_extn_uid)


    def test_05_get_details(self):

        details = self.client.software_specifications.get_details(TestPkgExtn.sw_spec_uid)
        self.assertTrue("sk_learn19_vshasha_new" in str(details))
        self.assertTrue("Pkg extension for conda" in str(details))



    def test_06_del_pkg_extn_to_sw_spec(self):
        self.client.software_specifications.delete_package_extension(TestPkgExtn.sw_spec_uid, TestPkgExtn.pkg_extn_uid)


    def test_07_get_details(self):

        details = self.client.software_specifications.get_details(TestPkgExtn.sw_spec_uid)
        self.assertTrue("sk_learn19_vshasha_new" in str(details))
        self.assertTrue("Pkg extension for conda" not in str(details))


    def test_08_list(self):
        self.client.software_specifications.list()

    def test_09_delete_pkg_extn(self):
        TestPkgExtn.logger.info("Delete deployment")
        self.client.package_extensions.delete(TestPkgExtn.pkg_extn_uid)

    def test_10_delete_sw_spec(self):
        TestPkgExtn.logger.info("Delete deployment")
        self.client.software_specifications.delete(TestPkgExtn.sw_spec_uid)

    def test_11_delete_space(self):
        TestPkgExtn.logger.info("Delete space")
        self.client.spaces.delete(TestPkgExtn.space_id)



if __name__ == '__main__':
    unittest.main()
