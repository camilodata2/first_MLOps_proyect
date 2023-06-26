#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import time

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *
import time
import ibm_watson_machine_learning._wrappers.requests as requests

class TestPkgExtn(unittest.TestCase):

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestPkgExtn.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        # self.space_id = get_project_id()
        # self.client.set.default_project(self.space_id)


        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)
        TestPkgExtn.space_id = self.client.spaces.get_id(self.space)
        # TestRshinyApp.space_id = '9cf73498-72d6-42eb-a6f4-424616de9f45'
        # TestPkgExtn.space_id = '590a241b-756b-4364-a8ae-380b3e6607f7'
        print('space_id: ', TestPkgExtn.space_id)

        print(self.client.spaces.get_details(TestPkgExtn.space_id))
        time.sleep(20)
        print(self.client.spaces.get_details(TestPkgExtn.space_id))
        self.client.set.default_space(TestPkgExtn.space_id)


    def test_00_create_pkg_extn(self):

        meta_prop_pkg_extn = {
            self.client.package_extensions.ConfigurationMetaNames.NAME: "Pkg extension for conda",
            self.client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Pkg extension for conda",
            self.client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml",
        }

        # print(self.client.spaces.get_details(TestPkgExtn.space_id))
        pkg_extn_details = self.client.package_extensions.store(meta_props=meta_prop_pkg_extn,
                                                                file_path="artifacts/conda_regex.yaml")

        TestPkgExtn.pkg_extn_id = self.client.package_extensions.get_id(pkg_extn_details)
        pkg_extn_url = self.client.package_extensions.get_href(pkg_extn_details)
        TestPkgExtn.logger.info("sw spec ID:" + str(TestPkgExtn.pkg_extn_id))
        TestPkgExtn.logger.info("w spec URL:" + str(pkg_extn_url))

        print("pkg_extn_id: {}; pkg_extn_url: {}".format(TestPkgExtn.pkg_extn_id, pkg_extn_url))

        self.assertIsNotNone(TestPkgExtn.pkg_extn_id)
        self.assertIsNotNone(pkg_extn_url)

    def test_01_get_details(self):

        details = self.client.package_extensions.get_details(TestPkgExtn.pkg_extn_id)
        print(details)
        self.assertTrue("Pkg extension for conda" in str(details))

        href = self.wml_credentials[u'url'] + details[u'entity'][u'package_extension'][u'href']

        try:
            os.remove('test.yaml')
        except:
            pass

        response = requests.get(href)

        print('href: ', href)
        print('response: ', response)

        with open('test.yaml', 'wb') as f:
            f.write(response.content)

        try:
            os.remove('test.yaml')
        except:
            pass


    def test_02_list_package_extensions(self):
        self.client.package_extensions.list()

    def test_03_create_sw_spec(self):

        self.client.software_specifications.ConfigurationMetaNames.show()
        # base_sw_spec_id = self.client.software_specifications.get_id_by_name("ai-function_0.1-py3.6")
        base_sw_spec_id = self.client.software_specifications.get_id_by_name("default_py3.7")

        meta_prop_sw_spec = {
            self.client.software_specifications.ConfigurationMetaNames.NAME: "sw_spec test" + str(uuid.uuid4()),
            self.client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification test",
            self.client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION:
                {"guid": base_sw_spec_id}
        }

        sw_spec_details = self.client.software_specifications.store(meta_props=meta_prop_sw_spec)

        print(sw_spec_details)

        TestPkgExtn.sw_spec_id = self.client.software_specifications.get_id(sw_spec_details)
        sw_spec_url = self.client.software_specifications.get_href(sw_spec_details)
        TestPkgExtn.logger.info("sw spec ID:" + str(TestPkgExtn.sw_spec_id))
        TestPkgExtn.logger.info("sw spec URL:" + str(sw_spec_url))

        print("sw_spec_id: {}; sw_spec_url: {}".format(TestPkgExtn.sw_spec_id, sw_spec_url))

        self.assertIsNotNone(TestPkgExtn.sw_spec_id)
        self.assertIsNotNone(sw_spec_url)

    def test_04_add_pkg_extn_to_sw_spec(self):

        self.client.software_specifications.add_package_extension(TestPkgExtn.sw_spec_id,
                                                                  TestPkgExtn.pkg_extn_id)

        ss_asset_details = self.client.software_specifications.get_details(TestPkgExtn.sw_spec_id)
        print('Package extensions',
              print(ss_asset_details['entity']['software_specification']['package_extensions']))

    def test_05_get_details(self):

        details = self.client.software_specifications.get_details(TestPkgExtn.sw_spec_id)
        print(details)
        self.assertTrue("sw_spec test" in str(details))
        self.assertTrue("Pkg extension for conda" in str(details))

    def test_06_del_pkg_extn_to_sw_spec(self):

        self.client.software_specifications.delete_package_extension(TestPkgExtn.sw_spec_id,
                                                                     TestPkgExtn.pkg_extn_id)

    def test_07_get_details(self):

        details = self.client.software_specifications.get_details(TestPkgExtn.sw_spec_id)
        print(details)
        self.assertTrue("sw_spec test" in str(details))
        self.assertTrue("Pkg extension for conda" not in str(details))

    def test_08_list(self):
        self.client.software_specifications.list()

    def test_09_delete_pkg_extn(self):
        TestPkgExtn.logger.info("Delete deployment")
        self.client.package_extensions.delete(TestPkgExtn.pkg_extn_id)

    def test_10_delete_sw_spec(self):
        TestPkgExtn.logger.info("Delete deployment")
        self.client.software_specifications.delete(TestPkgExtn.sw_spec_id)

    def test_11_delete_space(self):
        TestPkgExtn.logger.info("Delete space")
        self.client.spaces.delete(TestPkgExtn.space_id)

if __name__ == '__main__':
    unittest.main()
