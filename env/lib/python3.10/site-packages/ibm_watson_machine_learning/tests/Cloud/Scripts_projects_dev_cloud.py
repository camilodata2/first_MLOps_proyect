#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.Cloud.preparation_and_cleaning import *

class TestScripts(unittest.TestCase):
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestScripts.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.project_id = get_project_id()

        self.client.set.default_project(self.project_id)

        # print(self.client.service_instance.get_details())

        print("project_id: ", self.project_id)

    def test_01_create_script_asset(self):

        self.client.script.ConfigurationMetaNames.show()
        sw_spec_uid = self.client.software_specifications.get_uid_by_name("ai-function_0.1-py3.6")

        print("sw_spec_uid: ", sw_spec_uid)

        meta_prop_script = {
            self.client.script.ConfigurationMetaNames.NAME: "my script asset",
            self.client.script.ConfigurationMetaNames.DESCRIPTION: "script asset for deployment",
            self.client.script.ConfigurationMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid
        }

        script_details = self.client.script.store(meta_prop_script, file_path="artifacts/test.py")

        TestScripts.script_asset_id = self.client.script.get_id(script_details)
        TestScripts.script_asset_url = self.client.script.get_href(script_details)
        TestScripts.logger.info("script asset ID:" + str(TestScripts.script_asset_id))
        TestScripts.logger.info("script asset URL:" + str(TestScripts.script_asset_url))
        self.assertIsNotNone(TestScripts.script_asset_id)
        self.assertIsNotNone(TestScripts.script_asset_url)


    def test_02_download_script_content(self):
        try:
            os.remove('test_script_asset.zip')
        except:
            pass
        self.client.script.download(TestScripts.script_asset_id, filename='test_script_asset.zip')
        # try:
        #     os.remove('test_script_asset.zip')
        # except:
        #     pass

    def test_04_get_details(self):

        details = self.client.script.get_details(TestScripts.script_asset_id)
        print(details)
        self.assertTrue(TestScripts.script_asset_id in str(details))

    def test_05_list(self):
        self.client.script.list()

    def test_06_revisions(self):
        meta_prop_script1_revision1 = {
            self.client.shiny.ConfigurationMetaNames.NAME: "Script revision 1",
            self.client.shiny.ConfigurationMetaNames.DESCRIPTION: "Script revision 1"
        }

        # Update meta and attachment for revision 1 creation
        script_update = self.client.script.update(TestScripts.script_asset_id,
                                                  meta_prop_script1_revision1,
                                                  file_path="artifacts/test1.py.zip")
        attachment_id = script_update[u'metadata'][u'attachment_id']

        # Create revision
        revision = self.client.script.create_revision(TestScripts.script_asset_id)

        self.assertTrue(revision[u'metadata'][u'revision_id'] and revision[u'metadata'][u'revision_id'] == 1)

        new_attachment_id = revision[u'metadata'][u'attachment_id']

        self.assertTrue(attachment_id != new_attachment_id)

        # List revisions
        self.client.script.list_revisions(TestScripts.script_asset_id)

        try:
            os.remove('script1.zip')
        except:
            pass

        # Download revision 1 attachment
        self.client.script.download(TestScripts.script_asset_id,
                                    filename='script1.zip',
                                    rev_uid=1)
        try:
            os.remove('script1.zip')
        except:
            pass

    def test_15_delete_script_asset(self):
        TestScripts.logger.info("Delete function")
        self.client.script.delete(TestScripts.script_asset_id)

if __name__ == '__main__':
    unittest.main()
