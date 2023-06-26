#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.Cloud.preparation_and_cleaning import *

# This is only for dev testing, since otherwise we need to everytime create cos buckets, create projects
# via api, etc. Should be done when possible though
class TestRshinyApp(unittest.TestCase):
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.project_id = get_project_id()

        self.client.set.default_project(self.project_id)

        print(self.client.service_instance.get_details())

        print("project_id: ", self.project_id)

    def test_01_create_shiny_asset(self):

        self.client.shiny.ConfigurationMetaNames.show()

        meta_prop_shiny = {
            self.client.shiny.ConfigurationMetaNames.NAME: "my shiny app project",
            self.client.shiny.ConfigurationMetaNames.DESCRIPTION: "shiny app for project"
        }

        shiny_details = self.client.shiny.store(meta_prop_shiny, file_path="artifacts/app.R.zip")

        TestRshinyApp.shiny_asset_id = self.client.shiny.get_id(shiny_details)
        TestRshinyApp.shiny_asset_url = self.client.shiny.get_href(shiny_details)
        print("shiny asset ID:" + str(TestRshinyApp.shiny_asset_id))
        print("shiny asset URL:" + str(TestRshinyApp.shiny_asset_url))
        self.assertIsNotNone(TestRshinyApp.shiny_asset_id)
        self.assertIsNotNone(TestRshinyApp.shiny_asset_url)

    def test_02_download_shiny_content(self):
        try:
            os.remove('test_shiny_asset.zip')
        except:
            pass
        self.client.shiny.download(TestRshinyApp.shiny_asset_id, filename='test_shiny_asset.zip')
        try:
            os.remove('test_shiny_asset.zip')
        except:
            pass

    def test_03_get_details(self):

        details = self.client.shiny.get_details(TestRshinyApp.shiny_asset_id)
        print(details)
        self.assertTrue(TestRshinyApp.shiny_asset_id in str(details))

    def test_04_list(self):
        self.client.shiny.list()

    def test_05_revisions(self):
        meta_prop_shiny_revision1 = {
            self.client.shiny.ConfigurationMetaNames.NAME: "Shiny revision 1",
            self.client.shiny.ConfigurationMetaNames.DESCRIPTION: "Shiny revision 1"
        }

        # Update meta and attachment for revision 1 creation
        shiny_update = self.client.shiny.update(TestRshinyApp.shiny_asset_id,
                                                meta_prop_shiny_revision1,
                                                file_path="artifacts/app1.R.zip")
        attachment_id = shiny_update[u'metadata'][u'attachment_id']

        # Create revision
        revision = self.client.shiny.create_revision(TestRshinyApp.shiny_asset_id)

        self.assertTrue(revision[u'metadata'][u'revision_id'] and revision[u'metadata'][u'revision_id'] == 1)

        new_attachment_id = revision[u'metadata'][u'attachment_id']

        self.assertTrue(attachment_id != new_attachment_id)

        # List revisions
        self.client.shiny.list_revisions(TestRshinyApp.shiny_asset_id)

        self.client.shiny.get_revision_details(TestRshinyApp.shiny_asset_id, 1)

        try:
            os.remove('test_shiny_asset_rev1.zip')
        except:
            pass

        # Download revision 1 attachment
        self.client.shiny.download(TestRshinyApp.shiny_asset_id, filename='test_shiny_asset_rev1.zip', rev_uid=1)
        try:
            os.remove('test_shiny_asset_rev1.zip')
        except:
            pass

    def test_10_delete_shiny_asset(self):
        TestRshinyApp.logger.info("Delete function")
        self.client.shiny.delete(TestRshinyApp.shiny_asset_id)

if __name__ == '__main__':
    unittest.main()
