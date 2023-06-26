#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *

class TestRshinyApp(unittest.TestCase):
    runtime_uid = None
    deployment_uid = None
    function_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)
    shiny_asset_id = None

    @classmethod
    def setUpClass(self):
        TestRshinyApp.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        # self.space_id = get_project_id()
        # self.client.set.default_project(self.space_id)
        #
        # print(self.wml_credentials)
        # print(self.space_id)


        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestRshinyApp.space_id = self.client.spaces.get_id(self.space)
        # TestRshinyApp.space_id = '9cf73498-72d6-42eb-a6f4-424616de9f45'
        print('space_id: ', TestRshinyApp.space_id)
        self.client.set.default_space(TestRshinyApp.space_id)

    # def test_01_service_instance_details(self):
    #     TestAIFunction.logger.info("Check client ...")
    #     self.assertTrue(self.client.__class__.__name__ == 'APIClient')
    #
    #     TestAIFunction.logger.info("Getting instance details ...")
    #     details = self.client.service_instance.get_details()
    #     TestAIFunction.logger.debug(details)
    #
    #     self.assertTrue("published_models" in str(details))
    #     self.assertEqual(type(details), dict)

    def test_01_create_shiny_asset(self):

        self.client.shiny.ConfigurationMetaNames.show()


        meta_prop_shiny = {
            self.client.shiny.ConfigurationMetaNames.NAME: "my shiny app",
            self.client.shiny.ConfigurationMetaNames.DESCRIPTION: "shiny app for deployment"
        }

        shiny_details = self.client.shiny.store(meta_prop_shiny, file_path="artifacts/app.R.zip")

        print(shiny_details)

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

        print('update: ', shiny_update)

        attachment_id = shiny_update[u'metadata'][u'attachment_id']

        # Create revision
        revision = self.client.shiny.create_revision(TestRshinyApp.shiny_asset_id)

        print('revision: ', revision)

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

    def test_06_create_deployment(self):
        deploy_meta = {
                self.client.deployments.ConfigurationMetaNames.NAME: "deployment_rshiny",
                self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment rshiny deployment",
                self.client.deployments.ConfigurationMetaNames.R_SHINY: {"authentication" : "anyone_with_url" },
                self.client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {"name":"S", "num_nodes":1}
            }

        TestRshinyApp.logger.info("Create deployment")
        deployment = self.client.deployments.create(artifact_uid=TestRshinyApp.shiny_asset_id, meta_props=deploy_meta)
        TestRshinyApp.logger.debug("deployment: " + str(deployment))
        # TestRshinyApp.scoring_url = self.client.deployments.get_scoring_href(deployment)
        # TestRshinyApp.logger.debug("Scoring href: {}".format(TestRshinyApp.scoring_url))
        TestRshinyApp.deployment_uid = self.client.deployments.get_uid(deployment)
        TestRshinyApp.logger.debug("Deployment uid: {}".format(TestRshinyApp.deployment_uid))
        self.client.deployments.list()
        self.assertTrue("deployment_rshiny" in str(deployment))

    def test_07_update_deployment(self):
        patch_meta = {
            self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment_Updated_Shiny_Description",
        }
        self.client.deployments.update(TestRshinyApp.deployment_uid, patch_meta)

    def test_08_get_deployment_details(self):
        TestRshinyApp.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        print(deployment_details)
        TestRshinyApp.logger.debug("Deployment details: {}".format(deployment_details))
        self.assertTrue('deployment_Updated_Shiny_Description' in str(deployment_details))

    def test_09_delete_deployment(self):
        TestRshinyApp.logger.info("Delete deployment")
        self.client.deployments.delete(TestRshinyApp.deployment_uid)

    def test_10_delete_shiny_asset(self):
        TestRshinyApp.logger.info("Delete function")
        self.client.shiny.delete(TestRshinyApp.shiny_asset_id)
        self.client.spaces.delete(TestRshinyApp.space_id)


if __name__ == '__main__':
    unittest.main()
