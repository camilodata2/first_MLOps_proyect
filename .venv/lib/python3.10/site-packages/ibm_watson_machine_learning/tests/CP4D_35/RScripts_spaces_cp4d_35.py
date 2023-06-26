#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *

class TestRScripts(unittest.TestCase):
    runtime_uid = None
    deployment_uid = None
    function_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestRScripts.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestRScripts.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestRScripts.space_id)
        self.client.set.default_space(TestRScripts.space_id)

        # self.space_id = get_project_id()
        # self.client.set.default_project(self.space_id)

    def test_01_create_script_asset(self):

        self.client.script.ConfigurationMetaNames.show()
        # sw_spec_uid = self.client.software_specifications.get_uid_by_name("ai-function_0.1-py3.6")

        # sw_spec_uid = '0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda'
        sw_spec_id = self.client.software_specifications.get_id_by_name('default_r3.6')

        print("sw_spec_id: ", sw_spec_id)

        meta_prop_script = {
            self.client.script.ConfigurationMetaNames.NAME: "my R script asset",
            self.client.script.ConfigurationMetaNames.DESCRIPTION: "R script asset for deployment",
            self.client.script.ConfigurationMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
        }

        script_details = self.client.script.store(meta_prop_script, file_path="artifacts/test.R")

        TestRScripts.script_asset_id = self.client.script.get_id(script_details)
        TestRScripts.script_asset_url = self.client.script.get_href(script_details)
        TestRScripts.logger.info("script asset ID:" + str(TestRScripts.script_asset_id))
        TestRScripts.logger.info("script asset URL:" + str(TestRScripts.script_asset_url))
        self.assertIsNotNone(TestRScripts.script_asset_id)
        self.assertIsNotNone(TestRScripts.script_asset_url)


    def test_02_download_script_content(self):
        try:
            os.remove('test_script_asset.R')
        except:
            pass
        self.client.script.download(TestRScripts.script_asset_id, filename='test_script_asset.R')
        try:
            os.remove('test_script_asset.R')
        except:
            pass

    def test_04_get_details(self):
        details = self.client.script.get_details(TestRScripts.script_asset_id)
        print(json.dumps(details, indent=4))
        self.assertTrue(TestRScripts.script_asset_id in str(details))
        print(json.dumps(self.client.script.get_details(), indent=4))

    def test_05_list(self):
        self.client.script.list()

    def test_06_revisions(self):
        meta_prop_script1_revision1 = {
            self.client.script.ConfigurationMetaNames.NAME: "R Script revision 1",
            self.client.script.ConfigurationMetaNames.DESCRIPTION: "R Script revision 1"
        }

        # Update meta and attachment for revision 1 creation
        script_update = self.client.script.update(TestRScripts.script_asset_id,
                                                  meta_prop_script1_revision1,
                                                  file_path="artifacts/test1.R")

        print(script_update)

        attachment_id = script_update[u'metadata'][u'attachment_id']

        # Create revision
        revision = self.client.script.create_revision(TestRScripts.script_asset_id)

        self.assertTrue(revision[u'metadata'][u'revision_id'] == 1)

        new_attachment_id = revision[u'metadata'][u'attachment_id']

        self.assertTrue(attachment_id != new_attachment_id)

        # List revisions
        self.client.script.list_revisions(TestRScripts.script_asset_id)

        try:
            os.remove('script1.R')
        except:
            pass

        # Download revision 1 attachment
        self.client.script.download(TestRScripts.script_asset_id,
                                    filename='script1.R',
                                    rev_uid=1)
        try:
            os.remove('script1.R')
        except:
            pass


    def test_07_create_deployment(self):
        deploy_meta = {
                self.client.deployments.ConfigurationMetaNames.NAME: "deployment_rscript",
                self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment rscript deployment",
                self.client.deployments.ConfigurationMetaNames.BATCH: {},
                self.client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {"name":"S", "num_nodes":1}
            }

        TestRScripts.logger.info("Create deployment")
        deployment = self.client.deployments.create(artifact_uid=TestRScripts.script_asset_id, meta_props=deploy_meta)
        TestRScripts.logger.debug("deployment: " + str(deployment))
        # TestRScripts.scoring_url = self.client.deployments.get_scoring_href(deployment)
        # TestRScripts.logger.debug("Scoring href: {}".format(TestRScripts.scoring_url))
        TestRScripts.deployment_uid = self.client.deployments.get_uid(deployment)
        TestRScripts.logger.debug("Deployment uid: {}".format(TestRScripts.deployment_uid))
        self.client.deployments.list()
        self.assertTrue("deployment_rscript" in str(deployment))

    def test_08_update_deployment(self):
        patch_meta = {
            self.client.deployments.ConfigurationMetaNames.NAME: "updated_name",
            self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment_Updated_Script_Description",
        }
        self.client.deployments.update(TestRScripts.deployment_uid, patch_meta)

    def test_09_get_deployment_details(self):
        TestRScripts.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        print(deployment_details)
        TestRScripts.logger.debug("Deployment details: {}".format(deployment_details))
        self.assertTrue('deployment_Updated_Script_Description' in str(deployment_details))


    def test_10_create_job(self):
        TestRScripts.logger.info("Create job details")

        TestRScripts.data_asset_details = self.client.data_assets.create("input_file",file_path="artifacts/testR.csv")

        TestRScripts.data_asset_uid = self.client.data_assets.get_uid(TestRScripts.data_asset_details)
        TestRScripts.data_asset_href = self.client.data_assets.get_href(TestRScripts.data_asset_details)
        TestRScripts.logger.debug("Create job")

        TestRScripts.logger.info("data asset ID:" + str(TestRScripts.data_asset_uid))
        TestRScripts.logger.info("data asset URL:" + str(TestRScripts.data_asset_href))
        self.assertIsNotNone(TestRScripts.script_asset_id)
        self.assertIsNotNone(TestRScripts.script_asset_url)

        job_payload_ref = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES: [{
                "name": "test_ref_input",
                "type": "data_asset",
                "connection": {},
                "location": {
                    "href": TestRScripts.data_asset_href
                }
            }],
            self.client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE: {
                "type": "data_asset",
                "connection": {},
                "location": {
                    "name": "scripts_result_{}.zip".format(TestRScripts.deployment_uid),
                    "description": "testing zip results"
                }
            }
        }

        TestRScripts.job_details = self.client.deployments.create_job(TestRScripts.deployment_uid, meta_props=job_payload_ref)
        print(TestRScripts.job_details)
        TestRScripts.job_id = self.client.deployments.get_job_uid(TestRScripts.job_details)

    def test_11_get_job_status(self):
        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 10 * 60:
            time.sleep(3)
            response = self.client.deployments.get_job_status(TestRScripts.job_id)
            print(response)
            if response['state'] == 'completed' or response['state'] == 'error' or response['state'] == 'failed':
                break
            diff_time = time.time() - start_time

        self.assertIsNotNone(response)
        self.assertTrue(response['state'] == 'completed')

        job_details = self.client.deployments.get_job_details(TestRScripts.job_id)
        print(job_details)


    def test_12_list_jobs(self):
        self.client.deployments.list_jobs()


    def test_13_delete_job(self):
        self.client.deployments.delete_job(TestRScripts.job_id)

    def test_14_delete_deployment(self):
        TestRScripts.logger.info("Delete deployment")
        self.client.deployments.delete(TestRScripts.deployment_uid)

    def test_15_delete_script_asset(self):
        TestRScripts.logger.info("Delete function")
        self.client.script.delete(TestRScripts.script_asset_id)
        self.client.data_assets.delete(TestRScripts.data_asset_uid)

    def test_16_delete_space(self):
        self.client.spaces.delete(TestRScripts.space_id)

if __name__ == '__main__':
    unittest.main()
