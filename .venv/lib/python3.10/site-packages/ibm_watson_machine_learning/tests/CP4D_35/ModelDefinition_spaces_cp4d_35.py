#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *

class TestModelDefn(unittest.TestCase):
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestModelDefn.logger.info("Service Instance: setting up credentials")

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

        TestModelDefn.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestModelDefn.space_id)
        self.client.set.default_space(TestModelDefn.space_id)

    def test_01_create_model_defn_asset(self):

        self.client.model_definitions.ConfigurationMetaNames.show()

        meta_props = {
            self.client.model_definitions.ConfigurationMetaNames.NAME: "Test Model Definition",
            self.client.model_definitions.ConfigurationMetaNames.VERSION: "1.0",
            self.client.model_definitions.ConfigurationMetaNames.PLATFORM: {"name": "python", "versions": ["3.5"]},
            self.client.model_definitions.ConfigurationMetaNames.COMMAND: "test_command"
        }

        print(TestModelDefn.space_id)

        model_defn_details = self.client.model_definitions.store(meta_props = meta_props,
                                                                 model_definition="artifacts/pytorch_onnx_v_1.1.zip")

        print(model_defn_details)
        TestModelDefn.model_defn_asset_id = self.client.model_definitions.get_id(model_defn_details)
        TestModelDefn.model_defn_asset_url = self.client.model_definitions.get_href(model_defn_details)
        TestModelDefn.logger.info("model_defn asset ID:" + str(TestModelDefn.model_defn_asset_id))
        TestModelDefn.logger.info("model_defn asset URL:" + str(TestModelDefn.model_defn_asset_url))
        self.assertIsNotNone(TestModelDefn.model_defn_asset_id)
        self.assertIsNotNone(TestModelDefn.model_defn_asset_url)


    def test_02_download_model_defn_content(self):
        try:
            os.remove('test_model_defn_asset.zip')
        except:
            pass
        self.client.model_definitions.download(TestModelDefn.model_defn_asset_id, filename='test_model_defn_asset.zip')
        try:
            os.remove('test_model_defn_asset.zip')
        except:
            pass

    def test_03_get_details(self):

        details = self.client.model_definitions.get_details(TestModelDefn.model_defn_asset_id)
        print(details)
        self.assertTrue(TestModelDefn.model_defn_asset_id in str(details))

    def test_04_list(self):
        self.client.model_definitions.list()

    def test_05_revisions(self):
        meta_prop_model_defn_1_revision1 = {
            self.client.model_definitions.ConfigurationMetaNames.NAME: "model_definitions revision 1",
            self.client.model_definitions.ConfigurationMetaNames.DESCRIPTION: "model_definitions revision 1"
        }

        # Update meta and attachment for revision 1 creation
        model_defn_update = self.client.model_definitions.update(TestModelDefn.model_defn_asset_id,
                                                  meta_prop_model_defn_1_revision1,
                                                  file_path="artifacts/pytorch_onnx_v_1.1_rev1.zip")

        print(model_defn_update)

        attachment_id = model_defn_update[u'metadata'][u'attachment_id']

        # Create revision
        revision = self.client.model_definitions.create_revision(TestModelDefn.model_defn_asset_id)

        print('revision: ', revision)

        self.assertTrue(revision[u'metadata'][u'revision_id'] and revision[u'metadata'][u'revision_id'] == 1)

        new_attachment_id = revision[u'metadata'][u'attachment_id']

        self.assertTrue(attachment_id != new_attachment_id)

        # List revisions
        self.client.model_definitions.list_revisions(TestModelDefn.model_defn_asset_id)

        try:
            os.remove('test_model_defn_revision.zip')
        except:
            pass

        # Download revision 1 attachment
        self.client.model_definitions.download(TestModelDefn.model_defn_asset_id,
                                    filename='test_model_defn_revision.zip',
                                    rev_id=1)
        try:
            os.remove('test_model_defn_revision.zip')
        except:
            pass

    def test_07_update_entity_model_defn_asset(self):
        meta_prop_model_defn = {
            self.client.model_definitions.ConfigurationMetaNames.VERSION: "2.0",
            self.client.model_definitions.ConfigurationMetaNames.PLATFORM: {"name": "python", "versions": ["3.6"]},
            self.client.model_definitions.ConfigurationMetaNames.COMMAND: "test_command1"
        }

        model_defn_update = self.client.model_definitions.update(TestModelDefn.model_defn_asset_id,
                                                  meta_prop_model_defn)

        print(model_defn_update)

    def test_08_delete_model_defn_asset(self):
        TestModelDefn.logger.info("Delete model definition")
        self.client.model_definitions.delete(TestModelDefn.model_defn_asset_id)

    def test_09_delete_space(self):
        self.client.spaces.delete(TestModelDefn.space_id)

if __name__ == '__main__':
    unittest.main()
