#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *

class TestRemoteTrainingSystems(unittest.TestCase):
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):

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

        TestRemoteTrainingSystems.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestRemoteTrainingSystems.space_id)
        self.client.set.default_space(TestRemoteTrainingSystems.space_id)

    def test_01_create_remote_training_system(self):

        self.client.remote_training_systems.ConfigurationMetaNames.show()

        meta_props = {
            self.client.remote_training_systems.ConfigurationMetaNames.NAME: "Remote Training Definition",
            self.client.remote_training_systems.ConfigurationMetaNames.TAGS: ["tag1", "tag2"],
            self.client.remote_training_systems.ConfigurationMetaNames.ORGANIZATION: {"name": "name", "region": "EU"},
            self.client.remote_training_systems.ConfigurationMetaNames.ALLOWED_IDENTITIES: [{"id": "43689024", "type": "user"}],
            self.client.remote_training_systems.ConfigurationMetaNames.REMOTE_ADMIN: {"id": "43689024", "type": "user"}
        }

        print(TestRemoteTrainingSystems.space_id)

        rts_details = self.client.remote_training_systems.store(meta_props = meta_props)

        print(rts_details)

        TestRemoteTrainingSystems.rts_asset_id = self.client.remote_training_systems.get_id(rts_details)
        self.assertIsNotNone(TestRemoteTrainingSystems.rts_asset_id)

    def test_02_get_details(self):

        details = self.client.remote_training_systems.get_details(TestRemoteTrainingSystems.rts_asset_id)
        print(details)
        self.assertTrue(TestRemoteTrainingSystems.rts_asset_id in str(details))

    def test_03_list(self):
        self.client.remote_training_systems.list()

    def test_04_revisions(self):
        meta_prop_rts_1_revision1 = {
            self.client.remote_training_systems.ConfigurationMetaNames.NAME: "remote_training_system_update_rev1",
            self.client.remote_training_systems.ConfigurationMetaNames.ORGANIZATION: {'name': 'updated_org_name'}
        }

        rts_update = self.client.remote_training_systems.update(TestRemoteTrainingSystems.rts_asset_id,
                                                                meta_prop_rts_1_revision1)

        print(rts_update)

        self.assertTrue('remote_training_system_update_rev1' in rts_update[u'metadata'][u'name'])
        self.assertTrue('updated_org_name' in rts_update[u'entity'][u'organization'][u'name'])

        # Create revision
        revision = self.client.remote_training_systems.create_revision(TestRemoteTrainingSystems.rts_asset_id)

        print('revision: ', revision)

        self.assertTrue(revision[u'metadata'][u'rev'] == '1')

        # List revisions
        self.client.remote_training_systems.list_revisions(TestRemoteTrainingSystems.rts_asset_id)

    def test_05_delete_rts_asset(self):
        self.client.remote_training_systems.delete(TestRemoteTrainingSystems.rts_asset_id)

    def test_06_delete_space(self):
        self.client.spaces.delete(TestRemoteTrainingSystems.space_id)

    def test_07_create_party(self):

        party_meta = {
            self.client.remote_training_systems.ConfigurationMetaNames.DATA_HANDLER: {
                "info": {
                    "npz_file": "/artifacts/data_party1.npz"
                },
                "name": "MnistTFDataHandler",
                "path": "/artifacts/mnist_keras_data_handler.py"
            },
            self.client.remote_training_systems.ConfigurationMetaNames.LOCAL_TRAINING: {
                "name": "LocalTrainingHandler",
                "path": "ibmfl.party.training.local_training_handler"
            }
        }

        party = self.client.remote_training_systems.create_party(TestRemoteTrainingSystems.rts_asset_id, party_meta)
        self.assertIsNotNone(party)

if __name__ == '__main__':
    unittest.main()
