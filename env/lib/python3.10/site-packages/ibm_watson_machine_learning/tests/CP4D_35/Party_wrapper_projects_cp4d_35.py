#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *
from ibm_watson_machine_learning.party_wrapper import Party


class TestPartyWrapper(unittest.TestCase):
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.host = self.wml_credentials['url'].split('//')[1]
        meta_props = {
            self.client.remote_training_systems.ConfigurationMetaNames.NAME: "Remote Training Definition",
            self.client.remote_training_systems.ConfigurationMetaNames.TAGS: ["tag1", "tag2"],
            self.client.remote_training_systems.ConfigurationMetaNames.ORGANIZATION: {"name": "name", "region": "EU"},
            self.client.remote_training_systems.ConfigurationMetaNames.ALLOWED_IDENTITIES: [{"id": "43689024", "type": "user"}],
            self.client.remote_training_systems.ConfigurationMetaNames.REMOTE_ADMIN: {"id": "43689024", "type": "user"}
        }
        self.client.set.default_project(get_project_id())
        self.rts_details = self.client.remote_training_systems.store(meta_props = meta_props)

    def test_01_create_party_wrapper(self):

        party_config = {
            "aggregator": {
                "ip": self.host
            },
            "connection": {
                "info": {
                    "id": self.client.remote_training_systems.get_id(self.rts_details),
                }
            },
            "data": {
                "info": {
                    "npz_file": "/artifacts/data_party1.npz"
                },
                "name": "MnistTFDataHandler",
                "path": "/artifacts/mnist_keras_data_handler.py"
            },
            "local_training": {
                "name": "LocalTrainingHandler",
                "path": "ibmfl.party.training.local_training_handler"
            },
            "protocol_handler": {
                "name": "PartyProtocolHandler",
                "path": "ibmfl.party.party_protocol_handler"
            }
        }
        TestPartyWrapper.party_wrapper = Party(client=self.client, config_dict=party_config)

        self.assertIsNotNone(TestPartyWrapper.party_wrapper)

    def test_02_monitor_logs(self):

        TestPartyWrapper.party_wrapper.monitor_logs(log_level="DEBUG")
        self.assertTrue(TestPartyWrapper.party_wrapper.log_level == "DEBUG")

    def test_03_run_with_invalid_agg_id(self):

        TestPartyWrapper.party_wrapper.monitor_logs()
        agg_id = "00035adf"
        with self.assertRaises(Exception):
            TestPartyWrapper.party_wrapper.run(aggregator_id=agg_id, asynchronous=False)

    def test_04_monitor_metrics(self):

        TestPartyWrapper.party_wrapper.monitor_metrics()
        self.assertTrue("metrics_recorder" in TestPartyWrapper.party_wrapper.args.get('config_dict'))

    def test_05_is_running(self):

        is_running = TestPartyWrapper.party_wrapper.is_running
        self.assertFalse(is_running)

if __name__ == '__main__':
    unittest.main()




