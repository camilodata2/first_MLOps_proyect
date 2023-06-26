#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest,time

import logging
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.ICP.models_preparation import *
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *


class TestConnections(unittest.TestCase):

    deployment_uid = None
    function_uid = None
    job_id = None
    scoring_url = None
    connection_asset_uid = None
    connected_data_asset_url = None
    connected_data_id = None
    connection_asset_url = None
    model_uid = None
    space_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestConnections.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_scripts" + time.asctime() })
        TestConnections.space_id = self.client.spaces.get_uid(self.space)
        self.client.set.default_space(TestConnections.space_id)

    def test_01_publish_model(self):
        # space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case"})
        # space_id = self.client.spaces.get_uid(space)
        # self.client.set.default_space(space_id)
        # TestWMLClientWithScikitLearn.space_id = space_id
        # print("The space is" + space_id)
        TestConnections.logger.info("Creating scikit-learn model ...")

        model_data = create_scikit_learn_model_data()
        predicted = model_data['prediction']

        TestConnections.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
                       self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
                       self.client.repository.ModelMetaNames.RUNTIME_UID: "scikit-learn_0.20-py3.6"
                       }
        published_model_details = self.client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], training_target=model_data['training_target'])
        TestConnections.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestConnections.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestConnections.model_uid))
        self.logger.info("Published model URL:" + str(TestConnections.model_url))
        self.assertIsNotNone(TestConnections.model_uid)
        self.assertIsNotNone(TestConnections.model_url)

    def test_02_create_connection_asset(self):
        import json
        self.client.connections.ConfigurationMetaNames.show()
        data_source_type_id = self.client.connections.get_datasource_type_uid_by_name('cloudobjectstorage')
        conn_properties = json.loads(config.get(environment, 'connection_cos_hmac_keys'))
        meta_props_connection = {
            self.client.connections.ConfigurationMetaNames.NAME: "Sample Connection from Python client",
            self.client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: data_source_type_id,
            self.client.connections.ConfigurationMetaNames.DESCRIPTION: "Test connection object using fvt COS instance",
            self.client.connections.ConfigurationMetaNames.PROPERTIES: conn_properties
        }

        conn_details = self.client.connections.create(meta_props_connection)

        TestConnections.connection_asset_uid = self.client.connections.get_uid(conn_details)
        TestConnections.logger.info("Connection asset ID:" + str(TestConnections.connection_asset_uid))
        self.assertIsNotNone(TestConnections.connection_asset_uid)

    def test_03_create_connected_data(self):
        asset_meta_props = {
            self.client.data_assets.ConfigurationMetaNames.NAME: "scikit learn dataasset",
            self.client.data_assets.ConfigurationMetaNames.CONNECTION_ID: TestConnections.connection_asset_uid,
            self.client.data_assets.ConfigurationMetaNames.DESCRIPTION: "Test connection object using fvt COS instance",
            self.client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "/wml-v4-fvt-remote-tests/heart_test.csv"
        }
        asset_details = self.client.data_assets.store(asset_meta_props)
        TestConnections.connected_data_asset_url = self.client.data_assets.get_href(asset_details)
        TestConnections.connected_data_id = self.client.data_assets.get_uid(asset_details)
        self.assertIsNotNone(TestConnections.connected_data_id)
        self.assertIsNotNone(TestConnections.connected_data_asset_url)

    def test_04_get_details(self):

        details = self.client.connections.get_details(TestConnections.connection_asset_uid)
        self.assertTrue(TestConnections.connection_asset_uid in str(details))

    def test_05_list(self):
        self.client.connections.list()

    def test_06_create_deployment(self):
        deploy_meta = {
                self.client.deployments.ConfigurationMetaNames.NAME: "deployment_using_connected_data",
                self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "deployment rscript deployment",
                self.client.deployments.ConfigurationMetaNames.BATCH: {},
                self.client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {"name":"S", "num_nodes":1}
            }

        TestConnections.logger.info("Create deployment")
        deployment = self.client.deployments.create(artifact_uid=TestConnections.model_uid, meta_props=deploy_meta)
        TestConnections.logger.debug("deployment: " + str(deployment))
        # TestScripts.scoring_url = self.client.deployments.get_scoring_href(deployment)
        # TestScripts.logger.debug("Scoring href: {}".format(TestScripts.scoring_url))
        TestConnections.deployment_uid = self.client.deployments.get_uid(deployment)
        TestConnections.logger.debug("Deployment uid: {}".format(TestConnections.deployment_uid))
        self.client.deployments.list()
        self.assertTrue("deployment_using_connected_data" in str(deployment))

    def test_07_create_job(self):
        TestConnections.logger.info("Create job details")
        job_payload_ref = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES: [{
                "name": "test_ref_input",
                "type": "data_asset",
                "connection": {},
                "location": {
                    "href": TestConnections.connected_data_asset_url
                 }
            }],
            self.client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE: {

                "type": "data_asset",
                "connection": {},
                "location": {
                    "name": "scikit_{}.csv".format(TestConnections.deployment_uid),
                    "description": "testing csv results"
                }
            }
        }

        TestConnections.job_details = self.client.deployments.create_job(TestConnections.deployment_uid, meta_props=job_payload_ref)
        TestConnections.job_id = self.client.deployments.get_job_uid(TestConnections.job_details)

    def test_10_get_job_status(self):
        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 10 * 60:
            time.sleep(3)
            response = self.client.deployments.get_job_status(TestConnections.job_id)
            if response['state'] == 'completed' or response['state'] == 'error' or response['state'] == 'failed':
                break
            diff_time = time.time() - start_time

        self.assertIsNotNone(response)
       # self.assertTrue(response['state'] == 'completed')

    def test_11_list_jobs(self):
        self.client.deployments.list_jobs()

    def test_12_delete_job(self):
        self.client.deployments.delete_job(TestConnections.job_id)

    def test_13_delete_deployment(self):
        TestConnections.logger.info("Delete deployment")
        self.client.deployments.delete(TestConnections.deployment_uid)

    def test_14_delete_connection_asset(self):
        TestConnections.logger.info("Delete connection and connected data")
        self.client.data_assets.delete(TestConnections.connected_data_id)
        self.client.connections.delete(TestConnections.connection_asset_uid)

    def test_14_delete_model_asset(self):
        TestConnections.logger.info("Delete model")
        self.client.repository.delete(TestConnections.model_uid)

    def test_15_delete_space(self):
        self.client.spaces.delete(TestConnections.space_id)


if __name__ == '__main__':
    unittest.main()

