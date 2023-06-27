#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *
from configparser import RawConfigParser


class TestWMLClientWithSPSS_BIGQUERY_MYSQL(unittest.TestCase):
    deployment_id = None
    model_id = None
    scoring_url = None
    scoring_id = None
    logger = logging.getLogger(__name__)

    connection_mysql_asset_uid = None
    mysql_connected_data_asset_url = None
    mysql_connected_data_id = None

    connection_big_query_asset_uid = None
    big_query_connected_data_asset_url = None
    big_query_connected_data_id = None

    config = RawConfigParser()

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.space_name = str(uuid.uuid4())

        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
            self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestWMLClientWithSPSS_BIGQUERY_MYSQL.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientWithSPSS_BIGQUERY_MYSQL.space_id)
        self.client.set.default_space(TestWMLClientWithSPSS_BIGQUERY_MYSQL.space_id)
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'spss-churn.str')

    def test_01_publish_local_model_in_repository(self):
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Saving trained model in repo ...")
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.debug("Model path: {}".format(self.model_path))

        self.client.repository.ModelMetaNames.show()

        sw_spec_id = self.client.software_specifications.get_id_by_name("spss-modeler_18.1")

        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "spss-churn-model",
                       self.client.repository.ModelMetaNames.TYPE: "spss-modeler_18.1",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                            }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        print(published_model)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.model_id = self.client.repository.get_model_id(published_model)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Published model ID:" + str(TestWMLClientWithSPSS_BIGQUERY_MYSQL.model_id))
        self.assertIsNotNone(TestWMLClientWithSPSS_BIGQUERY_MYSQL.model_id)

    def test_02_get_model_details(self):
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Get model details")
        self.assertIsNotNone(self.client.repository.get_model_details(TestWMLClientWithSPSS_BIGQUERY_MYSQL.model_id))
        
    def test_03_create_connection_mysql__asset(self):
        import json
        self.client.connections.ConfigurationMetaNames.show()
        self.client.connections.list_datasource_types()
        mysql_data_source_type_id = self.client.connections.get_datasource_type_uid_by_name('mysql')
        print(mysql_data_source_type_id)
        conn_properties = json.loads(config.get(environment, 'mysql'))
        meta_props_connection = {
            self.client.connections.ConfigurationMetaNames.NAME: "Sample Connection for mysql from Python client",
            self.client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: mysql_data_source_type_id,
            self.client.connections.ConfigurationMetaNames.DESCRIPTION: "Test connection object using mysql instance",
            self.client.connections.ConfigurationMetaNames.PROPERTIES: conn_properties
        }

        conn_details = self.client.connections.create(meta_props_connection)

        TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_mysql_asset_uid = self.client.connections.get_uid(conn_details)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Connection asset ID:" + str(TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_mysql_asset_uid))
        self.assertIsNotNone(TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_mysql_asset_uid)

    def test_04_create_connected_mysql_data(self):
        asset_meta_props = {
            self.client.data_assets.ConfigurationMetaNames.NAME: "mysql",
            self.client.data_assets.ConfigurationMetaNames.CONNECTION_ID: TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_mysql_asset_uid,
            self.client.data_assets.ConfigurationMetaNames.DESCRIPTION: "mysql",
            self.client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "/CHURN/CHURN3"
            # self.client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "/b4p0kaxbycnhqvfyt3xj/CHURN"
        }
        asset_details = self.client.data_assets.store(asset_meta_props)
        print(asset_details)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.mysql_connected_data_asset_url = self.client.data_assets.get_href(asset_details)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.mysql_connected_data_id = self.client.data_assets.get_uid(asset_details)
        self.assertIsNotNone(TestWMLClientWithSPSS_BIGQUERY_MYSQL.mysql_connected_data_id)
        self.assertIsNotNone(TestWMLClientWithSPSS_BIGQUERY_MYSQL.mysql_connected_data_asset_url)

    def test_05_create_connection_bigquery__asset(self):
        import json
        self.client.connections.ConfigurationMetaNames.show()
        self.client.connections.list_datasource_types()
        big_query_data_source_type_id = self.client.connections.get_datasource_type_uid_by_name('bigquery')
        print(big_query_data_source_type_id)
        conn_properties = json.loads(config.get(environment, 'bigquery'))
        print(conn_properties)
        meta_props_connection = {
            self.client.connections.ConfigurationMetaNames.NAME: "Sample Connection for gbq from Python client",
            self.client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: big_query_data_source_type_id,
            self.client.connections.ConfigurationMetaNames.DESCRIPTION: "Test connection object using gbq instance",
            self.client.connections.ConfigurationMetaNames.PROPERTIES: conn_properties
        }

        conn_details = self.client.connections.create(meta_props_connection)
        print(conn_details)

        TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_big_query_asset_uid = self.client.connections.get_uid(conn_details)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Connection asset ID:" + str(TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_big_query_asset_uid))
        self.assertIsNotNone(TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_big_query_asset_uid)

    def test_06_create_connected_bigquery_data(self):
        asset_meta_props = {
            self.client.data_assets.ConfigurationMetaNames.NAME: "bigquery",
            self.client.data_assets.ConfigurationMetaNames.CONNECTION_ID: TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_big_query_asset_uid,
            self.client.data_assets.ConfigurationMetaNames.DESCRIPTION: "bigquery",
            # self.client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "/CHURN/CHURN3"
            self.client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: "/CHURN/CHURN4"
        }
        asset_details = self.client.data_assets.store(asset_meta_props)
        print(asset_details)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.big_query_connected_data_asset_url = self.client.data_assets.get_href(asset_details)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.big_query_connected_data_id = self.client.data_assets.get_uid(asset_details)
        self.assertIsNotNone(TestWMLClientWithSPSS_BIGQUERY_MYSQL.big_query_connected_data_id)
        self.assertIsNotNone(TestWMLClientWithSPSS_BIGQUERY_MYSQL.big_query_connected_data_asset_url)

    def test_07_create_deployment(self):
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Create deployment for mysql connected data asset")
        deployment_details = self.client.deployments.create(TestWMLClientWithSPSS_BIGQUERY_MYSQL.model_id,
                     meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test spss deployment with mysql",
                                 self.client.deployments.ConfigurationMetaNames.BATCH:{},
                                 self.client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {
                                                                                                     "name": "S",
                                                                                                     "num_nodes": 1
                                                                                               }})
        
        print(deployment_details)

        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.debug("Deployment details: {}".format(deployment_details))

        TestWMLClientWithSPSS_BIGQUERY_MYSQL.deployment_id = self.client.deployments.get_id(deployment_details)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.debug("Deployment uid: {}".format(TestWMLClientWithSPSS_BIGQUERY_MYSQL.deployment_id))

    def test_08_create_job(self):
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Create job details")
        job_payload_ref = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES: [{
                "name": "mysql_ref_input",
                "type": "data_asset",
                "connection": {},
                "location": {
                    "href": TestWMLClientWithSPSS_BIGQUERY_MYSQL.big_query_connected_data_asset_url
                }
            }],
            self.client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE: {

                "type": "data_asset",
                "connection": {},
                "location": {
                    "href": TestWMLClientWithSPSS_BIGQUERY_MYSQL.mysql_connected_data_asset_url
                }
            }
        }

        TestWMLClientWithSPSS_BIGQUERY_MYSQL.job_details = self.client.deployments.create_job(TestWMLClientWithSPSS_BIGQUERY_MYSQL.deployment_id,
                                                                               meta_props=job_payload_ref)
        print(TestWMLClientWithSPSS_BIGQUERY_MYSQL.job_details)
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.job_id = self.client.deployments.get_job_uid(TestWMLClientWithSPSS_BIGQUERY_MYSQL.job_details)

    def test_09_get_job_status(self):
        import time

        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 10 * 60:
            time.sleep(3)
            response = self.client.deployments.get_job_status(TestWMLClientWithSPSS_BIGQUERY_MYSQL.job_id)
            if response['state'] == 'completed' or response['state'] == 'error' or response['state'] == 'failed':
                break
            diff_time = time.time() - start_time

        print(response)
        self.assertIsNotNone(response)
       # self.assertTrue(response['state'] == 'completed')

    def test_10_list_jobs(self):
        self.client.deployments.list_jobs()

    def test_11_delete_job(self):
        self.client.deployments.delete_job(TestWMLClientWithSPSS_BIGQUERY_MYSQL.job_id)

    def test_12_delete_deployment(self):
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithSPSS_BIGQUERY_MYSQL.deployment_id)
        
    def test_13_delete_connection_asset(self):
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Delete connection and connected data")
        self.client.data_assets.delete(TestWMLClientWithSPSS_BIGQUERY_MYSQL.mysql_connected_data_id)
        self.client.connections.delete(TestWMLClientWithSPSS_BIGQUERY_MYSQL.connection_mysql_asset_uid)

    def test_14_delete_model(self):
        TestWMLClientWithSPSS_BIGQUERY_MYSQL.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithSPSS_BIGQUERY_MYSQL.model_id)
        # self.client.spaces.delete(TestWMLClientWithSPSS_BIGQUERY_MYSQL.space_id)

    def test_15_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithSPSS_BIGQUERY_MYSQL.space_id)

if __name__ == '__main__':
    unittest.main()
