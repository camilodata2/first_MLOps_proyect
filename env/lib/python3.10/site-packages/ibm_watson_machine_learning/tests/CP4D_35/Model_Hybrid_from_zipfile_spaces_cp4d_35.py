#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import time
import pandas as pd
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *


class TestWMLClientWithHybrid(unittest.TestCase):
    deployment_id = None
    model_id = None
    scoring_url = None
    scoring_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithHybrid.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()





        self.space_name = str(uuid.uuid4())

        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
            self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestWMLClientWithHybrid.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientWithHybrid.space_id)
        self.client.set.default_space(TestWMLClientWithHybrid.space_id)
       # self.model_path = os.path.join(os.getcwd(), 'artifacts', 'customer-satisfaction-prediction.str')

        # TestWMLClientWithDO.logger.info("Service Instance: setting up credentials")
        # self.wml_credentials = get_wml_credentials()
        # self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'DrugSelectionAutoAI_model_content.gzip')
        self.update_model_path = os.path.join(os.getcwd(), 'artifacts', 'pipeline-model.json')

    # def test_01_set_space(self):
    #     space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "DO_test_case"})
    #
    #     TestWMLClientWithDO.space_id = self.client.spaces.get_id(space)
    #     self.client.set.default_space(TestWMLClientWithDO.space_id)
    #     self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithDO.space_id))


    def test_02_publish_hybrid_model_in_repository(self):
        TestWMLClientWithHybrid.logger.info("Saving trained model in repo ...")
        TestWMLClientWithHybrid.logger.debug("Model path: {}".format(self.model_path))

        self.client.repository.ModelMetaNames.show()

        sw_spec_id = self.client.software_specifications.get_id_by_name("hybrid_0.1")
        print(sw_spec_id)
        if sw_spec_id is None:
            sw_spec_id = "8c1a58c6-62b5-4dc4-987a-df751c2756b6"
        print(sw_spec_id)
        output_data_schema = [{'id': 'stest',
                              'type': 'list',
                              'fields': [{'name': 'age', 'type': 'float'},
                                         {'name': 'sex', 'type': 'float'},
                                         {'name': 'cp', 'type': 'float'},
                                         {'name': 'restbp', 'type': 'float'},
                                         {'name': 'chol', 'type': 'float'},
                                         {'name': 'fbs', 'type': 'float'},
                                         {'name': 'restecg', 'type': 'float'},
                                         {'name': 'thalach', 'type': 'float'},
                                         {'name': 'exang', 'type': 'float'},
                                         {'name': 'oldpeak', 'type': 'float'},
                                         {'name': 'slope', 'type': 'float'},
                                         {'name': 'ca', 'type': 'float'},
                                         {'name': 'thal', 'type': 'float'}]
                               }, {'id': 'teste2',
                 'type': 'test',
                 'fields': [{'name': 'age', 'type': 'float'},
                            {'name': 'sex', 'type': 'float'},
                            {'name': 'cp', 'type': 'float'},
                            {'name': 'restbp', 'type': 'float'},
                            {'name': 'chol', 'type': 'float'},
                            {'name': 'fbs', 'type': 'float'},
                            {'name': 'restecg', 'type': 'float'},
                            {'name': 'thalach', 'type': 'float'},
                            {'name': 'exang', 'type': 'float'},
                            {'name': 'oldpeak', 'type': 'float'},
                            {'name': 'slope', 'type': 'float'},
                            {'name': 'ca', 'type': 'float'},
                            {'name': 'thal', 'type': 'float'}]}]
        training_data_ref = [{'connection': {'endpoint_url': '',
                         'access_key_id': '',
                         'secret_access_key': ''},
          'location': {'bucket': '', 'path': ''},
          'type': 'fs',
          'schema': {'id': '4cdb0a0a-1c69-43a0-a8c0-3918afc7d45f',
                     'fields': [{'metadata': {'name': 'AGE', 'scale': 0},
                                 'name': 'AGE',
                                 'nullable': True,
                                 'type': 'integer'},
                                {'metadata': {'name': 'SEX', 'scale': 0},
                                 'name': 'SEX',
                                 'nullable': True,
                                 'type': 'string'},
                                {'metadata': {'name': 'BP', 'scale': 0},
                                 'name': 'BP',
                                 'nullable': True,
                                 'type': 'string'},
                                {'metadata': {'name': 'CHOLESTEROL', 'scale': 0},
                                 'name': 'CHOLESTEROL',
                                 'nullable': True,
                                 'type': 'string'},
                                {'metadata': {'name': 'NA', 'scale': 6},
                                 'name': 'NA',
                                 'nullable': True,
                                 'type': 'decimal(12,6)'},
                                {'metadata': {'name': 'K', 'scale': 6},
                                 'name': 'K',
                                 'nullable': True,
                                 'type': 'decimal(13,6)'}],
                     'type': 'struct'}}]
        input_data_schema = [{'id': 'auto_ai_kb_input_schema',
                              'fields': [{'name': 'AGE', 'type': 'int64'},
                                         {'name': 'SEX', 'type': 'object'},
                                         {'name': 'BP', 'type': 'object'},
                                         {'name': 'CHOLESTEROL', 'type': 'object'},
                                         {'name': 'NA', 'type': 'float64'},
                                         {'name': 'K', 'type': 'float64'}]}]

        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "DrugSelection Model",
                       self.client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
                       self.client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES: training_data_ref,
                       self.client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: input_data_schema
                            }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        TestWMLClientWithHybrid.model_id = self.client.repository.get_model_id(published_model)
        TestWMLClientWithHybrid.logger.info("Published model ID:" + str(TestWMLClientWithHybrid.model_id))
        self.assertIsNotNone(TestWMLClientWithHybrid.model_id)


    def test_04_get_details(self):
        TestWMLClientWithHybrid.logger.info("Get details")
        self.assertIsNotNone(self.client.repository.get_details())
        det = self.client.repository.get_details()
        print(det)

    def test_05_get_model_details(self):
        TestWMLClientWithHybrid.logger.info("Get model details")
        details = self.client.repository.get_details(TestWMLClientWithHybrid.model_id)
        import json
        print(details)
        self.assertIsNotNone(self.client.repository.get_model_details(TestWMLClientWithHybrid.model_id))

    # def test_06_create_deployment(self):
    #     TestWMLClientWithHybrid.logger.info("Create deployment")
    #     hw_id = self.client.hardware_specifications.get_id_by_name("XS")
    #     print(hw_id)
    #     deployment_details = self.client.deployments.create(TestWMLClientWithHybrid.model_id, meta_props={
    #         self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",
    #         self.client.deployments.ConfigurationMetaNames.ONLINE:{},
    #         self.client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {"id":hw_id}
    #          })
    #
    #     TestWMLClientWithHybrid.logger.debug("Deployment details: {}".format(deployment_details))
    #
    #     TestWMLClientWithHybrid.deployment_id = self.client.deployments.get_id(deployment_details)
    #     TestWMLClientWithHybrid.logger.debug("Deployment uid: {}".format(TestWMLClientWithHybrid.deployment_id))
    #
    #     TestWMLClientWithHybrid.scoring_url = self.client.deployments.get_scoring_href(deployment_details)
    #     TestWMLClientWithHybrid.logger.debug("Scoring url: {}".format(TestWMLClientWithHybrid.scoring_url))
    #
    #     self.assertTrue('online' in str(deployment_details))

    # def test_07_get_deployment_details(self):
    #     TestWMLClientWithHybrid.logger.info("Get deployment details")
    #     deployment_details = self.client.deployments.get_details(deployment_id=TestWMLClientWithHybrid.deployment_id)
    #     self.assertIsNotNone(deployment_details)
    #
    # def test_08_scoring(self):
    #     TestWMLClientWithHybrid.logger.info("Score the model")
    #     scoring_payload = {
    #         self.client.deployments.ScoringMetaNames.INPUT_DATA: [
    #             {
    #                 "fields": ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    #                            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    #                            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    #                            "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    #                            "SampleWeight"],
    #                 "values":[["3638-WEABW","Female",0,"Yes","No",58,"Yes","Yes","DSL","No","Yes","No","Yes","No","No","Two year","Yes","Credit card (automatic)",59.9,3505.1,"No",2.768]]
    #             }
    #         ]
    #     }
    #
    #     scores = self.client.deployments.score(TestWMLClientWithHybrid.deployment_id, scoring_payload)
    #     self.assertIsNotNone(scores)

    def test_09_create_revision(self):
        TestWMLClientWithHybrid.logger.info("Create Revision")
        self.client.repository.create_model_revision(TestWMLClientWithHybrid.model_id)
        self.client.repository.list_models_revisions(TestWMLClientWithHybrid.model_id)
        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "updated do model",
                            }
        published_model_updated = self.client.repository.update_model(TestWMLClientWithHybrid.model_id,
                                                                      model_meta_props ) #self.update_model_path)
        self.assertIsNotNone(TestWMLClientWithHybrid.model_id)
        self.assertTrue("updated do model" in str(published_model_updated))
        self.client.repository.create_model_revision(TestWMLClientWithHybrid.model_id)
        rev_details = self.client.repository.get_model_revision_details(TestWMLClientWithHybrid.model_id, 2)
        self.assertTrue("updated do model" in str(rev_details))

    # def test_07_create_deployment(self):
    #     TestWMLClientWithDO.logger.info("Create deployment")
    #
    #     deploy_meta = {
    #         self.client.deployments.ConfigurationMetaNames.NAME: "deployment_DO",
    #         self.client.deployments.ConfigurationMetaNames.DESCRIPTION: "DO deployment",
    #         self.client.deployments.ConfigurationMetaNames.BATCH: {},

    #         self.client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {"name": "S", "num_nodes": 1}
    #     }
    #
    #
    #     deployment_details = self.client.deployments.create(TestWMLClientWithDO.model_id,
    #                                                         meta_props=deploy_meta)
    #
    #     TestWMLClientWithDO.logger.debug("Deployment details: {}".format(deployment_details))
    #
    #     TestWMLClientWithDO.deployment_id = self.client.deployments.get_id(deployment_details)
    #     TestWMLClientWithDO.logger.debug("Deployment uid: {}".format(TestWMLClientWithDO.deployment_id))
    #
    #
    # def test_08_get_deployment_details(self):
    #     TestWMLClientWithDO.logger.info("Get deployment details")
    #     deployment_details = self.client.deployments.get_details(deployment_id=TestWMLClientWithDO.deployment_id)
    #     self.assertIsNotNone(deployment_details)
    #
    #
    # def test_09_create_job(self):
    #     TestWMLClientWithDO.logger.info("Create job details")
    #     TestWMLClientWithDO.logger.debug("Create job")
    #
    #     # initialize list of lists
    #     diet_food = pd.DataFrame([["Roasted Chicken", 0.84, 0, 10],
    #                               ["Spaghetti W/ Sauce", 0.78, 0, 10],
    #                               ["Tomato,Red,Ripe,Raw", 0.27, 0, 10],
    #                               ["Apple,Raw,W/Skin", 0.24, 0, 10],
    #                               ["Grapes", 0.32, 0, 10],
    #                               ["Chocolate Chip Cookies", 0.03, 0, 10],
    #                               ["Lowfat Milk", 0.23, 0, 10],
    #                               ["Raisin Brn", 0.34, 0, 10],
    #                               ["Hotdog", 0.31, 0, 10]], columns=["name", "unit_cost", "qmin", "qmax"])
    #
    #     diet_food_nutrients = pd.DataFrame([
    #         ["Spaghetti W/ Sauce", 358.2, 80.2, 2.3, 3055.2, 11.6, 58.3, 8.2],
    #         ["Roasted Chicken", 277.4, 21.9, 1.8, 77.4, 0, 0, 42.2],
    #         ["Tomato,Red,Ripe,Raw", 25.8, 6.2, 0.6, 766.3, 1.4, 5.7, 1],
    #         ["Apple,Raw,W/Skin", 81.4, 9.7, 0.2, 73.1, 3.7, 21, 0.3],
    #         ["Grapes", 15.1, 3.4, 0.1, 24, 0.2, 4.1, 0.2],
    #         ["Chocolate Chip Cookies", 78.1, 6.2, 0.4, 101.8, 0, 9.3, 0.9],
    #         ["Lowfat Milk", 121.2, 296.7, 0.1, 500.2, 0, 11.7, 8.1],
    #         ["Raisin Brn", 115.1, 12.9, 16.8, 1250.2, 4, 27.9, 4],
    #         ["Hotdog", 242.1, 23.5, 2.3, 0, 0, 18, 10.4]
    #     ], columns=["Food", "Calories", "Calcium", "Iron", "Vit_A", "Dietary_Fiber", "Carbohydrates", "Protein"])
    #
    #     diet_nutrients = pd.DataFrame([
    #         ["Calories", 2000, 2500],
    #         ["Calcium", 800, 1600],
    #         ["Iron", 10, 30],
    #         ["Vit_A", 5000, 50000],
    #         ["Dietary_Fiber", 25, 100],
    #         ["Carbohydrates", 0, 300],
    #         ["Protein", 50, 100]
    #     ], columns=["name", "qmin", "qmax"])
    #
    #     job_payload_ref = {
    #         self.client.deployments.DecisionOptimizationMetaNames.INPUT_DATA: [
    #             {
    #                 "id": "diet_food.csv",
    #                 "values": diet_food
    #             },
    #             {
    #                 "id": "diet_food_nutrients.csv",
    #                 "values": diet_food_nutrients
    #             },
    #             {
    #                 "id": "diet_nutrients.csv",
    #                 "values": diet_nutrients
    #             }
    #         ],
    #         self.client.deployments.DecisionOptimizationMetaNames.OUTPUT_DATA: [
    #             {
    #                 "id": ".*.csv"
    #             }
    #         ]
    #     }
    #
    #     TestWMLClientWithDO.job_details = self.client.deployments.create_job(TestWMLClientWithDO.deployment_id,
    #                                                                          meta_props=job_payload_ref)
    #     TestWMLClientWithDO.job_id = self.client.deployments.get_job_id(TestWMLClientWithDO.job_details)
    #
    #
    # def test_10_get_job_status(self):
    #     start_time = time.time()
    #     diff_time = start_time - start_time
    #     while True and diff_time < 10 * 60:
    #         time.sleep(3)
    #         response = self.client.deployments.get_job_status(TestWMLClientWithDO.job_id)
    #         if response['state'] == 'completed' or response['state'] == 'error' or response['state'] == 'failed':
    #             break
    #         diff_time = time.time() - start_time
    #
    #     self.assertIsNotNone(response)
    #     self.assertTrue(response['state'] == 'completed')
    #
    # def test_11_extract_and_display_solution(self):
    #     job_details_do = self.client.deployments.get_job_details(TestWMLClientWithDO.job_id)
    #     kpi = job_details_do['entity']['decision_optimization']['solve_state']['details']['KPI.Total Calories']
    #     print(kpi)
    #     TestWMLClientWithDO.logger.debug("The value of kpi is ", kpi)
    #     self.assertTrue(kpi == '2000.0')
    #
    #
    # def test_12_list_jobs(self):
    #     self.client.deployments.list_jobs()
    #
    # def test_12_delete_job(self):
    #     self.client.deployments.delete_job(TestWMLClientWithDO.job_id)
    #
    # def test_13_delete_deployment(self):
    #     TestWMLClientWithDO.logger.info("Delete deployment")
    #     self.client.deployments.delete(TestWMLClientWithDO.deployment_id)

    def test_10_download_model(self):
        TestWMLClientWithHybrid.logger.info("Download model")
        try:
            os.remove('download_test_url-hybrid')
        except OSError:
            pass

        try:
            file = open('download_test_url2', 'r')
        except IOError:
            file = open('download_test_url2', 'w')
            file.close()

        self.client.repository.download(TestWMLClientWithHybrid.model_id, filename='download_test_url-hybrid')
       # self.assertRaises(WMLClientError, self.client.repository.download, TestWMLClientWithDO.model_id, filename='download_test_id')


    def test_11_delete_model(self):
        TestWMLClientWithHybrid.logger.info("Delete function")
        self.client.script.delete(TestWMLClientWithHybrid.model_id)

    def test_12_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithHybrid.space_id)


if __name__ == '__main__':
    unittest.main()
