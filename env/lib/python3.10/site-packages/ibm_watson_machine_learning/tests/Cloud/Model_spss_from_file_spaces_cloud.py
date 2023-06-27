#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.tests.Cloud.preparation_and_cleaning import *


class TestWMLClientWithSPSS(unittest.TestCase):
    deployment_id = None
    model_id = None
    scoring_url = None
    scoring_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSPSS.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.cos_credentials = get_cos_credentials()
        self.cos_resource_crn = self.cos_credentials['resource_instance_id']
        self.instance_crn = get_instance_crn()

        self.space_name = str(uuid.uuid4())

        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: 'space' + self.space_name,
            self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description',
            self.client.spaces.ConfigurationMetaNames.STORAGE: {
                "type": "bmcos_object_storage",
                "resource_crn": self.cos_resource_crn
            },
            self.client.spaces.ConfigurationMetaNames.COMPUTE: {
                "name": "test_tf",
                "crn": self.instance_crn
            }
        }

        self.space = self.client.spaces.store(meta_props=metadata, background_mode=False)

        TestWMLClientWithSPSS.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientWithSPSS.space_id)
        self.client.set.default_space(TestWMLClientWithSPSS.space_id)
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'customer-satisfaction-prediction.str')

    # def test_01_set_space(self):
    #     space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case_SPSS"})
    #
    #     TestWMLClientWithSPSS.space_id = self.client.spaces.get_id(space)
    #     self.client.set.default_space(TestWMLClientWithSPSS.space_id)
    #     self.assertTrue("SUCCESS" in self.client.set.default_space(TestWMLClientWithSPSS.space_id))


    def test_01_publish_local_model_in_repository(self):
        TestWMLClientWithSPSS.logger.info("Saving trained model in repo ...")
        TestWMLClientWithSPSS.logger.debug("Model path: {}".format(self.model_path))

        self.client.repository.ModelMetaNames.show()

        sw_spec_id = self.client.software_specifications.get_id_by_name("spss-modeler_18.1")

        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "LOCALLY created agaricus prediction model",
                       self.client.repository.ModelMetaNames.TYPE: "spss-modeler_18.1",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                            }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        print(published_model)
        TestWMLClientWithSPSS.model_id = self.client.repository.get_model_id(published_model)
        TestWMLClientWithSPSS.logger.info("Published model ID:" + str(TestWMLClientWithSPSS.model_id))
        self.assertIsNotNone(TestWMLClientWithSPSS.model_id)

    def test_03_load_model(self):
        TestWMLClientWithSPSS.logger.info("Load model from repository: {}".format(TestWMLClientWithSPSS.model_id))
        self.tf_model = self.client.repository.load(TestWMLClientWithSPSS.model_id)
        TestWMLClientWithSPSS.logger.debug("SPSS type: {}".format(type(self.tf_model)))
        self.assertTrue(self.tf_model)

    # def test_04_get_details(self):
    #     TestWMLClientWithSPSS.logger.info("Get details")
    #     self.assertIsNotNone(self.client.repository.get_details(TestWMLClientWithSPSS.model_id))

    def test_05_get_model_details(self):
        TestWMLClientWithSPSS.logger.info("Get model details")
        self.assertIsNotNone(self.client.repository.get_model_details(TestWMLClientWithSPSS.model_id))

    def test_07_create_deployment(self):
        TestWMLClientWithSPSS.logger.info("Create deployment")
        deployment_details = self.client.deployments.create(TestWMLClientWithSPSS.model_id, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithSPSS.logger.debug("Deployment details: {}".format(deployment_details))

        TestWMLClientWithSPSS.deployment_id = self.client.deployments.get_id(deployment_details)
        TestWMLClientWithSPSS.logger.debug("Deployment uid: {}".format(TestWMLClientWithSPSS.deployment_id))

        TestWMLClientWithSPSS.scoring_url = self.client.deployments.get_scoring_href(deployment_details)
        TestWMLClientWithSPSS.logger.debug("Scoring url: {}".format(TestWMLClientWithSPSS.scoring_url))

        self.assertTrue('online' in str(deployment_details))

    def test_08_get_deployment_details(self):
        TestWMLClientWithSPSS.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details(deployment_uid=TestWMLClientWithSPSS.deployment_id)
        self.assertIsNotNone(deployment_details)

    def test_09_scoring(self):
        TestWMLClientWithSPSS.logger.info("Score the model")
        scoring_payload = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    "fields": ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                               "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                               "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
                               "SampleWeight"],
                    "values":[["3638-WEABW","Female",0,"Yes","No",58,"Yes","Yes","DSL","No","Yes","No","Yes","No","No","Two year","Yes","Credit card (automatic)",59.9,3505.1,"No",2.768]]
                }
            ]
        }

        scores = self.client.deployments.score(TestWMLClientWithSPSS.deployment_id, scoring_payload)
        self.assertIsNotNone(scores)

    def test_10_create_revision(self):
        TestWMLClientWithSPSS.logger.info("Create Revision")
        self.client.repository.create_model_revision(TestWMLClientWithSPSS.model_id)
        self.client.repository.list_models_revisions(TestWMLClientWithSPSS.model_id)
        model_meta_props = { self.client.repository.ModelMetaNames.NAME: "updated prediction model",
                            }
        published_model_updated = self.client.repository.update_model(TestWMLClientWithSPSS.model_id, model_meta_props,self.model_path)
        self.assertIsNotNone(TestWMLClientWithSPSS.model_id)
        self.assertTrue("updated prediction model" in str(published_model_updated))

    def test_10_delete_deployment(self):
        TestWMLClientWithSPSS.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithSPSS.deployment_id)

    def test_11_delete_model(self):
        TestWMLClientWithSPSS.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithSPSS.model_id)
        self.client.spaces.delete(TestWMLClientWithSPSS.space_id)


if __name__ == '__main__':
    unittest.main()
