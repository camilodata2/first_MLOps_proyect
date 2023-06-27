#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import unittest

from ibm_watson_machine_learning.tests.base.abstract.abstract_online_deployment_test import AbstractOnlineDeploymentTest


class TestSPSSDeployment(AbstractOnlineDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying SPSS model
    using `str` file.
    """
    deployment_type = "spss-modeler_18.2"
    software_specification_name = "spss-modeler_18.2"
    model_name = deployment_name = "spss_model_from_file"
    file_name = "customer-satisfaction-prediction.str"
    IS_MODEL = True

    def get_model(self):
        return os.path.join(os.getcwd(), 'base', 'artifacts', 'spss', self.file_name)

    def create_model_props(self):
        return {
            self.wml_client.repository.ModelMetaNames.NAME: self.model_name,
            self.wml_client.repository.ModelMetaNames.TYPE: self.deployment_type,
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "fields": ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                           "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                           "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
                           "SampleWeight"],
                "values": [
                    ["3638-WEABW", "Female", 0, "Yes", "No", 58, "Yes", "Yes", "DSL", "No", "Yes", "No", "Yes", "No",
                     "No", "Two year", "Yes", "Credit card (automatic)", 59.9, 3505.1, "No", 2.768]]
            }]
        }


if __name__ == "__main__":
    unittest.main()
