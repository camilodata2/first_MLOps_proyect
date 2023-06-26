#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import unittest


from ibm_watson_machine_learning.tests.base.abstract.abstract_online_deployment_test import AbstractOnlineDeploymentTest


class TestAIFunction(AbstractOnlineDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying a Python function
    which is using keras library using compressed scenario.
    """
    software_specification_name = "runtime-22.2-py3.10"
    model_name = deployment_name = "ai_function_from_func"
    file_name = "ai_func_3.py.gz"
    IS_MODEL = False

    def get_model(self):
        return os.path.join(os.getcwd(), 'base', 'artifacts', 'python_function', self.file_name)

    def create_model_props(self):
        return {
            self.wml_client.repository.FunctionMetaNames.NAME: self.model_name,
            self.wml_client.repository.FunctionMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA:
                [{
                     "fields": ["Customer_Service"],
                     "values": [
                         [
                             "service was good."
                         ],
                         [
                             "The woman at the counter was friendly and tried to accommodate me as best she could. "
                             "The counter was close to the terminal and the whole thing was quick and expedient. "
                         ],
                         [
                             "I do not  understand why I have to pay additional fee if vehicle is returned without a "
                             "full tank. "
                         ]
                     ]
                }]
        }


if __name__ == "__main__":
    unittest.main()
