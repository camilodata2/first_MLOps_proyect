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
    using function definition.
    """
    software_specification_name = "runtime-22.2-py3.10"
    model_name = deployment_name = "ai_function_from_func"
    IS_MODEL = False

    def get_model(self):
        def score(payload):
            payload = payload['input_data'][0]
            values = [[row[0] * row[1]] for row in payload['values']]
            return {'predictions': [{'fields': ['multiplication'], 'values': values}]}

        return score

    def create_model_props(self):
        return {
            self.wml_client.repository.FunctionMetaNames.NAME: self.model_name,
            self.wml_client.repository.FunctionMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "fields": ["multiplication"],
                "values": [[2.0, 2.0], [99.0, 99.0]]
            }]
        }


if __name__ == "__main__":
    unittest.main()
