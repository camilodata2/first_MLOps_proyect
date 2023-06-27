#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc

from ibm_watson_machine_learning.tests.base.abstract.abstract_deployment_test import AbstractDeploymentTest



class AbstractOnlineDeploymentTest(AbstractDeploymentTest, abc.ABC):
    """
    Abstract class implementing scoring with online deployment.
    """
    def create_deployment_props(self):
        return {
            self.wml_client.deployments.ConfigurationMetaNames.NAME: self.deployment_name,
            self.wml_client.deployments.ConfigurationMetaNames.ONLINE: {}
        }

    def test_09_download_deployment(self):
        pass

    def test_10_score_deployments(self):
        scoring_payload = self.create_scoring_payload()
        predictions = self.wml_client.deployments.score(self.deployment_id, scoring_payload)

        self.assertIsNotNone(predictions)
        self.assertIn("predictions", predictions)
