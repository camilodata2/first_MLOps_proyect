#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import unittest

import numpy as np

from ibm_watson_machine_learning.tests.base.abstract.abstract_online_deployment_test import AbstractOnlineDeploymentTest


class TestPyTorchDeployment(AbstractOnlineDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying PyTorch model
    using compressed file.
    """
    deployment_type = "pytorch-onnx_rt22.2"
    software_specification_name = "runtime-22.2-py3.10"
    model_name = deployment_name = "pytorch_model_from_gz"
    file_name = "mnist_pytorch.tar.gz"
    IS_MODEL = True

    def get_model(self):
        return os.path.join(os.getcwd(), 'base', 'artifacts', 'pytorch', self.file_name)

    def create_model_props(self):
        return {
            self.wml_client.repository.ModelMetaNames.NAME: self.model_name,
            self.wml_client.repository.ModelMetaNames.TYPE: self.deployment_type,
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        dataset = np.load(os.path.join(os.getcwd(), 'base', 'datasets', 'pytorch', 'mnist.npz'))
        X = dataset['x_test']

        score_0 = [X[0].tolist()]
        score_1 = [X[1].tolist()]

        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                'values': [score_0, score_1]
            }]
        }


if __name__ == "__main__":
    unittest.main()
