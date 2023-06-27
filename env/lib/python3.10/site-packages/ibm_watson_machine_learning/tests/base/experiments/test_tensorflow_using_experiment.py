#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import unittest

from ibm_watson_machine_learning.tests.base.abstract.abstract_deep_learning_test import \
    AbstractDeepLearningExperimentTest


class TestTensorflowTraining(AbstractDeepLearningExperimentTest, unittest.TestCase):
    """
    Test case checking the scenario of training an tensorflow model
    using stored experiment.
    """

    model_definition_name = "tensorflow_model_definition"
    experiment_name = "tensorflow_experiment"
    training_name = "test_tensorflow_training"
    training_description = "TF-test-experiment"
    software_specification_name = "tensorflow_rt22.2-py3.10"
    execution_command = "python3 mnist_mlp.py"

    data_location = os.path.join(os.getcwd(), "base", "datasets", "tensorflow", "mnist.npz")
    data_cos_path = "mnist.npz"
    model_paths = [
        os.path.join(os.getcwd(), "base", "artifacts", "tensorflow", "tf-model-definition.zip")
    ]
    model_definition_ids = []
