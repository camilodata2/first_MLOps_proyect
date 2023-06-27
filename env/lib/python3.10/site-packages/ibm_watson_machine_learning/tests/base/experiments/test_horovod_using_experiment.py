#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import unittest

from ibm_watson_machine_learning.tests.base.abstract.abstract_deep_learning_test import \
    AbstractDeepLearningExperimentTest


class TestHorovodTraining(AbstractDeepLearningExperimentTest, unittest.TestCase):
    """
    Test case checking the scenario of training an horovod model
    using model_definition only.
    """

    model_definition_name = "horovod_model_definition"
    training_name = "test_horovod_training"
    experiment_name = "horovod_test_experiment"
    training_description = "training - Horovod"
    software_specification_name = "tensorflow_rt22.2-py3.10"
    execution_command = "python tensorflow_mnist.py"

    data_location = os.path.join(os.getcwd(), "base", "datasets", "tensorflow", "mnist.npz")
    data_cos_path = "mnist.npz"
    model_paths = [
        os.path.join(os.getcwd(), "base", "artifacts", "horovod", "tf_horovod.zip")
    ]
    model_definition_ids = []
