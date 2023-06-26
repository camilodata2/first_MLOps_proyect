#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import unittest

from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

from ibm_watson_machine_learning.tests.base.abstract.abstract_online_deployment_test import AbstractOnlineDeploymentTest


class TestTensorflowDeployment(AbstractOnlineDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying Tensorflow model
    using compressed file.
    """
    deployment_type = "tensorflow_2.9"
    software_specification_name = "runtime-22.2-py3.10"
    model_name = deployment_name = "tensorflow_model_from_h5zip"
    file_name = "keras_model.h5.zip"
    IS_MODEL = True

    def get_model(self):
        return os.path.join(os.getcwd(), 'base', 'artifacts', 'tensorflow', self.file_name)

    def create_model_props(self):
        return {
            self.wml_client.repository.ModelMetaNames.NAME: self.model_name,
            self.wml_client.repository.ModelMetaNames.TYPE: self.deployment_type,
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if K.image_data_format() == 'channels_first':
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255

        x_score_1 = x_test[23].tolist()
        x_score_2 = x_test[32].tolist()
        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                'values': [x_score_1, x_score_2]
            }]
        }


if __name__ == "__main__":
    unittest.main()
