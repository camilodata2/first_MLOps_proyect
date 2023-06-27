#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

from ibm_watson_machine_learning.tests.base.abstract.abstract_online_deployment_test import AbstractOnlineDeploymentTest


class TestTensorflowDeployment(AbstractOnlineDeploymentTest, unittest.TestCase):
    """
    Test case checking the scenario of storing & deploying Tensorflow model
    using object.
    """
    deployment_type = "tensorflow_2.9"
    software_specification_name = "runtime-22.2-py3.10"
    model_name = deployment_name = "tensorflow_model_from_object"
    IS_MODEL = True

    def get_model(self):
        batch_size = 128
        num_classes = 10
        epochs = 1
        num_train_samples = 500
        img_rows, img_cols = 28, 28

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        x_train = x_train[:num_train_samples]
        y_train = y_train[:num_train_samples]

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        X = Input(shape=(28, 28, 1))
        l1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(X)
        l2 = Conv2D(64, (3, 3), activation='relu')(l1)
        l21 = MaxPooling2D(pool_size=(2, 2))(l2)
        l22 = Dropout(0.25)(l21)
        l23 = Flatten()(l22)
        l3 = Dense(128, activation='relu')(l23)
        l31 = Dropout(0.5)(l3)
        y_hat = Dense(num_classes, activation='softmax', name="y_hat")(l31)
        model = Model(inputs=X, outputs=y_hat)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test))

        TestTensorflowDeployment.x_test = x_test

        return model

    def create_model_props(self):
        return {
            self.wml_client.repository.ModelMetaNames.NAME: self.model_name,
            self.wml_client.repository.ModelMetaNames.TYPE: self.deployment_type,
            self.wml_client.repository.ModelMetaNames.TF_MODEL_PARAMS: {"save_format": "h5"},
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.wml_client.software_specifications.get_id_by_name(self.software_specification_name)
        }

    def create_scoring_payload(self):
        x_score_1 = TestTensorflowDeployment.x_test[23].tolist()
        x_score_2 = TestTensorflowDeployment.x_test[32].tolist()
        return {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    'values': [x_score_1, x_score_2]
                }
            ]
        }


if __name__ == "__main__":
    unittest.main()
