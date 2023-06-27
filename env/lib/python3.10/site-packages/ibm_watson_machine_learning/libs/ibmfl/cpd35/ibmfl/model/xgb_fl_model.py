#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import time
import pickle
import logging

from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._hist_gradient_boosting.loss import *
from sklearn.ensemble._hist_gradient_boosting.common import *
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics import check_scoring, accuracy_score, r2_score
from sklearn.utils import check_X_y, check_random_state, check_array
from sklearn.ensemble._hist_gradient_boosting.grower import TreePredictor

from ibmfl.util import fl_metrics
from ibmfl.util import config
from ibmfl.model.fl_model import FLModel
from ibmfl.util.xgboost.utils import is_classifier
from ibmfl.util.xgboost.export import export_sklearn
from ibmfl.exceptions import ModelInitializationException, ModelException
from ibmfl.util.xgboost.hyperparams import init_parameters, \
    validate_parameters

logger = logging.getLogger(__name__)


class XGBFLModel(FLModel, ABC):
    """
    Wrapper class implementation for XGBoost containing the XGBoost Model Object
    """

    def __init__(self, model_name, model_spec, xgb_model=None, **kwargs):
        """
        Create a XGBFLModel instance for XGBoost model based either on an
        existing model object or an entirely new model object.

        :param model_type: String specifying the name of the model
        :type model_type: `str`
        :param model_spec: Hyperparameters associated with the model.
        :type model_spec: `dict`
        :param xgb_model: List of predictors existing predictor structures.
        :type xgb_model: `list`
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a XGBoost model.
        :type kwargs: `dict`
        """
        super().__init__(model_name, model_spec, **kwargs)

        # Initialize Input Model Object Parameters
        # Note: For model loading process, onboard from `xgb_model` parameter.
        self._predictors = []

        # Initialize Additional Internal Model Parameters
        self.model_type = 'XGBFLModel'
        self._baseline_prediction = None
        self._raw_predictions = None
        self.n_features_ = 0
        self.loss_ = None

        # Validate and Initialize Model Hyperparameters
        validate_parameters(model_spec)
        init_parameters(self, model_spec)

    def fit_model(self, train_data, fit_params=None, **kwargs):
        """
        This function is not implemented as model training is not operated
        within the FL Model object and is part of an external process.

        :return: `NotImplementedError`
        """
        return NotImplementedError

    def update_model(self, model_update=None, **kwargs):
        """
        Updates model using provided `model_update`. Additional arguments
        specific to the model can be added through `**kwargs`

        :param model_update: Model with update. This is specific to each model \
        type e.g., `ModelUpdateSGD`. The specific type should be checked by \
        the corresponding FLModel class.
        :type `ModelUpdate`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: None
        """
        if model_update is not None:
            if isinstance(model_update.get('xgb_model'), list):
                # Perform Model Update
                self._predictors.append(model_update.get('xgb_model'))
            else:
                raise ValueError('Provided model object is not of the correct '
                                 'data type. Should be a `list`. '
                                 'Type provided: ' + str(type(model_update)))

    def _raw_predict(self, x):
        """
        Internal helper function for generating predictions for the model.

        :param x: The input data for the prediction model.
        :type x: `np.array`
        :param model: Model object which is used for generating the prediction. \
        If not provided the default model assigned internally will be used.
        :type model: `XGBFLModel`
        :return: raw_predictions
        :rtype: `np.array`
        """
        logger.info('Performing Model Inference Process')

        # Initialize Prediction Object
        n_samples = x.shape[0]
        preds = np.zeros(shape=(self.n_trees, n_samples))

        # Append Null Model Predictions
        if self._baseline_prediction is not None:
            preds += self._baseline_prediction
        elif self._raw_predictions is not None:
            preds += self._raw_predictions

        # Generate Predictions
        for p in self._predictors:
            for k, p_i in enumerate(p):
                preds[k, :] += p_i.predict(x.astype(np.float64))

        return preds

    def get_model_update(self):
        """
        Since we are not using the conventional train() function at the local
        training handler, we respectively do not implement anything within here,
        nor this function is utilized anywhere else.
        """
        return NotImplementedError

    @abstractmethod
    def predict(self, x):
        """
        Given a set of inputs, generate inference for the given set of samples.

        :param x: The input data for the prediction model.
        :type x: `np.array`
        :param model: Model object which is used for generating the prediction. \
        If not provided the default model assigned internally will be used.
        :type model: `XGBFLModel`
        :return: `np.array`
        """
        return NotImplementedError

    def save_model(self, filename=None, path=None):
        """
        Saves a sklearn model to file in the format specific
        to the framework requirement as a pickle file (only for inference use).

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path \
        is specified, the model will be stored in the default data location of \
        the library `DATA_PATH`.
        :type path: `str`
        :return: filename
        """
        if filename is None:
            file = self.model_name if self.model_name else self.model_type
            filename = '{}_{}.pickle'.format(file, time.time())            
            
        full_path = super().get_model_absolute_path(filename)

        # Scikit-Learn Export Demo
        export_sklearn(self, full_path)

        if len(self._predictors) > 0:
            logger.info('Model saved in path: %s.', full_path)
        else:
            logger.info('Persisting empty model in path: %s.', full_path)

        return filename

    @staticmethod
    def load_model(filename):
        """
        Load model from provided filename

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path \
        is specified, the model will be stored in the default data location of \
        the library `DATA_PATH`.
        :type path: `str`
        :return: Returns the corresponding model object.
        :rtype: `XGBFLModel`
        """
        absolute_path = config.get_absolute_path(filename)

        with open(absolute_path, 'rb') as f:
            model = pickle.load(f)
            self = model

        if len(self._predictors) == 0:
            logger.info('Model does not contain any predictors.')

        return model

    @abstractmethod
    def get_loss(self):
        """
        Internal helper function used to obtain the corresponding loss function
        which is dependent on the learning task set by the hyperparameters.

        :return: Returns the loss object used to compute the loss function.
        :rtype: `BaseLoss` based object
        """
        return NotImplementedError

    def evaluate(self, test_dataset, **kwargs):
        """
        Evaluates the model given testing data.
        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, test) or a datagenerator of of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray`
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        """

        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            return self.evaluate_model(x_test, y_test)

        else:
            raise ModelException("Invalid test dataset!")

    @abstractmethod
    def evaluate_model(self, x, y, **kwargs):
        """
        Evaluates model given the samples x and true labels y.
        Multiple evaluation metrics are returned in a dictionary

        :param x: Samples with shape as expected by the model.
        :type x: Data structure as expected by the model \
        :param y: Corresponding labels to x
        :type y: Data structure the same as the type defines labels \
        in testing data.
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: Dictionary with all evaluation metrics provided by specific \
        implementation.
        :rtype: `dict`
        """
        return NotImplementedError


class XGBRegressorFLModel(XGBFLModel):
    _VALID_LOSSES = ('least_squares', 'least_absolute_deviation')

    def predict(self, x):
        """
        Perform prediction for a batch of inputs.

        :param x: Samples with shape as expected by the model.
        :type x: Data structure as expected by the model
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: Predictions
        :rtype: Data structure the same as the type defines labels
        in testing data.
        """
        if len(self._predictors) > 0:
            return self._raw_predict(x).ravel()
        else:
            raise Exception('Model has not been trained yet.')

    def encode_target(self, y):
        """
        Converts the input y to the expected dtype.

        :param y: The corresponding target data from the dataset to encode.
        :type y: `np.array`
        :return: Returns the corresponding encoded y values.
        :rtype: `np.array`
        """
        self.n_trees = 1
        return y.astype(Y_DTYPE, copy=False)

    def get_loss(self, sample_weight):
        """
        Given the initialized loss type defined under the hyerparameters, we
        return the corresponding loss function to dictate the corresponding
        learning task of the model.

        :param sample_weight: Weights of training data
        :type sample_weight: `np.ndarray`
        :return: Returns the respective loss object as defined in the FL \
        hyperparameters.
        :rtype: Derivation of `BaseLoss`
        """
        return _LOSSES[self.loss](sample_weight=sample_weight)

    def evaluate_model(self, x, y, **kwargs):
        """
        Given an input set of values and their values, generate the prediction
        and compute the R^2 metric.

        :param x: The input data for the prediction model.
        :type x: `np.array`
        :param y: The corresponding target value of the prediction.
        :type y: `np.array`
        :return: Return a dictionary containing the R^2 metric for the provided
        input and target values.
        :rtype: `dict`
        """
        metrics = {}
        if len(self._predictors) > 0:
            y_hat = self.predict(x)
            score = r2_score(y, y_hat)
            metrics['r2_score'] = score
            additional_metrics = fl_metrics.get_eval_metrics_for_regression(
                y, y_hat)
            metrics = {**metrics, **additional_metrics}
            return metrics
        else:
            logger.info('Model has not been trained yet.')
            return {}


class XGBClassifierFLModel(XGBFLModel):
    _VALID_LOSSES = ('binary_crossentropy', 'categorical_crossentropy', 'auto')

    def encode_target(self, y):
        """
        Converts the input y to the expected dtype and performs a label
        encoding. Here, we assume that each party has at least one sample of
        the corresponding class label type for each different classes.

        :param y: The corresponding target data from the dataset to encode.
        :type  y: `np.array`
        :return y: Returns the corresponding encoded y values.
        :rtype  y: `np.array`
        """
        # Validate Classification Target Values
        check_classification_targets(y)

        # Apply Label Encoder Transformation
        lab_enc = LabelEncoder()
        enc_y = lab_enc.fit_transform(y).astype(np.float64, copy=False)

        # Extract Encoded Target Sizes
        self.classes_ = lab_enc.classes_
        if self.classes_.shape[0] != self.num_classes:
            raise ValueError('Number of classes defined in configuration file '
                             'and the classes derived from the data does not '
                             'match. Found {0} classes, while config file '
                             'is defined as {1} classes.'.format(
                             self.classes_.shape[0], self.num_classes))

        if self.loss == 'auto':
            self.n_trees = 1 if self.classes_.shape[0] <= 2 else self.classes_.shape[0]
        else:
            self.n_trees = 1 if self.num_classes <= 2 else self.num_classes

        return enc_y

    def get_loss(self, sample_weight):
        """
        Given the initialized loss type defined under the hyerparameters, we
        return the corresponding loss function to dictate the corresponding
        learning task of the model. If auto is selected, then we will
        automatically determine whether the classification task is binary or
        multiclass given the label encoding cardinality.

        :param sample_weight: Weights of training data
        :type sample_weight: `np.ndarray`
        :return: Returns the respective loss object as defined in the FL \
        hyperparameters.
        :rtype: Derivation of `BaseLoss`
        """
        if (self.loss == 'categorical_crossentropy' and self.n_trees == 1):
            raise ValueError("Incompatible loss and target variable counts.")

        if self.loss == 'auto':
            return _LOSSES['binary_crossentropy']() if self.n_trees == 1 else \
                _LOSSES['categorical_crossentropy']()
        else:
            return _LOSSES[self.loss](sample_weight=sample_weight)

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs returned as class values.

        :param x: Samples with shape as expected by the model.
        :type x: Data structure as expected by the model
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: Predictions based on class values.
        :rtype: Data structure the same as the type defines labels
        in testing data.
        """
        if len(self._predictors) > 0:
            encoded_classes = np.argmax(self.predict_proba(x), axis=1)
            return self.classes_[encoded_classes]
        else:
            raise Exception('Model has not been trained yet.')

    def predict_proba(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs as probabilities.

        :param x: Samples with shape as expected by the model.
        :type x: Data structure as expected by the model
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: A probabilistic set of predictions
        :rtype: Data structure the same as the type defines labels \
        in testing data.
        """
        if len(self._predictors) > 0:
            raw_predictions = self._raw_predict(x)
            return self.loss_.predict_proba(raw_predictions)
        else:
            raise Exception('Model has not been trained yet.')

    def evaluate_model(self, x, y, **kwargs):
        """
        Given an input set of values and their values, generate the prediction
        and compute the accuracy.

        :param x: The input data for the prediction model.
        :type x: `np.array`
        :param y: The corresponding target value of the prediction.
        :type y: `np.array`
        :return: Return a dictionary containing the accuracy for the provided \
        input and target values.
        :rtype: `dict`
        """
        acc = {}
        if len(self._predictors) > 0:
            y_hat = self.predict(x)
            correct = 0
            for i in range(x.shape[0]):
                if y_hat[i] == y[i]:
                    correct += 1

            acc = {'acc': correct/float(len(y))}
            additional_metrics = fl_metrics.get_eval_metrics_for_classificaton(
                y, y_hat)

            acc = {**acc, **additional_metrics}
            return acc
        else:
            logger.info('No models have been trained yet.')
            return {}
