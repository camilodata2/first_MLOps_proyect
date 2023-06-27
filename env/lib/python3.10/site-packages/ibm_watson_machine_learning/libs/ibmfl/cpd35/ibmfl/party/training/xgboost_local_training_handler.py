#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import inspect
import logging
import operator
import itertools
import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from timeit import default_timer as time

from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble._hist_gradient_boosting.loss import _LOSSES
from sklearn.utils.multiclass import check_classification_targets
from sklearn.ensemble._hist_gradient_boosting.common import Y_DTYPE
from sklearn.utils import check_X_y, check_random_state, check_array
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper

from ibmfl.model.xgb_fl_model import XGBFLModel
from ibmfl.util.xgboost.utils import is_classifier
from ibmfl.util.xgboost.hyperparams import init_parameters

from ibmfl.exceptions import HyperparamsException
from ibmfl.exceptions import LocalTrainingException, ModelUpdateException
from ibmfl.party.training.local_training_handler import LocalTrainingHandler

logger = logging.getLogger(__name__)


class XGBoostBaseLocalTrainingHandler(LocalTrainingHandler, ABC):
    """
    Class implementation for XGBoost Base Local Training Handler
    """

    def update_model(self, model_update):
        """
        Update local model with model updates received from FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        :return: `None`
        """
        try:
            # Check to see if worker's core model object has been initialized.
            if self.fl_model is None:
                self.fl_model = XGBFLModel('XGBFLModel', self.hyperparams)
                logger.info(
                    'No model update was provided, initialized new FL model object.')

            # Update Model Object
            if model_update is not None:
                self.fl_model.update_model(model_update)
                logger.info('Local model updated.')
            else:
                logger.info('No model update was provided.')
        except Exception as ex:
            raise LocalTrainingException(
                'No query information is provided. '+ str(ex))

    def train(self, fit_params=None):
        """
        Primary wrapper function used for routing internal remote function calls
        within the Local Training Handler functions.

        :param fit_params: A dictionary payload structure containing two key \
        signatures, `func` and `args`, which respectively are the target \
        function defined within the Local Training Handler and the arguments \
        defined within the executing function, which is defined as a dictionary \
        containing key-value pairs of matching arguments.
        :type fit_params: `dict`
        :return: Returns the corresponding values depending on the function \
        remotely called from the aggregator.
        """
        result = None
        try:
            # Validate Incoming Payload Parameter
            if fit_params is None:
                raise LocalTrainingException('Provided fit_params is None, no '
                                             'functions were executed.')

            # Validate Payload Signature
            if 'func' not in fit_params or 'args' not in fit_params:
                raise LocalTrainingException('Malformed payload, must include '
                                             'func and args in payload.')

            # Validate Defined Function Header
            if not (isinstance(fit_params['func'], str) and
                    hasattr(self, fit_params['func'])):
                raise LocalTrainingHandler('Function header is not valid or is '
                                           'not defined within the scope of the '
                                           'local training handler.')

            # Validate Payload Argument Parameter Mappings Against Function
            spec = inspect.getargspec(eval('self.'+fit_params['func']))
            for k in fit_params['args'].keys():
                if k not in spec.args:
                    raise LocalTrainingHandler('Specified parameter argument is '
                                               'not defined in the function.')

            # Construct Function Call Command
            result = eval('self.'+fit_params['func'])(**fit_params['args'])
        except Exception as ex:
            raise LocalTrainingException('Error processing remote function ' +
                                         'call: ' + str(ex))

        return result

    def initialize(self, params):
        """
        A remote function call which performs the following procedures:
        1. Obtains global hyperparameters from the aggregator and initializes
        them for each of the local worker's parameters.
        2. Performs the following set of preliminary validation checks
        and processes:
           a. Perform dataset encoding and validation checks.
           b. Set local worker's seed and PRNG states.
           c. Initialize key parameters from defined hyperparameters in 1.
           d. Check for missing data within each local worker's dataset.
           e. Returns various parameters needed for aggregator to perform
           tree growth operation.

        :param params: A hyperparameter dictionary from the aggregator.
        :type params: `dict`
        :return: Dictionary containing parameters necessary for tree growth at \
        the aggregator.
        :rtype: `dict`
        """
        # Initialize Hyperparameters
        logger.info('Recieved and initializing local hyperparameters.')
        self.hyperparams = params

        init_parameters(self, params)

        logger.info(
            'Performing preliminary checks, validation, and initialization.')

        # Data Validation and Encoding Checks
        logger.info('[CHECK] Dataset Encoding and Validation Checks')
        self.data_val_enc()

        # Check PRNG State
        logger.info('[CHECK] PRNG State and Seed Validation')
        self.check_prng_state()

        # Initialize Key Parameters
        logger.info('[CHECK] Initialize Key Parameters')
        self.loss_ = self.get_loss(sample_weight=self.sample_weight)

        # Validate Dataset
        self.X_train, self.y_train = self.X, self.Y
        self.X_val, self.y_val = None, None

        # Check for Missing Data
        logger.info('[CHECK] Check for Missing Data')
        self.has_missing_values = np.isnan(
            self.X_train).any(axis=0).astype(np.uint8)

        # Prepare Payload
        params = {
            'has_missing_values': self.has_missing_values,
            'need_update_leaves_values': self.loss_.need_update_leaves_values,
            'n_features_': self.X.shape[1]
        }
        if is_classifier(self):
            params['classes_'] = self.classes_

        return params

    def check_prng_state(self):
        """
        Given the initialize `random_state` from the hyperparameter, we set the
        random seed value of the local training handler.

        :return: `None`
        """
        # Check PRNG Random State
        rng = check_random_state(self.random_state)
        self._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

    def get_sample_size(self):
        """
        Obtain the number of samples from the training set.

        :return: Returns the number of samples in each worker.
        :rtype: `int`
        """
        if hasattr(self, 'X_train') and self.X_train is not None:
            return len(self.X_train)
        else:
            return 0

    def set_eps(self, eps):
        """
        Set the eps defined by the aggregator, which also defines the `max_bins`
        size in the local training handler for the histogram.

        :param eps: Epsilon parameter which dictates the bin size of the \
        histogram.
        :type eps: `float`
        :return: `None`
        """
        if isinstance(eps, float) and eps > 0:
            self.eps = eps
            self.max_bins = int(1 // self.eps)

            logger.info('Setting local worker eps to ' + str(self.eps))
            logger.info('Setting local worker max_bin to ' +
                        str(self.max_bins))
        else:
            raise ValueError('Defined eps is not valid.')

    def bin_data(self):
        """
        Initialize process to bin the training data. The number of bins
        allocated here are based on the maximum number of bins set by the
        aggregator and 1 more to account for the missing values.

        :return: `None`
        """
        logger.info('Binning Dataset')
        self.n_bins = self.max_bins + 1  # +1 for Missing Data
        self.bin_mapper_ = _BinMapper(
            n_bins=self.n_bins, random_state=self._random_seed)

        self.X_binned_train = self.bin_mapper_.fit_transform(self.X_train)
        self.n_samples = self.X_binned_train.shape[0]

    def init_null_preds(self):
        """
        Initialize null predictions from baseline models. These values are used
        to construct the initial tree structure of the model. Note that
        raw_predictions has shape (n_trees_per_iteration, n_samples) whereas
        n_trees_per_iteration is n_classes in multiclass classification, else 1.
        Furthermore, setups initial data structures for metrics and performs
        early stop validation checks.

        :return: `None`
        """
        # Generate Baseline Predictions
        logger.info('Generating Initial Predictions from Null Model')
        self.baseline = self.loss_.get_baseline_prediction(
            self.y_train, self.sample_weight, self.n_trees)
        self.raw_predictions = np.zeros(shape=(self.n_trees, self.n_samples),
                                        dtype=self.baseline.dtype) + self.baseline

        # Initialize Relevant Datastructures for Metrics
        self._predictors = predictors = []
        self._scorer = None
        self.raw_predictions_val = None
        self.train_score_ = []
        self.validation_score_ = []

        self.begin_at_stage = 0

        # Initialize Model Attributes
        self.fl_model.n_trees = self.n_trees
        self.fl_model._baseline_prediction = self.baseline
        self.fl_model._raw_predictions = self.raw_predictions
        self.fl_model.loss_ = self.loss_
        self.fl_model.n_features_ = self.X.shape[1]
        if is_classifier(self):
            self.fl_model.classes_ = self.classes_

    def init_grad(self):
        """
        Process to initialize gradients and hessians on the Local Training
        Handler. This is an empty data structure used to contain and later
        will be used to send over to aggregator.
        Note: shape = (n_trees_per_iteration, n_samples)

        :return: `None`
        """
        self.gradients, self.hessians = self.loss_.init_gradients_and_hessians(
            n_samples=self.n_samples,
            sample_weight=self.sample_weight,
            prediction_dim=self.n_trees
        )

    def update_grad(self):
        """
        A wrapper function to update the corresponding gradient and hessian
        statistics inplace based on the raw predictions from the previous
        iteration of the model.

        :return: `None`
        """
        self.loss_.update_gradients_and_hessians(self.gradients, self.hessians,
                                                 self.y_train,
                                                 self.raw_predictions,
                                                 self.sample_weight)

    def collect_hist(self, k):
        """
        We return values of the surrogate histogram values, gradients, hessians,
        and the number of non-missing bin counts from the local worker.

        :param k: The correseponding feature index of the dataset.
        :type k: `int`
        :return: Returns a tuple comprising of the histogram bin index values, \
        gradient, hessian, missing value counts per feature, bin threshold value.
        :rtype: (`np.array`, `np.array`, `np.array`, `list`, `list`)
        """
        return self.X_binned_train, self.gradients[k, :], self.hessians[k, :], \
            self.bin_mapper_.n_bins_non_missing_, self.bin_mapper_.bin_thresholds_

    def update_raw_preds(self, k, predictor, use_val):
        """
        Function which updates internal local trainer handler's raw_prediction
        values given the intermediate state of the model.

        NOTE: Because we merged the histogram, the histogram mappings no longer
        apply in this context, so all predictions must be done entirely from
        scratch again - cannot use the histogram subtraction technique from
        previous implementation here.

        :param k: The correseponding feature index of the dataset.
        :type k: `int`
        :param predictor: The predictor object state of the XGBoost tree.
        :type  predictor: `TreePredictor`
        :param use_val: Boolean indicator whether to use the validation set. \
        (Currently this parameter is not supported at the moment.)
        :type use_val: `Boolean`
        :return: `None`
        """
        self.raw_predictions[k] += predictor.predict(self.X_train)

    def data_val_enc(self):
        """
        Performs a validation procedure of the input dataset to correspondingly
        check the data types as well as the data encoding.

        :return: `None`
        """
        logging.info('Performing local worker data encoding and validation.')
        (x, y), (_) = self.data_handler.get_data()

        # Perform Data Encoding Validation
        self.X, y = check_X_y(x, y, dtype=[np.float64], force_all_finite=False)
        self.Y = self.encode_target(y)

        # TODO load sample_weight from config
        self.sample_weight = None

    @abstractmethod
    def encode_target(self, y=None):
        raise NotImplementedError


class XGBRegressorLocalTrainingHandler(XGBoostBaseLocalTrainingHandler):
    _VALID_LOSSES = ('least_squares')

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


class XGBClassifierLocalTrainingHandler(XGBoostBaseLocalTrainingHandler):
    _VALID_LOSSES = ('binary_crossentropy', 'categorical_crossentropy', 'auto')

    def encode_target(self, y):
        """
        Converts the input y to the expected dtype and performs a label
        encoding. Here, we assume that each party has at least one sample of
        the corresponding class label type for each different classes.

        :param y: The corresponding target data from the dataset to encode.
        :type y: `np.array`
        :return: Returns the corresponding encoded y values.
        :rtype: `np.array`
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
            return _LOSSES['binary_crossentropy'](sample_weight=sample_weight) \
                if self.n_trees == 1 else \
                _LOSSES['categorical_crossentropy'](sample_weight=sample_weight)
        else:
            return _LOSSES[self.loss](sample_weight=sample_weight)
