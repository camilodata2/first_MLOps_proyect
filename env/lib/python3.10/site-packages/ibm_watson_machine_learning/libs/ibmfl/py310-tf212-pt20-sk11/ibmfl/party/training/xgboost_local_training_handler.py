#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import print_function

import inspect
import logging
import sys
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import scipy.stats.mstats as mstats
from ddsketch.ddsketch import DDSketch
from numcompress import compress
from scipy import stats
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE, X_DTYPE, Y_DTYPE
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.multiclass import check_classification_targets

from ibmfl.exceptions import LocalTrainingException
from ibmfl.message.message_type import MessageType
from ibmfl.model.xgb_fl_model import XGBFLModel
from ibmfl.party.training.local_training_handler import LocalTrainingHandler
from ibmfl.util.xgboost.hyperparams import init_parameters
from ibmfl.util.xgboost.utils import _LOSSES, is_classifier

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
        :return: `bool`
        """
        status = False
        try:
            # Check to see if worker's core model object has been initialized.
            if self.fl_model is None:
                self.fl_model = XGBFLModel("XGBFLModel", self.hyperparams)
                logger.info("update_model: Initialized a new FL model object [party_id={}]".format(self.party_id))
            # Update Model Object
            if model_update is not None:
                status = self.fl_model.update_model(model_update)
                logger.info("update_model: Local model updated [party_id={}]".format(self.party_id))
            else:
                logger.info("update_model: No model update was provided [party_id={}]".format(self.party_id))
        except Exception as ex:
            raise LocalTrainingException("update_model: [party_id={}] [ex={}]".format(self.party_id, ex))
        return status

    def sync_model_impl(self, payload=None):
        status = False
        model_update = payload["model_update"]
        status = self.update_model(model_update)
        return status

    def sync_model(self, payload=None):
        if not hasattr(self, "party_id"):
            logger.info("sync_model: Called before initialization")
            return
        logger.info("sync_model: Begin [party_id={}]".format(self.party_id))
        return super().sync_model(payload)

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
        logger.info(
            "train: Begin [func={}]".format(
                fit_params["func"] if isinstance(fit_params, dict) and "func" in fit_params else None
            )
        )
        logger.debug("train: [fit_params={}]".format(fit_params))

        result = None
        try:
            # Validate Incoming Payload Parameter
            if fit_params is None:
                raise LocalTrainingException("Provided fit_params is None, no " "functions were executed.")

            # Validate Payload Signature
            if "func" not in fit_params or "args" not in fit_params:
                raise LocalTrainingException("Malformed payload, must include " "func and args in payload.")

            # Validate Defined Function Header
            if not (isinstance(fit_params["func"], str) and hasattr(self, fit_params["func"])):
                raise LocalTrainingException(
                    "Function header is not valid or is "
                    "not defined within the scope of the "
                    "local training handler."
                )

            # Validate Payload Argument Parameter Mappings Against Function
            spec = inspect.getargspec(eval("self." + fit_params["func"]))
            for k in fit_params["args"].keys():
                if k not in spec.args:
                    raise LocalTrainingException("Specified parameter argument is " "not defined in the function.")

            # Construct Function Call Command
            result = eval("self." + fit_params["func"])(**fit_params["args"])
        except Exception as ex:
            raise LocalTrainingException("Error processing remote function " + "call: " + str(ex))

        logger.info("train: End")
        return result

    def initialize(self, params, party_id=None, last_step="update_raw_preds"):
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
        :param last_step: The last function call in a training round. Default to support pre-get_metrics  
        :type last_step: `string`
        :rtype: `dict`
        """
        # Initialize Hyperparameters
        logger.info("initialize: Received and initializing local hyper-parameters [party_id={}]".format(party_id))
        self.party_id = party_id
        self.hyperparams = params
        self.last_step = last_step

        logger.info("initialize: Performing preliminary checks and initialization.")
        init_parameters(self, params)

        # Data Validation and Encoding Checks
        logger.info("initialize: [CHECK] Dataset Encoding and Validation Checks")
        self.__data_val_enc()

        # Check PRNG State
        logger.info("initialize: [CHECK] PRNG State and Seed Validation")
        self.__check_prng_state()

        # Initialize Key Parameters
        logger.info("initialize: [CHECK] Initialize Key Parameters")
        self.loss_ = self.get_loss(sample_weight=self.sample_weight)

        # Validate Dataset
        self.X_train, self.y_train = self.X, self.Y
        self.X_val, self.y_val = None, None

        # Check for Missing Data
        logger.info("initialize: [CHECK] Check for Missing Data")
        self.has_missing_values = np.isnan(self.X_train).any(axis=0).astype(np.uint8)

        # Prepare Payload
        params = {
            "has_missing_values": self.has_missing_values,
            "need_update_leaves_values": self.loss_.need_update_leaves_values,
            "n_features_": self.X.shape[1],
        }
        if is_classifier(self):
            params["classes_"] = self.classes_

        return params

    def init_rejoin(self, party_id, params, curr_round, baseline, model_update, last_step="update_raw_preds"):
        """
        Initializes the current party when it rejoins the quorum after dropping.

        :return: `Bool`
        """

        logger.info("init_rejoin: Begin [party_id={}] [curr_round={}]".format(party_id, curr_round))

        self.n_completed_trains = curr_round - 1
        self.n_completed_trains_at_rejoin = curr_round

        try:
            logger.info("init_rejoin: Calling initialize [party_id={}]".format(party_id))
            self.initialize(params=params, party_id=party_id, last_step=last_step)
            logger.info("init_rejoin: Calling generate_sketch [party_id={}]".format(party_id))
            self.generate_sketch()
            logger.info("init_rejoin: Calling init_null_preds [party_id={}]".format(party_id))
            self.init_null_preds(baseline)
            logger.info("init_rejoin: Calling init_grad [party_id={}]".format(party_id))
            self.init_grad()
            if self.metrics_recorder:
                self.metrics_recorder.add_entry()
                self.metrics_recorder.set_round_no(self.get_n_completed_trains())
            logger.info("init_rejoin: Calling update_grad [party_id={}]".format(party_id))
            self.update_grad()
            logger.info("init_rejoin: Calling sync_model [party_id={}]".format(party_id))
            self.update_model(model_update)
            self.n_completed_trains += 1
        except Exception as ex:
            if hasattr(self, "party_id"):
                delattr(self, "party_id")
            logger.exception(
                "init_rejoin: [party_id={}] encountered an exception and must restart to rejoin [ex={}]".format(
                    party_id, ex
                )
            )
            return (party_id, False)

        logger.info("init_rejoin: End [party_id={}]".format(party_id))

        return (party_id, True)

    def __check_prng_state(self):
        """
        Given the initialize `random_state` from the hyperparameter, we set the
        random seed value of the local training handler.

        :return: `None`
        """
        rng = check_random_state(self.random_state)
        self._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")

    def generate_sketch(self, party_id=None):
        if not hasattr(self, "party_id"):
            logger.info("generate_sketch: Called before initialization")
            return
        if self.party_id is None:
            self.party_id = party_id

        logger.info("generate_sketch: Generate Local Party Data Sketch [party_id={}]".format(self.party_id))

        # Initialize Local Party Data Parameters
        self.n_samples = self.X_train.shape[0]

        # Initialize Sketch - TODO: Parameterize the sketch to choose sketch type.
        logger.info("generate_sketch: > Initialize Feature-Wise Sketchers")
        self.sketchers = [DDSketch(relative_accuracy=self.data_sketch_accuracy) for i in range(self.X_train.shape[1])]

        # Append Data to Sketch - TODO: Parallelize this process if necessary.
        logger.info("generate_sketch: > Allocate Data to Sketch Objects")
        for f_idx in range(self.X_train.shape[1]):
            for x in self.X_train[:, f_idx]:
                if x != np.NaN:
                    # TODO: Add DP Noise in Data - Use epsilon parameter for noise.
                    self.sketchers[f_idx].add(x)

        # Obtain Percentile Data
        logger.info("generate_sketch: > Generate Percentile Representations of Training Dataset")
        masked_x_train = np.ma.masked_invalid(self.X_train)
        self.X_ptiles_train = np.transpose(
            np.array(
                [
                    np.zeros_like(masked_x_train[:, idx])
                    if masked_x_train[:, idx].mask.all()
                    else mstats.rankdata(masked_x_train[:, idx]) * 100 / masked_x_train.shape[0]
                    for idx in range(masked_x_train.shape[1])
                ]
            )
        )
        self.X_ptiles_train[self.X_ptiles_train == 0] = -1

        # Perform Data Compression of Percentile Data (Experimental)
        # TODO: Enable this in a future release.
        if False:
            logger.info("> Data Compression Enabled - Performing Percentile Data Compression")
            X_ptile_train_compressed = compress(list(self.X_ptiles_train.flatten()), precision=4)
            data_dim = self.X_ptiles_train.shape

            # Compute Compression Ratio
            orig_size = sum(sys.getsizeof(i) for i in self.X_ptiles_train.flatten())
            comp_size = sys.getsizeof(X_ptile_train_compressed)
            ratio = (orig_size - comp_size) / orig_size

            logger.info("generate_sketch: > Compression Ratio: " + str(ratio))

            # Transmit X_ptile_train_compressed and data_dim to Aggregator

        return self.party_id, self.sketchers, self.X_ptiles_train

    def init_null_preds(self, baseline):
        """
        Initialize null predictions from baseline models. These values are used
        to construct the initial tree structure of the model. Note that
        raw_predictions has shape (n_samples, n_trees_per_iteration) whereas
        n_trees_per_iteration is n_classes in multiclass classification, else 1.
        Furthermore, setups initial data structures for metrics and performs
        early stop validation checks.

        :return: `None`
        """

        if not hasattr(self, "party_id"):
            logger.info("init_null_preds: Called before initialization")
            return

        logger.info(
            "init_null_preds: Generating Initial Predictions from Null Model [party_id={}]".format(self.party_id)
        )

        # Generate Baseline Predictions
        self.baseline = baseline
        self.raw_predictions = np.zeros(shape=(self.n_samples, self.n_trees), dtype=self.baseline.dtype) + self.baseline

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
        # we don't support n_threads currently
        self.fl_model.n_threads = 1
        self.fl_model.n_features_ = self.X.shape[1]
        if is_classifier(self):
            self.fl_model.classes_ = self.classes_
        self.fl_model._in_fit = True

    def init_grad(self):
        """
        Process to initialize gradients and hessians on the Local Training
        Handler. This is an empty data structure used to contain and later
        will be used to send over to aggregator.
        Note: shape = (n_samples, n_trees_per_iteration)

        :return: `None`
        """

        if not hasattr(self, "party_id"):
            logger.info("init_grad: Called before initialization")
            return

        logger.info("init_grad: Begin [party_id={}]".format(self.party_id))

        self.gradient, self.hessian = self.loss_.init_gradient_and_hessian(
            n_samples=self.n_samples, dtype=G_H_DTYPE, order="F"
        )

    def update_grad(self):
        """
        A wrapper function to update the corresponding gradient and hessian
        statistics inplace based on the raw predictions from the previous
        iteration of the model.

        :return: `None`
        """

        if not hasattr(self, "party_id"):
            logger.info("update_grad: Called before initialization")
            return

        logger.info("update_grad: Begin [party_id={}]".format(self.party_id))

        if (
            not hasattr(self, "n_completed_trains_at_rejoin")
            or self.n_completed_trains > self.n_completed_trains_at_rejoin
        ):
            self.get_train_metrics_pre()

        if self.loss_.constant_hessian:
            self.loss_.gradient(
                y_true=self.y_train,
                raw_prediction=self.raw_predictions,
                sample_weight=self.sample_weight,
                gradient_out=self.gradient,
            )
        else:
            self.loss_.gradient_hessian(
                y_true=self.y_train,
                raw_prediction=self.raw_predictions,
                sample_weight=self.sample_weight,
                gradient_out=self.gradient,
                hessian_out=self.hessian,
            )

    def collect_hist(self, k, party_id=None):
        """
        We return values of the surrogate histogram values, gradients, hessians,
        and the number of non-missing bin counts from the local worker.

        :param k: The corresponding feature index of the dataset.
        :type k: `int`
        :return: Returns a tuple comprising of the histogram bin index values, \
        gradient, hessian, missing value counts per feature, bin threshold value.
        :rtype: (`np.array`, `np.array`, `np.array`, `list`, `list`)
        """

        if not hasattr(self, "party_id"):
            logger.info("collect_hist: Called before initialization")
            return
        if self.party_id is None:
            self.party_id = party_id

        logger.info("collect_hist: Begin [party_id={}]".format(self.party_id))

        if self.gradient.ndim == 1:
            g_view = self.gradient.reshape((-1, 1))
            h_view = self.hessian.reshape((-1, 1))
        else:
            g_view = self.gradient
            h_view = self.hessian
        return self.party_id, g_view[:, k], h_view[:, k]

    def set_cat_bitsets(self, known_cat_bitsets, f_idx_map):
        """
        Redistribute the known cateogrical bitset values from the aggregator's
        bin mapper function back to the local party's fl model object. These
        two parameters are now required for performing inference over raw data.

        :param known_cat_bitsets: Array of bitsets of known categories, for \
        each categorical feature. Numpy dimension of (n_categorical_features, 8)
        :type: kwown_cat_bitsets: `np.array`
        :param f_idx_map: Map from original feature index to the corresponding \
        index in the known_cat_bitsets array. Numpy dimension of (n_features,)
        :type: f_idx_map: `np.array`

        :return: `None`
        """

        if not hasattr(self, "party_id"):
            logger.info("set_cat_bitsets: Called before initialization")
            return

        logger.info("set_cat_bitsets: Begin [party_id={}]".format(self.party_id))

        self.fl_model.known_cat_bitsets = known_cat_bitsets
        self.fl_model.f_idx_map = f_idx_map

    def update_raw_preds(self, k, predictor):
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

        :return: `None`
        """

        if not hasattr(self, "party_id"):
            logger.info("update_raw_preds: Called before initialization")
            return

        logger.info("update_raw_preds: Begin [party_id={}]".format(self.party_id))
        logger.debug("update_raw_preds: [k={}] [predictor={}]".format(k, predictor))

        # Define Predict Function
        predict = partial(
            predictor.predict,
            known_cat_bitsets=self.fl_model.known_cat_bitsets,
            f_idx_map=self.fl_model.f_idx_map,
            n_threads=self.fl_model.n_threads,
        )

        # Update Raw Prediction
        self.raw_predictions[:, k] += predict(self.X_train)

    def __data_val_enc(self):
        """
        Performs a validation procedure of the input dataset to correspondingly
        check the data types as well as the data encoding.

        :return: `None`
        """

        (x, y), (_) = self.data_handler.get_data()

        # Perform Data Encoding Validation
        self.X, y = check_X_y(x, y, dtype=[X_DTYPE], force_all_finite=False)
        self.Y = self.encode_target(y)

        # TODO load sample_weight from config
        self.sample_weight = None

    def set_fit(self, in_fit):
        """
        Sets the state of the in fit function, which is used to determine
        whether or not to used the binned functionality during inference.

        :param in_fit: Attribute to set the in fit function to.
        :type  in_fit: `Boolean`
        :return: `None`
        """

        if not hasattr(self, "party_id"):
            logger.info("set_fit: Called before initialization")
            return

        logger.info("set_fit: Begin [party_id={}]".format(self.party_id))

        self.fl_model._in_fit = in_fit

    def get_metrics(self):
        """
        Returns party side metrics.
        Adding the metrics to the reply message is done by party_protocol_handler.py.
        Therefore, this function returns None as the payload part.

        :return: `None`
        """

        if not hasattr(self, "party_id"):
            logger.info("get_metrics: Called before initialization")
            return None

        logger.info("get_metrics: Begin [party_id={}]".format(self.party_id))

        self.get_train_metrics_post()

        return None

    def determine_train_msg_seq(self, message_type, payload: dict):
        """
        Determines if the specific training message is the first or the last in a training round sequence.

        :param message_type: The type of the message.
        :type message_type: `enum`
        :param payload: Payload of the message.
        :type payload: `dict`
        :return: Tuple indicating if the message is first or last.
        :rtype: `tuple`
        """
        first_step = last_step = False
        if message_type is MessageType.TRAIN.value:
            first_step = payload.get("func") == "update_grad"
            if hasattr(self, "last_step"):
                if "update_raw_preds" == self.last_step:
                    last_step = (payload.get("func") == self.last_step) and (self.n_trees - 1) == payload.get(
                        "args"
                    ).get("k")
                else:
                    last_step = payload.get("func") == self.last_step
        return (first_step, last_step)

    @abstractmethod
    def encode_target(self, y=None):
        raise NotImplementedError


class XGBRegressorLocalTrainingHandler(XGBoostBaseLocalTrainingHandler):
    _VALID_LOSSES = "least_squares"

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

    def get_avg_stats(self):
        """
        Helper process to compute necessary statistics for computing the global
        average. Wrapper function calls respective function to derive statistics
        based on the defined loss function.

        :return: Returns y_hat * n and n.
        :rtype: (`np.array`, `np.array`)
        """

        if not hasattr(self, "party_id"):
            logger.info("get_avg_stats: Called before initialization")
            return

        logger.info("get_avg_stats: Compute Local Party Average Statistics [party_id={}]".format(self.party_id))

        return (np.array(np.average(self.y_train, weights=self.sample_weight)), np.array([len(self.y_train)]))


class XGBClassifierLocalTrainingHandler(XGBoostBaseLocalTrainingHandler):
    _VALID_LOSSES = ("binary_crossentropy", "categorical_crossentropy", "auto")

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
            raise ValueError(
                "Number of classes defined in configuration file "
                "and the classes derived from the data does not "
                "match. Found {0} classes, while config file "
                "is defined as {1} classes.".format(self.classes_.shape[0], self.num_classes)
            )

        if self.loss == "auto":
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
        if self.loss == "categorical_crossentropy" and self.n_trees == 1:
            raise ValueError("Incompatible loss and target variable counts.")

        if self.loss == "auto":
            return (
                _LOSSES["binary_crossentropy"](sample_weight=sample_weight)
                if self.n_trees == 1
                else _LOSSES["categorical_crossentropy"](sample_weight=sample_weight, n_classes=self.n_trees)
            )
        elif self.loss == "categorical_crossentropy":
            return _LOSSES[self.loss](sample_weight=sample_weight, n_classes=self.n_trees)
        else:
            return _LOSSES[self.loss](sample_weight=sample_weight)

    def get_avg_stats(self):
        """
        Helper process to compute necessary statistics for computing the global
        average. Wrapper function calls respective function to derive statistics
        based on the defined loss function.

        :return: Returns y_hat * n and n.
        :rtype: (`np.array`, `np.array`)
        """

        if not hasattr(self, "party_id"):
            logger.info("get_avg_stats: Called before initialization")
            return

        logger.info("get_avg_stats: Compute Local Party Average Statistics [party_id={}]".format(self.party_id))

        # Compute Average Depending on the Value
        if (self.n_trees == 1) or self.loss == "binary_crossentropy":
            return (np.array([np.average(self.y_train, weights=self.sample_weight)]), np.array([len(self.y_train)]))
        elif self.loss == "categorical_crossentropy":
            average, counts = [], []
            for k in range(self.num_classes):
                average.append(np.average(self.y_train == k, weights=self.sample_weight))
                counts.append(len(self.y_train == k))
            return (np.array(average), np.array(counts))
