#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import joblib
import logging

from sklearn.base import is_classifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE

from sklearn.ensemble._hist_gradient_boosting.loss import BinaryCrossEntropy, \
    CategoricalCrossEntropy, LeastSquares

logger = logging.getLogger(__name__)


def export_sklearn(model, full_path=None):
    """
    Auxillary function to first convert the FL XGBoost Model to Scikit Learn's
    Histogram Gradient Boosting model object, then persist the object as a
    pickled model object. This enables standalone use of the trained model
    without dependency to FL's libraries.

    Parameters/Features that are Not Supported or Transferred:
    - monotonic_cst (in v0.21)
    - warm_start
    - early_stopping
    - scoring
    - validation_fraction
    - n_iter_no_change
    - tol

    :param model: The FL XGBoost model object to convert from.
    :type  model: `XGBFLModel`
    :param full_path: The full absolute path of where the model will persist.
    :type  full_path: `str`
    :return: `None`
    """
    # Initialize Model Object (Parameter Transfer from FL Model to Sklearn)
    if model.loss == 'least_squares':
        export_model = HistGradientBoostingRegressor(
            loss=model.loss,
            learning_rate=model.learning_rate,
            max_iter=model.max_iter,
            max_leaf_nodes=model.max_leaf_nodes,
            max_depth=model.max_depth,
            min_samples_leaf=model.min_samples_leaf,
            l2_regularization=model.l2_regularization,
            max_bins=model.max_bins,
            verbose=model.verbose,
            random_state=model.random_state)
    elif model.loss == 'categorical_crossentropy' or \
            model.loss == 'binary_crossentropy' or \
            model.loss == 'auto':
        export_model = HistGradientBoostingClassifier(
            loss=model.loss,
            learning_rate=model.learning_rate,
            max_iter=model.max_iter,
            max_leaf_nodes=model.max_leaf_nodes,
            max_depth=model.max_depth,
            min_samples_leaf=model.min_samples_leaf,
            l2_regularization=model.l2_regularization,
            max_bins=model.max_bins,
            verbose=model.verbose,
            random_state=model.random_state)

    # Attribute Transfer
    export_model._baseline_prediction = model._baseline_prediction
    export_model.n_trees_per_iteration_ = model.n_trees
    export_model.n_features_ = model.n_features_

    # Model Predictor Object (Deep Object Rebuild)
    export_model._predictors = []
    for i, pred in enumerate(model._predictors):
        export_model._predictors.append([])
        for j, p in enumerate(pred):
            nodes = model._predictors[i][j].nodes.astype(
                PREDICTOR_RECORD_DTYPE).copy()
            export_model._predictors[i].append(TreePredictor(nodes))

    # Model Loss Function
    if hasattr(model, 'classes_'):
        if model.loss == 'binary_crossentropy' or len(model.classes_) == 2:
            export_model.loss_ = BinaryCrossEntropy(None)
        elif model.loss == 'categorical_crossentropy':
            export_model.loss_ = CategoricalCrossEntropy(None)
    elif model.loss == 'least_squares':
        export_model.loss_ = LeastSquares(None)

    # Target Class Encoding (For Classification Models)
    if hasattr(model, 'classes_'):
        export_model.classes_ = model.classes_

    # Perform Model Pickling
    with open(full_path, 'wb') as f:
        joblib.dump(export_model, f)
