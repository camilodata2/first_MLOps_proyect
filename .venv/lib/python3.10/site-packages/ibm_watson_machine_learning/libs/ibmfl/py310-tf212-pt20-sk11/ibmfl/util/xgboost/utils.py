#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function

from sklearn._loss.loss import (
    _LOSSES,
    AbsoluteError,
    BaseLoss,
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
    HalfSquaredError,
    PinballLoss,
)


def is_classifier(cls):
    """
    Auxillary function to validate whether a given object is a
    classification-based model. (Can be used for the Fusion Handler as well as
    the Local Training Handler).

    :param cls: Object of interest to identify the type of model.
    :type  cls: `Object`
    :return `bool`
    """
    return cls.loss == "auto" or cls.loss == "binary_crossentropy" or cls.loss == "categorical_crossentropy"


_LOSSES = _LOSSES.copy()
_LOSSES.update(
    {
        "least_squares": HalfSquaredError,
        "least_absolute_deviation": AbsoluteError,
        "binary_crossentropy": HalfBinomialLoss,
        "categorical_crossentropy": HalfMultinomialLoss,
    }
)
