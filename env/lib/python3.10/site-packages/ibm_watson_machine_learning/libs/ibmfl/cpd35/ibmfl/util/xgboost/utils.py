#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function


def is_classifier(cls):
    """
    Auxillary function to validate whether a given object is a
    classification-based model. (Can be used for the Fusion Handler as well as
    the Local Training Handler).

    :param cls: Object of interest to identify the type of model.
    :type  cls: `Object`
    :return `bool`
    """
    return cls.loss == 'auto' or cls.loss == 'binary_crossentropy' or \
        cls.loss == 'categorical_crossentropy'
