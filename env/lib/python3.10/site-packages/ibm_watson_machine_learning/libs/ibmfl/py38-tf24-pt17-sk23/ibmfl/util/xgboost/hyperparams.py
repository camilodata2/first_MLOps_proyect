#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import logging
from ibmfl.exceptions import HyperparamsException

VALID_LOSSES = ['binary_crossentropy', 'categorical_crossentropy',
                'least_squares', 'least_absolute_deviation', 'auto']
VALID_SCORES = []

logger = logging.getLogger(__name__)


def validate_parameters(hyperparams):
    """
    Valdiates specific hyperparameters parameters that are defined in
    `hyperparams` variables. Raises an exception for certain parameters that
    are either required or beyond the defined range if they are defined.
    :param hyperparams: Dictionary containing values for the hyperparameters.
    :type  hyperparams: `dict`
    :rtype: None
    """
    try:
        # Check Hyperparameter Type
        if not isinstance(hyperparams, dict):
            raise ValueError('Provided hyperparameter is not valid.')

        # Global Hyperparameter Check
        if 'global' in hyperparams:
            params = hyperparams['global']
        else:
            raise ValueError('Global parameters have not been defined.')

        if 'learning_rate' in params:
            if params['learning_rate'] <= 0:
                raise ValueError('learning_rate={} must be strictly '
                                 'positive'.format(params['learning_rate']))
        else:
            raise ValueError('learning_rate has not been defined.')

        if 'loss' in params:
            if params['loss'] not in VALID_LOSSES:
                raise ValueError('Loss {} is currently not supported.'
                                 'Accpted losses: {}'.format(params['loss'],
                                                             ', '.join(VALID_LOSSES)))
        else:
            raise ValueError('loss has not been defined.')

        if 'num_classes' in params:
            # Validate Class Parameter Types
            if type(params['num_classes']) is not int:
                raise ValueError('Provided classes value\'s type is not valid, '
                                 'should be an int value >= 2 for classification.')

            # Validate Classification Case
            if params['loss'] != 'least_squares':
                if params['num_classes'] < 0:
                    raise ValueError('Provided class value must be >= 2 for '
                                     'classification.')

                if params['loss'] == 'binary_crossentropy' and \
                        params['num_classes'] != 2:
                    raise ValueError('Binary class models must have class of 2.')
                elif params['loss'] == 'categorical_crossentropy' and \
                        params['num_classes'] <= 2:
                    raise ValueError('Multiclass models must have class > 2.')
                elif params['loss'] == 'auto':
                    if params['num_classes'] < 2:
                        raise ValueError('Class value must be >= 2.')
                    else:
                        logging.warning(
                            'Obtaining class labels based on local dataset. '
                            'This may cause failures during aggregation '
                            'when parties have distinctive class labels.')
        else:
            # Handle Classes Not Defined Case
            if params['loss'] != 'least_squares':
                raise ValueError('Classes has not been defined. Should provide '
                                 'a value >= 2 for classification models.')

        if 'max_bins' in params:
            if not (2 <= params['max_bins'] and params['max_bins'] <= 255):
                raise ValueError('max_bins={} should be no smaller than 2 '
                                 'and no larger than 255.'.format(params['max_bins']))

        if 'max_iter' in params:
            if params['max_iter'] < 1:
                raise ValueError('max_iter={} must not be smaller '
                                 'than 1.'.format(params['max_iter']))
        else:
            raise ValueError('max_iter has not been defined.')

        if 'max_depth' in params:
            if params['max_depth'] is not None and params['max_depth'] < 0:
                raise ValueError('max_depth={} must be strictly greater'
                                 'than 0.'.format(params['max_depth']))

        if 'max_leaf_nodes' in params:
            if params['max_leaf_nodes'] is not None and params['max_leaf_nodes'] <= 1:
                raise ValueError('max_leaf_nodes={} must be strictly greater'
                                 'than 1.'.format(params['max_leaf_nodes']))

        if 'min_samples_leaf' in params:
            if params['min_samples_leaf'] is not None and params['min_samples_leaf'] < 0:
                raise ValueError('min_sample_leaf={} must not be smaller '
                                 'than 0'.format(params['min_samples_leaf']))

    except Exception as ex:
        logger.exception(str(ex))
        raise HyperparamsException('Defined global hyperparameters malformed. '+str(ex))


def init_parameters(obj, hyperparameters):
    """
    Initializes the hyperparameters to the corresponding object as an attribute.
    The parameters are propagated by reference, so changes should propagate to
    respective object passed into the function.
    :param obj: Any class object which to tie initialize the hyperparameters with.
    :type  obj: Object
    :param hyperparameters: Dictionary containing values for the hyperparameters.
    :type  hyperparameters: `dict`
    """
    # Initialize Global Configuration Parameter
    params = hyperparameters['global']
    setattr(obj, 'param', params)

    # Initialize Attributes (Pre-Checked Parameters)
    setattr(obj, 'learning_rate', params['learning_rate'])
    setattr(obj, 'loss', params['loss'])
    setattr(obj, 'max_iter', params['max_iter'])

    if params['loss'] == 'least_squares':
        setattr(obj, 'num_classes', 1)
    elif params['loss'] in ['binary_crossentropy', 'categorical_crossentropy', 'auto']:
        setattr(obj, 'num_classes', params['num_classes'])

    # Initialize Attributes (Optional Values - Based on Default Parameters)
    if 'l2_regularization' not in params or params['l2_regularization'] is None:
        setattr(obj, 'l2_regularization', 0)
    else:
        setattr(obj, 'l2_regularization', params['l2_regularization'])

    if 'max_bins' not in params:
        setattr(obj, 'max_bins', 255)
    else:
        setattr(obj, 'max_bins', params['max_bins'])

    if 'max_depth' not in params or params['max_depth'] is None:
        setattr(obj, 'max_depth', None)
    else:
        setattr(obj, 'max_depth', params['max_depth'])

    if 'max_leaf_nodes' not in params or params['max_leaf_nodes'] is None:
        setattr(obj, 'max_leaf_nodes', 31)
    else:
        setattr(obj, 'max_leaf_nodes', params['max_leaf_nodes'])

    if 'min_samples_leaf' not in params or params['min_samples_leaf'] is None:
        setattr(obj, 'min_samples_leaf', 20)
    else:
        setattr(obj, 'min_samples_leaf', params['min_samples_leaf'])

    if 'random_state' in params:
        setattr(obj, 'random_state', params['random_state'])
    else:
        setattr(obj, 'random_state', None)

    if 'scoring' in params:
        setattr(obj, 'scoring', params['scoring'])
    else:
        setattr(obj, 'scoring', None)

    if 'verbose' not in params or params['verbose'] is None:
        setattr(obj, 'verbose', False)
    else:
        setattr(obj, 'verbose', True)

    return obj
