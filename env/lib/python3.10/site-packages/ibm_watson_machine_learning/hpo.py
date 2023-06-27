#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.wml_client_error import WMLClientError, UnexpectedType
from ibm_watson_machine_learning.wml_resource import WMLResource


class BothValuesAndRangePassed(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, "Both values and range were passed. Only one should be used at the same time.")


class EmptyValues(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, "No values were passed.")


class InconsistentValuesTypes(WMLClientError):
    def __init__(self, types):
        WMLClientError.__init__(self, "Passed values have inconsistent types: {}".format(types))


class MissingElementOfRange(WMLClientError):
    def __init__(self, el):
        WMLClientError.__init__(self, "Missing element of range: {}".format(el))

def HPOParameter(name, values=None, max=None, min=None, step=None):
    """Prepares dict element describing hyper parameter. Hyper parameter may be contructed in two ways:

    * ``HPOParameter('name', values=['val1', 'val2'])   # values``
    * ``HPOParameter('name', min=0.5, max=10, step=0.1) # range``

    In range description `min` and `step` are optional (by default `min` is set to 0 and `step` is set to 1).

    :param name: name of parameter
    :type name: str
    :param values: if parameter should have only values provided by user, this param should contain these values
    :type values: list[str] or list[int] or list[float], optional
    :param max: if parameter should contain numbers from range, this will be maximal value of this range
    :type max: int or float, optional
    :param min: if parameter should contain numbers from range, this will be minimal value of this range
    :type min: int or float, optional
    :param step: if parameter should contain numbers from range, this will be step between elements from range
    :type step: int or float, optional

    :return: description of HPO parameter
    :rtype: dict

    **Example**

    .. code-block:: python

        HPOParameter('param1', values=['a', 'b', 'c']),
        HPOParameter('param2', values=[0, 1, 9]),
        HPOParameter('param3', values=[0.1, 0.5, 0.8]),
        HPOParameter('param4', max=10),
        HPOParameter('param5', min=2, max=10),
        HPOParameter('param6', max=10, step=2),
        HPOParameter('param7', max=10.0),
        HPOParameter('param8', min=0.1, max=10),
        HPOParameter('param9', min=0.5, max=10, step=0.1)

    """
    WMLResource._validate_type(name, "name", str, True)
    WMLResource._validate_type(values, "values", list, False)
    WMLResource._validate_type(max, "max", [int, float], False)
    WMLResource._validate_type(min, "min", [int, float], False)
    WMLResource._validate_type(step, "step", [int, float], False)

    if values is not None and (max is not None or min is not None or step is not None):
        raise BothValuesAndRangePassed()
    elif values is not None:
        types = [type(v) for v in values]
        types = list(set(types))

        if len(values) == 0:
            raise EmptyValues()

        if len(types) > 1:
            raise InconsistentValuesTypes(types)

        if types[0] is str:
            return {
                "name": name,
                "string_values": values
            }
        elif types[0] is int:
            return {
                "name": name,
                "int_values": values
            }
        elif types[0] is float:
            return {
                "name": name,
                "double_values": values
            }
        else:
            raise UnexpectedType("values", "str or int or float", types[0])
    elif max is not None:
        if min is None:
            min = 0

        if step is None:
            step = 1

        if type(max) is float or type(min) is float or type(step) is float:
            return {
                "name": name,
                "double_range": {
                    "min_value": min,
                    "max_value": max,
                    "step": step
                }
            }
        else:
            return {
                "name": name,
                "int_range": {
                    "min_value": min,
                    "max_value": max,
                    "step": step
                }
            }
    else:
        raise MissingElementOfRange("max")


def HPOMethodParam(name=None, value=None):
    result = {}

    if name is not None:
        result['name'] = name

    if type(value) == str:
        result['string_value'] = value
    elif type(value) == float:
        result['double_value'] = value
    elif type(value) == int:
        result['int_value'] = value
    else:
        raise UnexpectedType('value', 'str or float or int', type(value))

    return result