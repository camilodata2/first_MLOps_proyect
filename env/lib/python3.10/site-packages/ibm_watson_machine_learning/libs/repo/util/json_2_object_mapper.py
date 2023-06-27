#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json
from collections import namedtuple


class Json2ObjectMapper(object):
    @staticmethod
    def read(json_holder):  # TODO map
        if json_holder is None:
            return {}
        elif isinstance(json_holder, str):
            return json.loads(json_holder)
        else:
            return json_holder  # TODO json.loads(json_str, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    @staticmethod
    def to_dict(json_str):
        if isinstance(json_str, str):
            return json.loads(json_str)
        else:
            raise ValueError('Incorrect type')

    @staticmethod
    def to_object(json_holder):
        if isinstance(json_holder, str):
            return json.loads(json_holder, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        elif isinstance(json_holder, object):
            return json_holder
        else:
            raise ValueError('Incorrect type')
