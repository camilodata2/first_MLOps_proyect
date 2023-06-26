#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from pprint import pformat
from six import iteritems

class ContentStatus(object):
    def __init__(self, state=None, message=None, failure=None):
        """
        ModelMetrics - a model defined in Swagger

        :param dict swaggerTypes: The key is attribute name
                                  and the value is attribute type.
        :param dict attributeMap: The key is attribute name
                                  and the value is json key in definition.
        """
        self.swagger_types = {
            'state': 'str',
            'message': 'str',
            'failure': 'ErrorSchemaRepository'
        }

        self.attribute_map = {
            'state': 'state',
            'message': 'message',
            'failure': 'failure'
        }

        self._state = state
        self._message = message
        self._failure = failure

    @property
    def state(self):
        """
        Gets the code of this ContentStatus.


        :return: The code of this ContentStatus.
        :rtype: str
        """
        return self._state

    @state.setter
    def state(self, state):
        """
        Sets the state of this ContentStatus.


        :param state: The state of this ContentStatus.
        :type: str
        """
        self._state = state

    @property
    def message(self):
        """
        Gets the error message of this ContentStatus.


        :return: The error message of this ContentStatus.
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message):
        """
        Sets the message of this ContentStatus.


        :param message: The message of this ContentStatus.
        :type: str
        """
        self._message = message

    @property
    def failure(self):
        """
        Gets the details for the failure of this ContentStatus.


        :return: The the details for the failure of this ContetnStatus.
        :rtype: ErrorSchemaRepository
        """
        return self._failure

    @failure.setter
    def failure(self, failure):
        """
        Sets the failure details of this ContentStatus.


        :param failure: The failure details of this ContentStatus.
        :type: ErrorSchemaRepository
        """
        self._failure = failure

    def to_dict(self):
        """
        Returns the model properties as a dict
        """
        result = {}

        for attr, _ in iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """
        Returns the string representation of the model
        """
        return pformat(self.to_dict())

    def __repr__(self):
        """
        For `print` and `pprint`
        """
        return self.to_str()

    def __eq__(self, other):
        """
        Returns true if both objects are equal
        """
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """
        Returns true if both objects are not equal
        """
        return not self == other