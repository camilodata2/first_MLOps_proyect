#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import abc


class KeyManager(abc.ABC):
    """
    The abstract class for `KeyManager` object.
    """

    @abc.abstractmethod
    def initialize_keys(self, **kwargs):
        """
        initialize key(s) for a crypto system.

        :param kwargs: Parameters required for the private key generation.
        :type kwargs: `dict`
        """
        raise NotImplementedError
