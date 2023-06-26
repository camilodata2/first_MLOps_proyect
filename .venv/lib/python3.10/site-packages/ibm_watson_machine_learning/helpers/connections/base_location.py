#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC

__all__ = [
    "BaseLocation"
]


class BaseLocation(ABC):
    """Base class for storage Location."""
    def to_dict(self) -> dict:
        """Get a json dictionary representing this model."""
        return vars(self)

    def _get_file_size(self, **kwargs) -> 'int':
        raise NotImplementedError
