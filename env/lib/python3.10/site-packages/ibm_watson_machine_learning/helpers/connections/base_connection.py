#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC
from copy import deepcopy

__all__ = [
    "BaseConnection"
]


class BaseConnection(ABC):
    """Base class for storage Connections."""

    def to_dict(self) -> dict:
        """Get a json dictionary representing this model."""
        return vars(self)
