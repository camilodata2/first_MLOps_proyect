#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC, abstractmethod

__all__ = [
    "BaseExperiment"
]


class BaseExperiment(ABC):
    """Base abstract class for Experiment."""

    @abstractmethod
    def runs(self, *, filter: str):
        """Get the historical runs but with WML Pipeline name filter.

        :param filter: WML Pipeline name to filter the historical runs
        :type filter: str
        """
        pass
