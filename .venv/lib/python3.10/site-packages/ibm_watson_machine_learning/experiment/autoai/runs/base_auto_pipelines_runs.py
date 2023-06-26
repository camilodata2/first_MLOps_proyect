#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pandas import DataFrame
    from ibm_watson_machine_learning.helpers import DataConnection

__all__ = [
    "BaseAutoPipelinesRuns"
]


class BaseAutoPipelinesRuns:
    """Base abstract class for Pipeline Optimizers Runs."""

    @abstractmethod
    def list(self) -> 'DataFrame':
        """Lists historical runs/fits with status."""
        pass

    @abstractmethod
    def get_params(self, run_id: str = None) -> dict:
        """Get executed optimizers configs parameters based on the run_id."""
        pass

    @abstractmethod
    def get_run_details(self, run_id: str = None) -> dict:
        """Get run details. If run_id is not supplied, last run will be taken."""
        pass

    @abstractmethod
    def get_optimizer(self, run_id: str):
        """Create instance of AutoPipelinesRuns with all computed pipelines computed by AutoAi on WML."""
        pass

    @abstractmethod
    def get_data_connections(self, run_id: str) -> List['DataConnection']:
        """Create DataConnection objects for further user usage"""
        pass
