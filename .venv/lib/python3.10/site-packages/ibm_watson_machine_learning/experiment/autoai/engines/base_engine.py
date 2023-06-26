#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame
    from sklearn.pipeline import Pipeline
    from ibm_watson_machine_learning.helpers.connections import DataConnection

__all__ = [
    "BaseEngine"
]


class BaseEngine(ABC):
    """Base abstract class for Engines."""

    @abstractmethod
    def get_params(self) -> dict:
        """Fetch configuration parameters"""
        pass

    @abstractmethod
    def fit(self,
            training_data_reference: List['DataConnection'],
            training_results_reference: 'DataConnection',
            background_mode: bool = True) -> dict:
        """Schedule a fit/run/training."""
        pass

    @abstractmethod
    def get_run_status(self) -> str:
        """Fetch status of a training."""
        pass

    @abstractmethod
    def get_run_details(self) -> dict:
        """Fetch training details"""
        pass

    @abstractmethod
    def summary(self) -> 'DataFrame':
        """Fetch all pipelines results"""
        pass

    @abstractmethod
    def get_pipeline_details(self, pipeline_name: str = None) -> dict:
        """Fetch details of particular pipeline"""
        pass

    @abstractmethod
    def get_pipeline(self, pipeline_name: str, local_path: str = '.') -> 'Pipeline':
        """Download and load computed pipeline"""
        pass

    @abstractmethod
    def get_best_pipeline(self, local_path: str = '.') -> 'Pipeline':
        """Download and load the best pipeline"""
        pass
