#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pandas import DataFrame
    from ibm_watson_machine_learning.utils.autoai.enums import PipelineTypes
    from sklearn.pipeline import Pipeline
    from numpy import ndarray

__all__ = [
    "BaseAutoPipelines"
]


class BaseAutoPipelines:
    """Base abstract class for Pipeline Optimizers."""

    @abstractmethod
    def get_params(self) -> dict:
        """Get configuration parameters of AutoPipelines"""
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> 'Pipeline':
        """Run fit job."""
        pass

    @abstractmethod
    def summary(self) -> 'DataFrame':
        """List all computed pipelines."""
        pass

    @abstractmethod
    def get_pipeline_details(self, pipeline_name: str = None) -> dict:
        """Get details of computed pipeline. Details like pipeline steps."""
        pass

    @abstractmethod
    def get_pipeline(self, pipeline_name: str, astype: 'PipelineTypes') -> Union['Pipeline', 'TrainablePipeline']:
        """Get particular computed Pipeline"""
        pass

    def get_pipeline_notebook(self,
                     pipeline_name: str = None,
                     persist: 'bool' = False) -> Union['Pipeline', 'TrainablePipeline']:
        """Get particular computed pipeline notebook"""
        pass

    @abstractmethod
    def predict(self, X: Union['DataFrame', 'ndarray']) -> 'ndarray':
        """Use predict on top of the computed pipeline."""
        pass
