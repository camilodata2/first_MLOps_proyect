"""
Module to where data handler are implemented.
"""
#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc

from ibmfl.data.data_handler import DataHandler
from ibmfl.data.env_spec import EnvHandler


class EnvDataHandler(DataHandler):
    """
    Base class to load data and  environment for reinforcement learning.
    """

    @abc.abstractmethod
    def get_data(self, **kwargs):
        """
        Read train data and test data for reinforcement learning
        :return:
        """

    @abc.abstractmethod
    def get_env_class_ref(self) -> EnvHandler:
        """
           Get environment reference for RL trainer, the instance is created in
           model class as part of trainer initialization
        """
