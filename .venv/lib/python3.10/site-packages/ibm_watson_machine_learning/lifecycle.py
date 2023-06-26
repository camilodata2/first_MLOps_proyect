#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from enum import Enum


class SpecStates(Enum):
    SUPPORTED = 'supported'
    UNSUPPORTED = 'unsupported'
    DEPRECATED = 'deprecated'
    CREATE_UNSUPPORTED = 'create-unsupported'
