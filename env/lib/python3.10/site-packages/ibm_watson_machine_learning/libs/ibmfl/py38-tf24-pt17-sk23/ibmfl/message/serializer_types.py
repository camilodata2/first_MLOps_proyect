#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from enum import Enum


class SerializerTypes(Enum):
    """
    Types of supported Serializers
    """
    PICKLE = 1
    JSON_PICKLE = 2
