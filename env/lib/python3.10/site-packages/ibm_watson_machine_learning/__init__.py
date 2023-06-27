#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

pkg_name = "ibm-watson-machine-learning"

try:
    from importlib.metadata import version
    version = version(pkg_name)

except (ModuleNotFoundError, AttributeError):
    from importlib_metadata import version as imp_lib_ver
    version = imp_lib_ver(pkg_name)

from ibm_watson_machine_learning.client import APIClient
APIClient.version = version

from .utils import is_python_2
if is_python_2():
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Python 2 is not officially supported.")