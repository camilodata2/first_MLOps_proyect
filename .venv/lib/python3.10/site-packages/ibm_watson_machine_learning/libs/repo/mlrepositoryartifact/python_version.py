#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import sys


class PythonVersion(object):
    @staticmethod
    def significant():
        return "{}.{}".format(sys.version_info[0], sys.version_info[1])
