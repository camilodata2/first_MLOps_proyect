#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import mlpipelinepy


class MLPipelineVersion(type):
    _version = mlpipelinepy.__version__
    _version_parts = _version.split(".")

    @classmethod
    def major_ver(mcs):
        return int(mcs._version_parts[0])

    @classmethod
    def minor_ver(mcs):
        return int(mcs._version_parts[1])

    @classmethod
    def full_version(mcs):
        return mcs._version
