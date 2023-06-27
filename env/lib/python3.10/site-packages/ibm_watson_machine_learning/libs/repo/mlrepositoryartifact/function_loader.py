#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.function_artifact_loader  import FunctionArtifactLoader


class FunctionLoader(FunctionArtifactLoader):
    """
        Returns  Generic function instance associated with this function artifact.

        :return: function
        :rtype:
        """

    def function_instance(self):
        """
         :return: returns function path
         """
        return self.load()

    def download_function(self,path):
        """
         :return: returns function path
         """
        return self.load(path)

