#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.generic_file_artifact_loader  import GenericFileArtifactLoader


class GenericFilePipelineModelLoader(GenericFileArtifactLoader):
    """
        Returns  Generic pipeline model instance associated with this model artifact.

        :return: pipeline model
        :rtype: spss.learn.Pipeline
        """
    def load_model(self):
        return(self.model_instance())


    def model_instance(self):
        """
         :return: returns Spss model path
         """
        return self.load()


    def pipeline_instance(self):
        """
         :return: returns Spss model path
         """
        return self.load()