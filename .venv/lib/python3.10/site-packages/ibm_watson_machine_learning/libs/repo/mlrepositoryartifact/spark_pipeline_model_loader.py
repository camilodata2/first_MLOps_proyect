#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .spark_artifact_loader import SparkArtifactLoader


class SparkPipelineModelLoader(SparkArtifactLoader):
    """
        Returns pipeline model instance associated with this model artifact.

        :return: pipeline model
        :rtype: pyspark.ml.PipelineModel
        """
    def model_instance(self):
        return self.load()