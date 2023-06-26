#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .spark_artifact_loader import SparkArtifactLoader


class SparkPipelineLoader(SparkArtifactLoader):
    """
    Returns pipeline instance associated with this pipeline artifact.

    :return: pipeline
    :rtype: pyspark.ml.Pipeline
    """
    def pipeline_instance(self):
        return self.load()
