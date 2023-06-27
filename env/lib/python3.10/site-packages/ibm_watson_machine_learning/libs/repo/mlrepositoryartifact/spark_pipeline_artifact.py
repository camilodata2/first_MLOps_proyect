#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .spark_pipeline_reader import SparkPipelineReader
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames, MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepository import PipelineArtifact
from .version_helper import VersionHelper
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

if lib_checker.installed_libs[PYSPARK]:
    from pyspark.ml import Pipeline


class SparkPipelineArtifact(PipelineArtifact):
    """
    Class of pipeline artifacts created with MLRepositoryCLient.

    :param pyspark.ml.Pipeline ml_pipeline: Pipeline which will be wrapped

    :ivar pyspark.ml.Pipeline ml_pipeline: Pipeline associated with this artifact
    """
    def __init__(self, ml_pipeline, uid=None, name=None, meta_props=MetaProps({})):
        super(SparkPipelineArtifact, self).__init__(uid, name, meta_props)

        type_identified = False
        if lib_checker.installed_libs[PYSPARK]:
            if issubclass(type(ml_pipeline), Pipeline):
                type_identified = True

        if not type_identified and lib_checker.installed_libs[MLPIPELINE]:
            from mlpipelinepy.mlpipeline import MLPipeline
            if issubclass(type(ml_pipeline), MLPipeline):
                type_identified = True
        if not type_identified:
            raise ValueError('Invalid type for ml_pipeline: {}'.format(ml_pipeline.__class__.__name__))

        self.ml_pipeline = ml_pipeline
        self.meta.merge(
            MetaProps({
                MetaNames.FRAMEWORK_NAME: VersionHelper.pipeline_type(ml_pipeline),
                MetaNames.FRAMEWORK_VERSION: VersionHelper.getFrameworkVersion(ml_pipeline)
            })
        )

    def pipeline_instance(self):
        return self.ml_pipeline

    def reader(self):
        """
        Returns reader used for getting pipeline content.

        :return: reader for pyspark.ml.Pipeline
        :rtype: SparkPipelineReader
        """
        try:
            return self._reader
        except:
            self._reader = SparkPipelineReader(self.ml_pipeline, 'pipeline')
            return self._reader

    def _copy(self, uid):
        return SparkPipelineArtifact(self.ml_pipeline, uid, self.name, self.meta)
