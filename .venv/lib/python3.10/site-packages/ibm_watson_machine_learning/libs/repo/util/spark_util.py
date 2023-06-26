#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

if lib_checker.installed_libs[PYSPARK]:
    from pyspark.ml import Pipeline, PipelineModel


class SparkUtil(object):
    DEFAULT_LABEL_COL = 'label'

    @staticmethod
    def get_label_col(spark_artifact):
        lib_checker.check_lib(PYSPARK)
        if isinstance(spark_artifact, PipelineModel):
            pipeline = Pipeline(stages=spark_artifact.stages)
            return SparkUtil.get_label_col_from__stages(pipeline.getStages())
        elif isinstance(spark_artifact, Pipeline):
            return SparkUtil.get_label_col_from__stages(spark_artifact.getStages())
        else:
            return SparkUtil.DEFAULT_LABEL_COL

    @staticmethod
    def get_label_col_from__stages(stages):
        lib_checker.check_lib(PYSPARK)
        label = SparkUtil._get_label_col_from_python_stages(stages)

        if label == SparkUtil.DEFAULT_LABEL_COL:
            label = SparkUtil._get_label_col_from_java_stages(stages)

        return label

    @staticmethod
    def _get_label_col_from_python_stages(stages):
        try:
            label_col = stages[-1].getLabelCol()
        except Exception as ex:
            label_col = SparkUtil.DEFAULT_LABEL_COL

        reversed_stages = stages[:]
        reversed_stages.reverse()

        for stage in reversed_stages[1:]:
            try:
                if stage.getOutputCol() == label_col:
                    label_col = stage.getInputCol()
            except Exception as ex:
                pass

        return label_col

    @staticmethod
    def _get_label_col_from_java_stages(stages):
        try:
            label_col = stages[-1]._call_java("getLabelCol")
        except Exception as ex:
            label_col = SparkUtil.DEFAULT_LABEL_COL

        reversed_stages = stages[:]
        reversed_stages.reverse()

        for stage in reversed_stages[1:]:
            try:
                if stage._call_java("getOutputCol") == label_col:
                    label_col = stage._call_java("getInputCol")
            except Exception as ex:
                pass

        return label_col
