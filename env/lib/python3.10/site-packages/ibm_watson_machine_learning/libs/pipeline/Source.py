#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from pyspark import SparkContext
from py4j.java_collections import MapConverter
from pyspark.sql import DataFrame,SQLContext
import inspect

class Source (object):
    """
    Class for creating a Source object to fetch data from an external DataSource like AmazonS3, Swift, DashDB etc.
    A Source object can be used as the first stage of a :class:`pipeline.DAG.DAG`

    :param symbolicconst: Connection string to access the external DataSource.
    :param optionsmap: A dictionary specifying the options to be passed to the DataSource.
    """
    _sc = SparkContext._active_spark_context
    _logSrcLang = "Py:"

    def __init__(self, symbolicconst, optionsmap):

        self._jPipeline = self._sc._jvm.com.ibm.analytics.ngp.pipeline.pythonbinding.Pipelines
        self._jLogger = self._sc._jvm.org.apache.log4j.Logger
        self._to_map = self._jPipeline.toScalaMap
        self.joptionsmap = MapConverter().convert(optionsmap, self._sc._gateway._gateway_client)
        sca_map = self._to_map(self.joptionsmap)
        self.logger = self._jLogger.getLogger(self.__class__.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: symbolicconst => " + str(
            symbolicconst) + " | optionsmap => " + str(optionsmap) + "]"
        self.logger.info(logMsg)

        self._jSource = self._jPipeline.getSource().apply(symbolicconst,sca_map)

    def run(self):
        """
        Used to retrieve the DataFrame from a :class:`Source` object that refers to an external DataSource.

        :return: A DataFrame
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: None]"
        self.logger.info(logMsg)

        return DataFrame(self._jSource.run(), SQLContext.getOrCreate(self._sc))

    def _to_java(self):
        return self._jSource
