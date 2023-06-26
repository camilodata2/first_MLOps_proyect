#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from pyspark import SparkContext
from py4j.java_collections import MapConverter
import inspect

class Sink (object):

    """
    Class for creating a Sink object to sink data to an external DataSource like AmazonS3, Swift, DashDB etc.
    A Sink object can be the last stage of a :class:`pipeline.DAG.DAG`

    :param symbolicconst: Connection string to access the external DataSource.
    :param optionsmap: A dictionary specifying the options to be passed to the DataSource.
    """
    _sc = SparkContext._active_spark_context
    _logSrcLang = "Py:"

    def __init__(self, symbolicconst, optionsmap):

        self._jPipeline = self._sc._jvm.com.ibm.analytics.ngp.pipeline.pythonbinding.Pipelines
        self._jLogger = self._sc._jvm.org.apache.log4j.Logger
        self._to_map = self._jPipeline.toScalaMap
        self.logger = self._jLogger.getLogger(self.__class__.__name__)
        methodname = str(inspect.stack()[0][3])

        joptionsmap = MapConverter().convert(optionsmap, self._sc._gateway._gateway_client)
        sca_map = self._to_map(joptionsmap)
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: symbolicconst => " + str(
            symbolicconst) + " | optionsmap => " + str(optionsmap) + "]"
        self.logger.info(logMsg)
        self._jSink = self._jPipeline.getSink().apply(symbolicconst,sca_map)

    def run(self, dataframe):
        """
        Used to execute a :class:`Sink` object referring to an external DataSource, passing a dataframe.

        :param dataframe: A DataFrame
        :return: None
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: dataframe => " + str(
            dataframe) + "]"
        self.logger.info(logMsg)

        self._jSink.run (dataframe._jdf)

    def _to_java(self):
        return self._jSink
