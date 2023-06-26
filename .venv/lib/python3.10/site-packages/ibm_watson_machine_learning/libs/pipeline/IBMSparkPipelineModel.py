#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import inspect
from pyspark import SparkContext
from pyspark.ml.pipeline import PipelineModel

from Wrapper import PythonJavaConversions, JavaParams


class IBMSparkPipelineModel(PipelineModel):
    """
    The IBMSparkPipelineModel extends from Spark ML PipelineModel.
    """
    _sc = SparkContext._active_spark_context

    def __init__(self, stages):
        self._jLogger = self._sc._jvm.org.apache.log4j.Logger
        self._logSrcLang = "Py:"
        self.logger = self._jLogger.getLogger(self.__class__.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + "stages => " + str(
                stages) + "]"
        self.logger.info(logMsg)

        super(IBMSparkPipelineModel, self).__init__(stages)

    def _to_java(self):
        """
        Transfer this instance to a Java PipelineModel.  Used for ML persistence.

        :return: Java object equivalent to this instance.
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: None]"
        self.logger.info(logMsg)

        gateway = SparkContext._gateway
        clsname = SparkContext._jvm.org.apache.spark.ml.Transformer
        java_stages = gateway.new_array(clsname, len(self.stages))
        for idx, stage in enumerate(self.stages):
            java_stages[idx] = PythonJavaConversions._to_java_stage(stage)

        _java_obj = \
            JavaParams._new_java_obj("org.apache.spark.ml.IBMSparkPipelineModel", self.uid, java_stages)

        return _java_obj

    def save(self, path):
        """
        Saves a :class:`IBMSparkPipelineModel` object to GPFS.

        :param filename: filename to be saved in GPFS.
        :return: None.
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " filename => " + str(
                path) + "]"
        self.logger.info(logMsg)

        if not isinstance(path, basestring):
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": filename should be a " \
                    "basestring, got " + str(type(path))
            self.logger.error(logMsg)
            raise TypeError("filename should be a basestring, got %s" % type(path))

        _jModel = self._to_java()
        _jModel.save(path)

    def saveGpfs(self, path):
        """
        Saves a :class:`IBMSparkPipelineModel` object to GIT.

        :param path: filename to be saved in GIT.
        :return: None.
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " filename => " + str(
                path) + "]"
        self.logger.info(logMsg)

        _jModel = self._to_java()
        _jModel.writer().saveGpfs(path)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java PipelineModel, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        # Load information from java_stage to the instance.

        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        _logSrcLang = "Py:"

        methodname = str(inspect.stack()[0][3])
        logMsg = _logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + " java_stage => " + str(
                java_stage) + "]"
        logger.info(logMsg)

        py_stages = [PythonJavaConversions._from_java_stage(s) for s in java_stage.stages()]
        # Create a new instance of this stage.
        py_stage = IBMSparkPipelineModel(py_stages)

        py_stage = PythonJavaConversions._resetUid(py_stage, java_stage.uid())
        return py_stage

    @classmethod
    def load(cls, path):
        """
        Loads a :class:`IBMSparkPipelineModel` object from the saved path in GPFS.

        :param path: filename which was saved in GPFS.
        :return: A :class:`IBMSparkPipelineModel` object.
        """
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        _logSrcLang = "Py:"
        methodname = str(inspect.stack()[0][3])
        logMsg = _logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + " filename => " + str(path) + "]"
        logger.info(logMsg)

        _jread = PythonJavaConversions._load_java_obj(cls)

        if not isinstance(path, basestring):
            logMsg = _logSrcLang + cls.__name__ + ":" + methodname + ": filename should be a basestring, got " \
                     + str(type(path))
            logger.error(logMsg)
            raise TypeError("filename should be a basestring, got type %s" % type(path))

        java_obj = _jread.load(path)

        return cls._from_java(java_obj)

    @classmethod
    def loadGpfs(cls, path):
        """
        Loads a :class:`IBMSparkPipelineModel` object from the saved path in GPFS.

        :param path: filename which was saved in GIT.
        :return: A :class:`IBMSparkPipelineModel` object.
        """
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        _logSrcLang = "Py:"
        methodname = str(inspect.stack()[0][3])
        logMsg = _logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + " filename => " + str(path) + "]"
        logger.info(logMsg)

        # return Reader(cls).load(path)

        _jread = PythonJavaConversions._load_java_obj(cls)

        if not isinstance(path, basestring):
            raise TypeError("filename should be a basestring, got type %s" % type(path))

        java_obj = _jread.reader().loadGpfs(path)

        return cls._from_java(java_obj)
