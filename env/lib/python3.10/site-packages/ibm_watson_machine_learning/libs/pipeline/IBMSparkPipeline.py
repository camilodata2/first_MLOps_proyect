#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import inspect
import sys
import getpass
import os
from py4j.java_collections import MapConverter,ListConverter
from pyspark import SparkContext
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.util import keyword_only
from pyspark.sql.dataframe import DataFrame

from DAG import DAG
from Result import DataResult,ModelResult,SinkResult
from IBMSparkPipelineModel import IBMSparkPipelineModel
from Wrapper import PythonJavaConversions, JavaParams

class IBMSparkPipeline (Pipeline):
    """
    Base class for creating ML and ETL Pipeline. This class extends from Spark ML Pipeline.
    """
    _sc = SparkContext._active_spark_context
    _logSrcLang = "Py:"

    @keyword_only
    def __init__(self, stages=None):

        self._jLogger = self._sc._jvm.org.apache.log4j.Logger
        self.logger = self._jLogger.getLogger(self.__class__.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + "stages => " + str(
            stages) + "]"
        self.logger.info(logMsg)

        super(IBMSparkPipeline, self).__init__()

        self._jPipeline = self._sc._jvm.com.ibm.analytics.ngp.pipeline.pythonbinding.Pipelines
        self.jIBMSparkPipeline = self._jPipeline.getIBMSparkPipeline().apply()
        self._to_map = self._jPipeline.toScalaMap

        self.jTkObj = self._sc._jvm.com.ibm.analytics.ngp.util.NotebookRequestParams
        self.blueId = getpass.getpass('ibm.ax.blueid')
        self.orgId = getpass.getpass('ibm.ax.orgid')
        self.jTkObj.setBlueIdToken(self.blueId)
        self.jTkObj.setOrgId(self.orgId)

        self.tenantId = os.environ['SPARK_TENANT_ID']
        self.jTkObj.setTenant(self.tenantId)

    def __setIBMSparkPipelineObject(self, IBMSparkPipelineObj):
        self.jIBMSparkPipeline = IBMSparkPipelineObj

    def save(self, path):
        """
        Saves an :class:`IBMSparkPipeline` object to GIT.

        :param filename: filename to be saved in GIT.
        :return: None.
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " filename => " + str(
            path) + "]"
        self.logger.info(logMsg)

        if not isinstance(path, basestring):
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": path should be a " \
                    "basestring, got " + str(type(path))
            self.logger.error(logMsg)
            raise TypeError("path should be a basestring, got %s" % type(path))

        if self.getStages() is not None:
            self._to_java().save(path)
        else:
            self.jIBMSparkPipeline.save(path)


    def saveGpfs (self, path):
        """
        Saves an :class:`IBMSparkPipeline` object to GPFS.

        :param filename: filename to be saved in GPFS.
        :return: None
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " filename => " + str(
                path) + "]"
        self.logger.info(logMsg)

        if self.getStages() is not None:
            self._to_java().writer().saveGpfs(path)
        else:
            self.jIBMSparkPipeline.writer().saveGpfs(path)


    def setStages(self, stages):
        """
        Sets the stages of an ML or ETL Pipeline.

        :param stages: A list of Pipeline stages for a ML Pipeline; a :class:`pipeline.DAG.DAG`
                      object for an ETL Pipeline.
        :return: An :class:`IBMSparkPipeline` object with stages set.
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " \
                 + " stages => " + str(stages) + "]"
        self.logger.info(logMsg)

        if type(stages) is not list and not isinstance(stages,DAG):
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": Input for setStages()" \
                    "should be a list of stages for ML Pipeline,  " \
                    "DAG for ETL Pipeline, got " + str(type(stages))
            self.logger.error(logMsg)
            raise TypeError("Input for setStages() should be a list of stages for ML Pipeline, "
                            "DAG for ETL Pipeline, got %s" % type(stages))

        if isinstance(stages, DAG):
            self.jIBMSparkPipeline = self.jIBMSparkPipeline.setStages(stages._to_java())
        else:
            super(IBMSparkPipeline, self).setStages(stages)
        return self

    def _to_java_stages(self):
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: None]"
        self.logger.info(logMsg)

        gateway = SparkContext._gateway
        cls = SparkContext._jvm.org.apache.spark.ml.PipelineStage
        java_stages = gateway.new_array(cls, len(self.getStages()))
        for idx, stage in enumerate(self.getStages()):
            java_stages[idx] = PythonJavaConversions._to_java_stage(stage)

        return java_stages

    def fit(self, dataset=None):
        """
        Runs an ML or ETL Pipeline

        :param dataset: A DataFrame for a ML Pipeline; None for ETL Pipeline.
        :return: An :class:`pipeline.IBMSparkPipelineModel.IBMSparkPipelineModel` object for ML Pipeline.

                An array of :class:`pipeline.Result.Result` for ETL Pipeline.

                The Result can be either of:

                - :class:`pipeline.Result.DataResult` : If the end stage of a DAG is a Transformer.
                - :class:`pipeline.Result.ModelResult` : If the end stage of a DAG is an Estimator.
                - :class:`pipeline.Result.SinkResult`: If the end stage of a DAG is :class:`pipeline.Sink.Sink`
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " dataset => " + str(
                dataset) + "]"
        self.logger.info(logMsg)

        if dataset is not None and not isinstance(dataset,DataFrame):
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + "Input dataset for fit() " \
                    "should be a DataFrame, got " + str(type(dataset))
            self.logger.error(logMsg)
            raise TypeError("Input dataset for fit() should be a DataFrame, got %s" % type(dataset))

        if self.getStages() is not None:
            if dataset is not None:
                jIBMSparkPipelineModel = self.jIBMSparkPipeline.setStages(self._to_java_stages()).fit(dataset._jdf)
                return IBMSparkPipelineModel._from_java(jIBMSparkPipelineModel)
            else:
                self.jIBMSparkPipeline.fit()

        else:
            if dataset is None:
                sca_seq = self.jIBMSparkPipeline.fit()
                jlist = PythonJavaConversions._to_java_list(self._jPipeline, sca_seq)
                fullclassname = sca_seq.head().getClass().toString()
                classname = fullclassname.split(".")[-1]
                resultclass = getattr(sys.modules[__name__], classname)
                resultlist = resultclass._populateResult(jlist)
                return resultlist
            else:
                self.jIBMSparkPipeline.fit(dataset._jdf)

    def updateBindings (self, binddict):
        """
        Updates the input DataFrame to be passed to the DAG root node(s) after a saved ETL Pipeline
        is loaded back.

        :param binddict: A dictionary that specifies the label and input DataFrame
                         for the DAG root node.
        :return: An :class:`IBMSparkPipeline` object with bindings updated.
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " binddict => " + str(
            binddict) + "]"
        self.logger.info(logMsg)

        if type(binddict) is not dict :
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": Label and Input DataFrame for" \
                    "updateBindings() must be specified as a dictionary, got " + str(type(binddict))
            self.logger.error(logMsg)
            raise TypeError ("Label and Input DataFrame for updateBindings() must be specified as a  dictionary, "
                             "got %s" % type(binddict))

        partialJbinds = {}
        for k, v in binddict.iteritems():
            partialJbinds[k] = v._jdf

        jbinds = MapConverter().convert(partialJbinds, self._sc._gateway._gateway_client)
        sca_map = self._to_map(jbinds)
        self.jIBMSparkPipeline = self.jIBMSparkPipeline.updateBindings(sca_map)
        return self

    def getRootLabels(self) :
        """
        Returns the Root Labels for the DAG root node(s).
        """
        slabellist = self.jIBMSparkPipeline.getRootLabels()
        jlabellist = PythonJavaConversions._to_java_list(self._jPipeline, slabellist)
        plabellist=[]
        for index in range(len(jlabellist)):
            plabellist.append(str(jlabellist[index]))
        return plabellist


    @classmethod
    def load(cls, path):
        """
        Loads an :class:`IBMSparkPipeline` object from the saved path in GPFS.

        :param filename: filename which was saved in GPFS
        :return: A :class:`IBMSparkPipeline` object.
        """
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + " path => " + str(path) + "]"
        logger.info(logMsg)

        if not isinstance(path, basestring):
            logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": path should be a basestring, got " \
                     + str(type(path))
            cls.logger.error(logMsg)
            raise TypeError("path should be a basestring, got type %s" % type(path))
        else:
            _jread = PythonJavaConversions._load_java_obj(cls)
            java_obj = _jread.load(path)

        numstages = java_obj.getNonLinearStages().getStages().length()
        # If the below check returns 0, then its an ML Pipeline object. Otherwise its an ETL Pipleline object.
        if numstages == 0:
            return cls._from_java(java_obj)
        else:
            loadpipeline = IBMSparkPipeline()
            loadpipeline.__setIBMSparkPipelineObject(java_obj)
            return loadpipeline

    @classmethod
    def loadGpfs(cls, path):
        """
        Loads an :class:`IBMSparkPipeline` object from the saved path in GIT.

        :param filename: filename which was saved in GIT.
        :return: An :class:`IBMSparkPipeline` object.
        """
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + " path => " + str(path) + "]"
        logger.info(logMsg)

        # return Reader(cls).load(path)

        _jread = PythonJavaConversions._load_java_obj(cls)

        if not isinstance(path, basestring):
            raise TypeError("path should be a basestring, got type %s" % type(path))

        else:
            java_obj = _jread.reader().loadGpfs(path)

        numstages = java_obj.getNonLinearStages().getStages().length()
        # If the below check returns 0, then its an ML Pipeline object. Otherwise its an ETL Pipleline object.
        if numstages == 0:
            return cls._from_java(java_obj)
        else:
            loadpipeline = IBMSparkPipeline()
            loadpipeline.__setIBMSparkPipelineObject(java_obj)
            return loadpipeline

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java Pipeline, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        # Create a new instance of this stage.
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + " java_stage => " + str(
            java_stage) + "]"
        logger.info(logMsg)

        py_stage = cls()
        # Load information from java_stage to the instance.
        py_stages = [PythonJavaConversions._from_java_stage(s) for s in java_stage.getStages()]
        py_stage.setStages(py_stages)
        py_stage = PythonJavaConversions._resetUid(py_stage, java_stage.uid())
        return py_stage

    def _to_java(self):
        """
        Transfer this instance to a Java Pipeline.  Used for ML persistence.

        :return: Java object equivalent to this instance.
        """
        self.logger = self._jLogger.getLogger(self.__class__.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: None]"
        self.logger.info(logMsg)

        gateway = SparkContext._gateway
        cls = SparkContext._jvm.org.apache.spark.ml.PipelineStage
        java_stages = gateway.new_array(cls, len(self.getStages()))
        for idx, stage in enumerate(self.getStages()):
            java_stages[idx] = PythonJavaConversions._to_java_stage(stage)

        _java_obj = JavaParams._new_java_obj("com.ibm.analytics.ngp.pipeline.IBMSparkPipeline",
                                             self.uid)
        _java_obj.setStages(java_stages)

        return _java_obj


