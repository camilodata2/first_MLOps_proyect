#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import inspect
import sys

from py4j.java_collections import MapConverter, ListConverter
from pyspark import SparkContext

from Wrapper import PythonJavaConversions

class DAG:
    """
    The base class for creating a DAG object.

    A DAG is a graph data structure in which each stage is an ETL transformer which
    transforms one dataframe into another.
    The start node of a DAG is referred to as root and the end nodes are referred to as leaves.
    A DAG can have one or more stages in the root as well as in the leaves.
    """
    _sc = SparkContext._active_spark_context
    _logSrcLang = "Py:"

    def __init__(self):
        self._jLogger = self._sc._jvm.org.apache.log4j.Logger
        self.logger = self._jLogger.getLogger(self.__class__.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: None]"
        self.logger.info(logMsg)

        self._jDAGObject = None
        self._jPipelineWrapper = self._sc._jvm.com.ibm.analytics.ngp.pipeline.pythonbinding.Pipelines
        self._to_list = self._jPipelineWrapper.toList
        self._to_tuple = self._jPipelineWrapper.toTuple2
        self._to_map = self._jPipelineWrapper.toScalaMap

    def _to_java_stage(self, stage):
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " stage => " + str(
                stage) + " ]"
        self.logger.info(logMsg)

        if hasattr(stage, "_to_java"):
            jstage = stage._to_java()
        else:
            jstage = PythonJavaConversions._to_java_stage(stage)
        return jstage

    def __isvalidtype(self, stage, label=None):
        methodname = str(inspect.stack()[1][3])

        if not (hasattr(stage, 'transform') or hasattr(stage, 'fit') or
                hasattr(stage, 'run')):
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": Stage assigned to a DAG " \
                    "should be a Transformer, Estimator or a Source, got " + str(type(stage)) + \
                    ". Refer the API documentation for syntax."
            self.logger.error(logMsg)
            raise TypeError("Stage assigned to a DAG should be a Transformer, Estimator or a Source, "
                            "got %s" % type(stage))

        if label and not isinstance(label, basestring):
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": Label assigned to a DAG " \
                    "should be a basestring, got " + str(type(label)) + ". Refer the API documentation for syntax."
            self.logger.error(logMsg)
            raise TypeError("Label assigned to a DAG should be a basestring, got %s" % type(label))

        else:
            return True

    def start(self, stages):
        """
        Associates stage(s) to the start of the DAG.

        :param stages: stages can be

        - A single transformer

        - List of transformers

        - A single transformer and a label specified as a tuple. The label has to be specified
          only when the Pipeline needs to be saved and loaded later.

        - List of transformers with labels, where each element of the list is a tuple consisting of
          transformer and a label. The label has to be specified only when the Pipeline needs to be
          saved and loaded later.

        :return: A :class:`DAG` object.
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " stages => " + str(
                stages) + " ]"
        self.logger.info(logMsg)

        if type(stages) is list:
            pjstages = []
            isTuple = 0
            for element in stages:
                if type(element) is tuple :
                    isTuple=1
                    pstage, label = element
                    if self.__isvalidtype(pstage,label):
                        jstage = self._to_java_stage(pstage)
                        pjstages.append((jstage, label))
                else:
                    pstage = element
                    if self.__isvalidtype(pstage):
                        jstage = self._to_java_stage(pstage)
                        pjstages.append(jstage)
            if isTuple :
                jlist = ListConverter().convert(pjstages, self._sc._gateway._gateway_client)
                stuple = self._jPipelineWrapper.toScalaTupleFromJavaList(jlist)
                self._jDAGObject = self._jPipelineWrapper.getDAG().start(stuple,
                                                                         self._jPipelineWrapper.getDummyImplicit())
            else:
                jstages = self._to_list(ListConverter().convert(pjstages, self._sc._gateway._gateway_client))
                self._jDAGObject = self._jPipelineWrapper.getDAG().start(jstages)

        elif type(stages) is tuple:
            pstage, label = stages
            pjstages=[]
            if self.__isvalidtype(pstage,label):
                jstage = self._to_java_stage(pstage)
                pjstages.append(jstage)
                pjstages.append(label)
                jlist = ListConverter().convert(pjstages, self._sc._gateway._gateway_client)
                stuple = self._jPipelineWrapper.toTuple2(jlist)
                self._jDAGObject = self._jPipelineWrapper.getDAG().start(stuple)

        else:
            if self.__isvalidtype(stages):
                jstage = self._to_java_stage(stages)
                self._jDAGObject = self._jPipelineWrapper.getDAG().start(jstage)

        return self

    def append(self, stage, childstage):
        """
        Appends a child stage to an existing stage of a DAG node. The Transformer
        specified by `childstage` is applied on the DataFrame after `stage`.

        :param stage: An existing Transformer stage of a DAG
        :param childstage: A new Transformer stage to be appended to `stage`
        :return: A :class:`DAG` object with `childstage` appended to `stage`
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " stage => " + str(
                stage) + " childstage => " + str(childstage) + " ]"
        self.logger.info(logMsg)

        if self.__isvalidtype(stage) and self.__isvalidtype(childstage):
            self._jDAGObject = self._jDAGObject.append(self._to_java_stage(stage), self._to_java_stage(childstage))
            return self

    def merge(self, parentstagetuple, stage):
        """
        Merges two Transformer stage outputs of a DAG to form a single stage. The DataFrames
        from the Transformer stages are passed to `stage` Transformer, which produces a single
        DataFrame as output.

        :param parentstagetuple: A tuple specifying the Transformer stages of the DAG to be merged.
        :param stage: A Transformer stage, which takes the outputs from two DataFrames and merges.
        :return: A :class:`DAG` object
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " parentstagetuple => " \
                "" + str(parentstagetuple) +  " stage => " + str(stage) + " ]"
        self.logger.info(logMsg)

        if type(parentstagetuple) is not tuple:
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": Parent stages for merge " \
                    "should be a tuple, got " + str(type(parentstagetuple)) + ". Refer the API " \
                    "documentation for syntax."
            self.logger.error(logMsg)
            raise TypeError("Parent stages for merge should be a tuple, got %s" % type(parentstagetuple))

        parentstage1, parentstage2 = parentstagetuple
        if self.__isvalidtype(stage):
            jstage = self._to_java_stage(stage)

        if self.__isvalidtype(parentstage1) and self.__isvalidtype(parentstage2):
            jparentstage1 = self._to_java_stage(parentstage1)
            jparentstage2 = self._to_java_stage(parentstage2)
            # Convert the tuple parentstages2 to list ,since java does not have tuple
            pjparentStages = [jparentstage1, jparentstage2]

        # call _to_tuple to convert java list to scala tuple2
        jparentstages = self._to_tuple(
                ListConverter().convert(pjparentStages, self._sc._gateway._gateway_client))
        self._jDAGObject = self._jDAGObject.merge(jparentstages, jstage)
        return self

    def setBindings(self, binddict):
        """
        Sets the input DataFrame(s) for the root stage(s) of the DAG.
        
        :param binddict: A dictionary specifying the input DataFrame(s) to root stage(s) mapping.
        :return: A :class:`DAG` object
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " binddict => " + str(
                binddict) + " ]"
        self.logger.info(logMsg)

        if type(binddict) is not dict:
            logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": Stage and Input " \
                    "DataFrame for a DAG should be specified as a dictionary, got " \
                    + str(type(binddict))
            self.logger.error(logMsg)
            raise TypeError("Stage and Input DataFrame for a DAG should be specifed as a dictionary, got %s"
                            % type(binddict))
        
        partialJbinds = {}
        for k, v in binddict.iteritems():
            partialJbinds[self._to_java_stage(k)] = v._jdf

        jbinds = MapConverter().convert(partialJbinds, self._sc._gateway._gateway_client)
        sca_map = self._to_map(jbinds)
        self._jDAGObject = self._jDAGObject.setBindings(sca_map)
        return self

    def fork(self, stage, childstages):
        """
        Forks the DAG to form child stages after stage. The `childstages` here specifies a
        list of Transformers.  If the list has more than one Transformer, each Transformer
        specified in `childstages` is applied separately on the dataframe after `stage` resulting
        in more than one execution paths.

        :param stage: An existing Transformer stage of a DAG
        :param childstages: A list of Transformer stages to be forked after stage.
        :return: A :class:`DAG` object with `childstages` forked after `stage`
        """
        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: " + " stage => " + str(
                stage) + " childstages => " + str(childstages) + " ]"
        self.logger.info(logMsg)

        if self.__isvalidtype(stage):
            pjChildStages = []
            for pstage in childstages:
                if self.__isvalidtype(pstage):
                    pjChildStages.append(self._to_java_stage(pstage))

        jChildStages = self._to_list(ListConverter().convert(pjChildStages, self._sc._gateway._gateway_client))
        self._jDAGObject = self._jDAGObject.fork(self._to_java_stage(stage), jChildStages)

        return self

    def _to_java(self):

        methodname = str(inspect.stack()[0][3])
        logMsg = self._logSrcLang + self.__class__.__name__ + ":" + methodname + ": [Params: None]"
        self.logger.info(logMsg)

        return self._jDAGObject
