#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import copy
import inspect
from abc import ABCMeta
from py4j.java_collections import JavaArray, JavaList
from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaObject
from py4j.protocol import Py4JJavaError
from pyspark import RDD, SparkContext
from pyspark.ml.param import Params
from pyspark.ml.wrapper import JavaModel
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from pyspark.sql import DataFrame, SQLContext



class PythonJavaConversions(object):

    _sc = SparkContext._active_spark_context
    _logSrcLang = "Py:"

    @classmethod
    def _resetUid(cls, stage, uid):

        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + "stage => " + str(
                stage) + "\" | uid => \"" + str(uid) + "\"]"
        logger.info(logMsg)

        newUid = unicode(uid)
        stage.uid = newUid
        newDefaultParamMap = dict()
        newParamMap = dict()
        for param in stage.params:
            newParam = copy.copy(param)
            newParam.parent = newUid
            if param in stage._defaultParamMap:
                newDefaultParamMap[newParam] = stage._defaultParamMap[param]
            if param in stage._paramMap:
                newParamMap[newParam] = stage._paramMap[param]
            param.parent = newUid
        stage._defaultParamMap = newDefaultParamMap
        stage._paramMap = newParamMap
        return stage

    @classmethod
    def _to_java_stage(cls, stage):
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + "stage => " + str(stage) + "]"
        logger.info(logMsg)

        paramMap = stage.extractParamMap()
        for param in stage.params:
            if param in paramMap:
                pair = cls._make_java_param_pair(stage, param, paramMap[param])
                stage._java_obj.set(pair)
        return stage._java_obj

    @classmethod
    def _py2java(cls, sc, obj):
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + "sc => " + str(
                sc) + " | obj => " + str(obj) + "]"
        logger.info(logMsg)

        """ Convert Python object into Java """
        if isinstance(obj, RDD):
            obj = cls._to_java_object_rdd(obj)
        elif isinstance(obj, DataFrame):
            obj = obj._jdf
        elif isinstance(obj, SparkContext):
            obj = obj._jsc
        elif isinstance(obj, list):
            obj = ListConverter().convert([cls._py2java(sc, x) for x in obj],
                                          sc._gateway._gateway_client)
        elif isinstance(obj, JavaObject):
            pass
        elif isinstance(obj, (int, long, float, bool, bytes, unicode)):
            pass
        else:
            data = bytearray(PickleSerializer().dumps(obj))
            obj = sc._jvm.SerDe.loads(data)
        return obj

    """
    Krishna: _java2py is called only by _call_java(). But _call_java is not used anywhere. Hence commenting it for now

    @classmethod
    def _java2py(sc, r, encoding="bytes"):
        _logger = _jLogger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = _logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + "sc => " + str(sc) + " | r => " + str(
                r) + " | encoding => " + str(encoding) + "]"
        _logger.info(logMsg)

        if isinstance(r, JavaObject):
            clsName = r.getClass().getSimpleName()
            # convert RDD into JavaRDD
            if clsName != 'JavaRDD' and clsName.endswith("RDD"):
                r = r.toJavaRDD()
                clsName = 'JavaRDD'

            if clsName == 'JavaRDD':
                jrdd = sc._jvm.SerDe.javaToPython(r)
                return RDD(jrdd, sc)

            if clsName == 'Dataset':
                return DataFrame(r, SQLContext.getOrCreate(sc))

            if clsName in _picklable_classes:
                r = sc._jvm.SerDe.dumps(r)
            elif isinstance(r, (JavaArray, JavaList)):
                try:
                    r = sc._jvm.SerDe.dumps(r)
                except Py4JJavaError:
                    pass  # not pickable

        if isinstance(r, (bytearray, bytes)):
            r = PickleSerializer().loads(bytes(r), encoding=encoding)
            return r
    """

    @staticmethod
    def _to_java_object_rdd(rdd):
        """ Return an JavaRDD of Object by unpickling

        It will convert each Python object into Java object by Pyrolite, whenever the
        RDD is serialized in batch or not.
        """
        rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
        return rdd.ctx._jvm.SerDe.pythonToJava(rdd._jrdd, True)

    @classmethod
    def _make_java_param_pair(cls, stage, param, value):
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + "stage => " + str(
                stage) + " | param => " + str(param) + " | encoding => " + str(value) + "]"
        logger.info(logMsg)

        param = stage._resolveParam(param)
        java_param = stage._java_obj.getParam(param.name)
        java_value = cls._py2java(cls._sc, value)
        return java_param.w(java_value)

    @classmethod
    def _from_java_stage(cls, java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.

        Meta-algorithms such as Pipeline should override this method as a classmethod.
        """

        def __get_class(clazz):
            """
            Loads Python class from its name.
            """
            parts = clazz.split('.')
            module = ".".join(parts[:-1])
            m = __import__(module)
            for comp in parts[1:]:
                m = getattr(m, comp)
            return m

        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + "java_stage => " + str(
                java_stage) + "]"
        logger.info(logMsg)

        stage_name = java_stage.getClass().getName().replace("org.apache.spark", "pyspark")
        # Generate a default new instance from the stage_name class.
        py_type = __get_class(stage_name)

        if issubclass(py_type, JavaModel):
            py_stage = py_type(java_stage)
        else:
            py_stage = py_type()

        # Load information from java_stage to the instance.
        py_stage._java_obj = java_stage
        py_stage = cls._resetUid(py_stage, java_stage.uid())
        py_stage._transfer_params_from_java()
        return py_stage

    @staticmethod
    def _to_java_list(java_obj, scala_seq):
        return java_obj.toJavaList(scala_seq)

    @classmethod
    def _load_java_obj(cls, clazz):
        """
        >>> Load the peer Java object of the ML instance.
        """
        # java_class = cls._java_loader_class(clazz)

        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + "clazz => " + str(clazz) + "]"
        logger.info(logMsg)

        if clazz.__name__ == "IBMSparkPipeline":
            java_class = "com.ibm.analytics.ngp.pipeline.IBMSparkPipeline"

        else:
            java_class = "org.apache.spark.ml.IBMSparkPipelineModel"

        java_obj = cls._sc._jvm
        for name in java_class.split("."):
            java_obj = getattr(java_obj, name)

        return java_obj


class JavaWrapper(object):
    """
    Wrapper class for a Java companion object
    """
    _sc = SparkContext._active_spark_context
    _logSrcLang = "Py:"

    def __init__(self, java_obj=None):
        super(JavaWrapper, self).__init__()
        self._java_obj = java_obj


    """
    Krishna: _create_from_java_class is not used anywhere. Hence commenting it for now
    @classmethod
    def _create_from_java_class(cls, java_class, *args):

        # Construct this object from given Java classname and arguments

        _logger = _jLogger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = _logSrcLang + cls.__name__ + ":" + methodname + ": [Params: " + "java_class => " + str(
                java_class) + " | args => "
        for curArg in args[:-1]:
            logMsg = logMsg + str(curArg) + ","
        logMsg = logMsg + args[-1] + "]"
        _logger.info(logMsg)

        java_obj = JavaWrapper._new_java_obj(java_class, *args)
        return cls(java_obj)
    """

    """
    Krishna: _call_java is not used anywhere. Hence commenting it for now
    def _call_java(self, name, *args):
        m = getattr(self._java_obj, name)
        sc = SparkContext._active_spark_context
        java_args = [PythonJavaConversions._py2java(sc, arg) for arg in args]
        return PythonJavaConversions._java2py(sc, m(*java_args))
    """

    @classmethod
    def _new_java_obj(cls,java_class, *args):
        """
        Returns a new Java object.
        """
        logger = cls._sc._jvm.org.apache.log4j.Logger.getLogger(cls.__name__)
        methodname = str(inspect.stack()[0][3])
        logMsg = cls._logSrcLang + 'JavaWrapper' + ":" + methodname + ": [Params: " + "java_class => " + str(
                java_class) + " | args => "
        for curArg in args[:-1]:
            logMsg = logMsg + str(curArg) + ","
        logMsg = logMsg + str(args[-1]) + "]"
        logger.info(logMsg)

        sc = SparkContext._active_spark_context
        java_obj = SparkContext._active_spark_context._jvm
        for name in java_class.split("."):
            java_obj = getattr(java_obj, name)
        java_args = [PythonJavaConversions._py2java(sc, arg) for arg in args]
        return java_obj(*java_args)


class JavaParams(JavaWrapper, Params):
    """
    Utility class to help create wrapper classes from Java/Scala
    implementations of pipeline components.
    """
    #: The param values in the Java object should be
    #: synced with the Python wrapper in fit/transform/evaluate/copy.

    __metaclass__ = ABCMeta
