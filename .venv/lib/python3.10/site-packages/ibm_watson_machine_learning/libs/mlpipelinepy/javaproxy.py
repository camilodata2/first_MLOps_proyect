#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc
import logging
from abc import abstractmethod

from pyspark import SparkContext, SQLContext
from pyspark.ml.common import inherit_doc
from pyspark.ml.param import Param
from pyspark.ml.util import Identifiable
from pyspark.ml.wrapper import _jvm
from pyspark.sql import DataFrame

from mlpipelinepy.utils import estimatorCopy
from pyspark.sql.types import _parse_datatype_json_string

logger = logging.getLogger("mlpipelinepy")


class JavaProxyMLOp(object):
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def uid(self):
        return "dummyId"

    def toString(self):
        return str(self)

    @abstractmethod
    def to_java_object(self):
        return None


@inherit_doc
class EstimatorProxyML(JavaProxyMLOp):
    '''
    Internal class to keep a Java mapped estimator of Python instance
    '''

    def __init__(self, estimator):
        '''
        Constructor for EstimatorProxy object
        '''
        self._estimator = estimator

    def uid(self):
        return self._estimator.uid

    def fit(self, jdataFrame):
        sc = SparkContext.getOrCreate()
        ctx = SQLContext.getOrCreate(sc)
        try:
            dataFrame = DataFrame(jdataFrame, ctx)
            pModel = self._estimator.fit(dataFrame)
            return ModelProxyML(pModel)
        except Exception as e:
            logger.exception(e)
            raise

    def __str__(self):
        return "EstimatorProxy uid: " + self.uid()

    def transformSchema(self, jsonSchema):
        #self._estimator.transformSchema(_parse_datatype_json_string(jsonSchema))
        logger.debug("Transform Schema call")
        return jsonSchema

    def copy(self, extra=None):
        try:
            jvmObject = _jvm()
            extraNew = dict()
            if extra is not None:
                _jHelperObj = jvmObject.com.ibm.analytics.wml.pipeline.pythonbinding.Helper
                scalaListParamPair = extra.toSeq().toList()
                jListParamPair = _jHelperObj.scalaToJavaList(scalaListParamPair)
                for pair in jListParamPair:
                    try:
                        param = pair.param()
                        jParamName = param.name()
                        jParamUID = param.parent()
                        jParamDoc = param.doc()
                        idfDummy = Identifiable()
                        idfDummy.uid = str(jParamUID)
                        pParam = Param(idfDummy, str(jParamName), str(jParamDoc))
                        jValue = pair.value()
                        extraNew[pParam] = jValue
                    except Exception as e:
                        logger.exception(e)
                        raise
            return EstimatorProxyML(estimatorCopy(self._estimator, jvmObject, extraNew))
        except Exception as ee:
            logger.exception(ee)
            raise

    def to_java_object(self):
        jvmObject = _jvm()
        newEstimatorProxyObject = jvmObject.com.ibm.analytics.wml.pipeline.pythonbinding.EstimatorProxy
        return newEstimatorProxyObject(self)

    class Java:
        implements = ['com.ibm.analytics.wml.pipeline.pythonbinding.PythonEstimator']


class TransformerProxyML(JavaProxyMLOp):
    '''
    Internal class to keep a Java mapped transformer of Python instance
    '''

    def __init__(self, transformer):
        '''
        Constructor for TransformerProxy object
        '''
        self._transformer = transformer

    def uid(self):
        return self._transformer.uid

    def transform(self, jDataFrame):
        logger.debug("Transform DF")
        sc = SparkContext.getOrCreate()
        ctx = SQLContext.getOrCreate(sc)
        try:
            pDataFrame = DataFrame(jDataFrame, ctx)
            pTransfDataFrame = self._transformer.transform(pDataFrame)
            return pTransfDataFrame._jdf
        except Exception as e:
            logger.exception(e)
            raise

    def transformSchema(self, jsonInputSchema):
        logger.debug("Transform Schema call")
        return jsonInputSchema

    def __str__(self):
        return "TransformerProxy uid: " + self.uid()

    def toString(self):
        return str(self)

    def format(strrr, obj1, obj2):
        print(str(strrr))
        return "format method"

    def to_java_object(self):
        try:
            jvmObject = _jvm()
            newTransformerProxyObject = jvmObject.com.ibm.analytics.wml.pipeline.pythonbinding.TransformerProxy
            return newTransformerProxyObject(self)
        except Exception as e:
            logger.exception(e)
            raise

    class Java:
        implements = ['com.ibm.analytics.wml.pipeline.pythonbinding.PythonTransformer']


class ModelProxyML(TransformerProxyML):
    def __init__(self, model):
        '''
        Constructor for TransformerProxy object
        '''
        super(ModelProxyML, self).__init__(model)

    def __str__(self):
        return "ModelProxy uid: " + self.uid()

    def to_java_object(self):
        jvmObject = _jvm()
        newModelProxyObject = jvmObject.com.ibm.analytics.wml.pipeline.pythonbinding.ModelProxy
        return newModelProxyObject(self)

    class Java:
        implements = ['com.ibm.analytics.wml.pipeline.pythonbinding.PythonModel']
