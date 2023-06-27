#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging
from abc import ABCMeta, abstractmethod

from pyspark import SparkContext
from pyspark.ml.common import _py2java
from pyspark.ml.common import inherit_doc
from pyspark.ml.pipeline import Transformer, Estimator
from pyspark.ml.util import JavaMLWritable, MLReadable
from pyspark.ml.wrapper import JavaTransformer, JavaEstimator, JavaWrapper, JavaParams
from pyspark.ml.wrapper import _jvm

from mlpipelinepy.edge import DataEdge, MetaEdge
from mlpipelinepy.exception import UnsupportedNodeTypeError
from mlpipelinepy.javaproxy import TransformerProxyML, EstimatorProxyML
from mlpipelinepy.serialization import PipelineJavaMLReader

__all__ = ['MLPipeline', "MLPipelineModel", "SparkDataSources"]

logger = logging.getLogger("mlpipelinepy")


@inherit_doc
class MLPipelineBasic(JavaWrapper):
    """
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(MLPipelineBasic, self).__init__()

    def bind(self, from_node_id, to_node_id, from_edge_id=None):
        """
         Binds 'to_node_id' node with 'from_edge_id' edge of the 'from_node_id' node
        :param from_node_id: the source node
        :param to_node_id: the destination node
        :param from_edge_id: the edge of the source node
        :returns: pipeline instance
        """
        if from_edge_id is None:
            self._java_obj = self._java_obj.bind(from_node_id, to_node_id)
        else:
            self._java_obj = self._java_obj.bind(from_node_id, from_edge_id, to_node_id)
        return self

    def transform(self, spark_data_source, extra=None):
        """
        Transforms the input dataset with optional parameters.
        :param spark_data_source: SparkDataSouce object containing mapping between nodeid and dataframe
        :param extra: an optional param map that overrides embedded params.
        :returns: transformed dataset
        """
        try:
            jvm_object = _jvm()
            java_p_map = self._to_scala_param_map(extra)
            seq_m = self._java_obj.transform(spark_data_source.to_java_object(), java_p_map)
            _jDataEdgeList = jvm_object.com.ibm.analytics.wml.pipeline.pythonbinding.Helper.scalaToJavaList(seq_m)
            pDataEdgeList = []
            for _sDataEdge in _jDataEdgeList:
                pDataEdgeList.append(DataEdge(_sDataEdge))
            return pDataEdgeList
        except Exception as e:
            logger.exception(e)
            raise

    @abstractmethod
    def copy(self, extra=None):
        raise NotImplementedError()

    def node_ids(self):
        jvm_object = _jvm()
        s_nodsIds = self._java_obj.nodeIds()
        j_list = jvm_object.com.ibm.analytics.wml.pipeline.pythonbinding.Helper.scalaToJavaList(s_nodsIds)
        return map(str, j_list)

    def _to_scala_param_map(self, extra=None):
        sc = SparkContext._active_spark_context
        jvmObject = _jvm()
        p_map = jvmObject.org.apache.spark.ml.param.ParamMap.empty()
        if extra is None:
            extra = dict()
        if isinstance(extra, dict):
            for param in extra:
                try:
                    uid = param.parent
                    name = param.name
                    java_param_try = self._java_obj.nodeParam(uid, name)
                    if java_param_try.isSuccess():
                        java_param = java_param_try.get()
                        value = extra[param]
                        java_value = _py2java(sc, value)
                        p_map.put([java_param.w(java_value)])
                except Exception as e:
                    logger.exception(e)
                    raise
        else:
            raise ValueError("Params must be a param map, but got %s." % type(extra))
        return p_map

    def schema(self, spark_data_source, extra=None):
        """Returns the list of MetaEdge of this :class:`MLPipelineBasic` `.
        >>> pipeline.schema(SparkDataSources({"id": df}))
        MetaEdge (is, schema)
        """
        try:
            java_p_map = self._to_scala_param_map(extra)
            seq_meta_struc_type = self._java_obj.schema(spark_data_source.to_java_object(), java_p_map)
            jvm_object = _jvm()
            j_list = jvm_object.com.ibm.analytics.wml.pipeline.pythonbinding.Helper.scalaToJavaList(seq_meta_struc_type)
            return [MetaEdge(item) for item in j_list]
        except AttributeError as e:
            raise Exception(
                "Unable to parse datatype from schema. %s" % e)


@inherit_doc
class MLPipeline(MLPipelineBasic, JavaParams, JavaMLWritable, MLReadable):
    """
    MLPipeline wrapper class
    """

    def __init__(self, nodes=[]):
        super(MLPipeline, self).__init__()
        jvm_obj = _jvm()
        java_nodes = self._nodes_to_java_nodes(nodes, jvm_obj)
        _sNodesList = jvm_obj.com.ibm.analytics.wml.pipeline.pythonbinding.Helper.javaToScalaList(java_nodes)
        self._java_obj = jvm_obj.com.ibm.analytics.wml.pipeline.spark.MLPipeline.apply(_sNodesList)

    @classmethod
    def _to_java_node(cls, node, jvm_obj):
        """
        :param node: python estimator / transformer node
        :param jvm_obj: JVM object
        :return: java computation node required by scala MLPipeline instance
        """
        java_obj = None
        if isinstance(node, Transformer):
            if isinstance(node, JavaTransformer):
                node._transfer_params_to_java()
                java_obj = node._java_obj
            else:
                new_transformer_wrapper = TransformerProxyML(node)
                java_obj = new_transformer_wrapper.to_java_object()
            return jvm_obj.com.ibm.analytics.wml.pipeline.spark.package.transformer2PipelineNode(java_obj)
        elif isinstance(node, Estimator):
            if isinstance(node, JavaEstimator):
                node._transfer_params_to_java()
                java_obj = node._java_obj
            else:
                new_estimator_wrapper = EstimatorProxyML(node)
                java_obj = new_estimator_wrapper.to_java_object()
            return jvm_obj.com.ibm.analytics.wml.pipeline.spark.package.estimator(java_obj)
        else:
            raise UnsupportedNodeTypeError("Unsupport node type %s" % node.__class__.__name__)

    def _nodes_to_java_nodes(self, nodes, jvm_object):
        jNodes = []
        for node in nodes:
            jNodes.append(self._to_java_node(node, jvm_object))
        return jNodes

    def copy(self, extra=None):
        """
        Creates a copy of this instance.

        :param extra: extra parameters
        :returns: new instance
        """
        java_p_map = self._to_scala_param_map(extra)
        new_java_obj = self._java_obj.copy(java_p_map)
        ret_val = MLPipeline([])
        ret_val._java_obj = new_java_obj
        return ret_val

    def fit(self, spark_data_source, extra=None):
        """
        Fits a pipeline with optional parameters and produce a pipeline model

        :param params: an optional param map that overrides embedded params.
        :returns: fitted pipeline model
        """
        if extra is None:
            extra = dict()
        java_p_map = self._to_scala_param_map(extra)

        if isinstance(extra, dict):
            try:
                return MLPipelineModel(self, self._java_obj.fit(spark_data_source.to_java_object(), java_p_map))
            except Exception as e:
                logger.exception(e)
                raise
        else:
            raise ValueError("Params must be a param map, but got %s." % type(extra))

    def _to_java(self):
        return self._java_obj

    @classmethod
    def _from_java(cls, java_obj):
        """
        Given a Java Pipeline, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        # Create a new instance of this stage.
        py_obj = cls()
        py_obj._java_obj = java_obj
        return py_obj

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return PipelineJavaMLReader(cls, "com.ibm.analytics.wml.pipeline.spark.MLPipeline")


class MLPipelineModel(MLPipelineBasic, JavaMLWritable, MLReadable):
    """
    PipelineModel wrapper class
    """

    def __init__(self, parent_pipeline=None, java_model=None):
        super(MLPipelineModel, self).__init__()
        self._java_obj = java_model
        self._parent = parent_pipeline

    def copy(self, extra=None):
        java_p_map = self._to_scala_param_map(extra)
        new_java_obj = self._java_obj.copy(java_p_map)
        return MLPipelineModel(new_java_obj)

    @property
    def parent(self):
        return self._parent

    def _to_java(self):
        return self._java_obj

    @classmethod
    def _from_java(cls, java_obj):
        """
        Given a Java Pipeline, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        # Create a new instance of this stage.
        py_obj = cls()
        py_obj._java_obj = java_obj
        if java_obj is None and java_obj.parentPipeline().isDefined():
            py_parent = MLPipeline()
            py_parent._java_obj = java_obj.parentPipeline().get()
            py_obj._parent = py_parent
        return py_obj

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return PipelineJavaMLReader(cls, "com.ibm.analytics.wml.pipeline.spark.MLPipelineModel")


class SparkDataSources(object):
    """
    SparkDataSources wrapper class
    """

    def __init__(self, idDfMapping=dict()):
        super(SparkDataSources, self).__init__()
        self._idDfMapping = idDfMapping

    def to_java_object(self):
        jvm_object = _jvm()
        elems = []
        for key, value in list(self._idDfMapping.items()):
            elems.append(jvm_object.scala.Tuple2(key, value._jdf))

        scalaList = jvm_object.com.ibm.analytics.wml.pipeline.pythonbinding.Helper.javaToScalaList(elems)
        return jvm_object.com.ibm.analytics.wml.pipeline.spark.SparkDataSources(
            jvm_object.com.ibm.analytics.wml.pipeline.pythonbinding.Helper.scalaListOfPairsToMap(scalaList))
