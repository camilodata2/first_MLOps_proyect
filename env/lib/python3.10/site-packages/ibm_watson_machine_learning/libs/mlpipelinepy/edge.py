#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging

from pyspark import SparkContext, SQLContext
from pyspark.sql import DataFrame
from pyspark.sql.types import _parse_datatype_json_string

__all__ = ['DataEdge', "MetaEdge"]

logger = logging.getLogger("mlpipelinepy")


class DataEdge(object):
    """
    DataEdge information
    """

    def __init__(self, s_tuple_id_df):
        try:
            self._java_obj = s_tuple_id_df
            self._id = s_tuple_id_df._1()
            self.j_Data_Frame = s_tuple_id_df._2()
        except Exception as e:
            logger.exception(e)
            raise

    @property
    def id(self):
        return self._id

    @property
    def data_frame(self):
        sc = SparkContext.getOrCreate()
        ctx = SQLContext.getOrCreate(sc)
        return DataFrame(self.j_Data_Frame, ctx)


class MetaEdge(object):
    """
    MetaEdge information
    """

    def __init__(self, s_meta_edge):
        try:
            self._id = str(s_meta_edge._1())
            self._fromNode = None
            self._schema = _parse_datatype_json_string(s_meta_edge._2().json())
        except Exception as e:
            logger.exception(e)
            raise

    @property
    def id(self):
        return self._id

    @property
    def schema(self):
        return self._schema

    def __str__(self):
        return "Id: " + self._id + " Schema: " + str(self._schema)
