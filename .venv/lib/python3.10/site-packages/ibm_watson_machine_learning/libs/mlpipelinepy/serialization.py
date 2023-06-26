#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from pyspark.ml.common import inherit_doc
from pyspark.ml.util import MLReadable, JavaMLReader, _jvm


class PipelineJavaMLReader(JavaMLReader):
    """
    MLPipeline JavaML Reader
    """

    def __init__(self, owner_clazz, class_name):
        self._className = class_name
        super(PipelineJavaMLReader, self).__init__(owner_clazz)

    def _java_loader_class(self, clazz):
        return self._className

    def _load_java_obj(self, clazz):
        """Load the peer Java object of the ML instance."""
        java_class = self._java_loader_class(clazz)
        java_obj = _jvm()
        for name in java_class.split("."):
            java_obj = getattr(java_obj, name)
        return java_obj





