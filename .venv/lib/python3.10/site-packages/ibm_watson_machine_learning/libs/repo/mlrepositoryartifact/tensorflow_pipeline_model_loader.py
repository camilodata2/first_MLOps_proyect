#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.tensorflow_artifact_loader import TensorflowArtifactLoader
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

class TensorflowPipelineModelLoader(TensorflowArtifactLoader):
    """
        Returns Tensorflow Runtime  instance associated with this model artifact.

        :return: TensorflowArtifactRunTime instance
        :rtype: TensorflowRuntimeArtifact
        """
    def load_model(self,session=None,tags=None):
        return(self.model_instance(session,tags))



    def model_instance(self,session=None,tags=None):
        lib_checker.check_lib(TENSORFLOW)
        import tensorflow as tf
        if session is None:
            if '2.1.0' in tf.__version__:
                session = tf.compat.v1.Session(graph=tf.compat.v1.Graph())
            else:
                session = tf.Session(graph=tf.Graph())
        elif session is not None:
            if '2.1.0' in tf.__version__ and not isinstance(session, tf.compat.v1.Session):
                raise TypeError("sess should be of type : %s" % tf.compat.v1.Session)
            elif not isinstance(session,tf.Session):
                raise TypeError("sess should be of type : %s" % tf.Session)

        if tags is None:
            if '2.1.0' in tf.__version__:
                tags = [tf.compat.v1.saved_model.tag_constants.SERVING]
            else:
                tags = [tf.saved_model.tag_constants.SERVING]
        elif tags is not None:
            if not isinstance(tags,list):
                raise TypeError("tags should be of type : %s" % list)


        return self.load(session,tags)
