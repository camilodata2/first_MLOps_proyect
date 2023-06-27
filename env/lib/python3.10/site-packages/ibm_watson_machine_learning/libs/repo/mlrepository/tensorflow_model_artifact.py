#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepository import ModelArtifact

class TensorflowModelArtifact(ModelArtifact):
    """
    Class representing Tensorflow model artifact
    """
    def __init__(self, uid, name, meta_props):
        """
        Constructor for TensorFlow model artifact
        :param uid: unique id for TensorFlow model artifact
        :param name: name of the model
        :param metaprops: properties of the model and model artifact
        """
        super(TensorflowModelArtifact, self).__init__(uid, name, meta_props)