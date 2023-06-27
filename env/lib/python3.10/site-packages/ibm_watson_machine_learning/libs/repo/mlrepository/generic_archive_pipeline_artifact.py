#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepository import PipelineArtifact

class GenericArchivePipelineArtifact(PipelineArtifact):
    """
    Class representing archive pipeline artifact
    """
    def __init__(self, uid, name, meta_props):
        """
        Constructor for Generic archive pipeline artifact
        :param uid: unique id for Generic archive pipeline artifact
        :param name: name of the pipeline
        :param metaprops: properties of the pipeline and pipeline artifact
        """
        super(GenericArchivePipelineArtifact, self).__init__(uid, name, meta_props)