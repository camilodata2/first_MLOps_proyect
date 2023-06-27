#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepository import ModelArtifact


class ScikitModelArtifact(ModelArtifact):
    """
    Class representing model artifact.

    :param str uid: optional, uid which indicate that artifact already exists in repository service
    :param str name: optional, name of artifact
    :param MetaProps meta_props: optional, props used by other services
    """
    def __init__(self, uid, name, meta_props):
        super(ScikitModelArtifact, self).__init__(uid, name, meta_props)

    def pipeline_artifact(self):
        """
        Returns None. pipeline artifact for scikit model has not been implemented.

        :rtype: ModelArtifact
        """
        pass
