#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .artifact import Artifact


class ModelArtifact(Artifact):
    """
    Class representing model artifact.

    :param str uid: optional, uid which indicate that artifact already exists in repository service
    :param str name: optional, name of artifact
    :param MetaProps meta_props: optional, props used by other services
    """
    def __init__(self, uid, name, meta_props):
        super(ModelArtifact, self).__init__(uid, name, meta_props)

    def pipeline_artifact(self):
        """
        Returns pipeline artifact from which this model artifact was created.

        :rtype: ModelArtifact
        """
        try:
            return self._pipeline_artifact
        except:
            self._pipeline_artifact = self.client.pipelines.version_from_url(self._training_definition_version_url)
            return self._pipeline_artifact
