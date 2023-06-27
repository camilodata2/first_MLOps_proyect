#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .artifact import Artifact


class WmlLibrariesArtifact(Artifact):
    """
    Class representing WmlLibraryArtifact artifact.

    :param str uid: optional, uid which indicate that artifact already exists in repository service
    :param str name: optional, name of artifact
    :param MetaProps meta_props: optional, props used by other services
    """
    def __init__(self, uid, name, meta_props):
        super(WmlLibrariesArtifact, self).__init__(uid, None, meta_props)
