#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames, MetaProps, WmlLibrariesArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.libraries_artifact_reader import LibrariesArtifactReader
from ibm_watson_machine_learning.libs.repo.util.exceptions import MetaPropMissingError

class LibrariesArtifact(WmlLibrariesArtifact):
    """
    Class representing LibrariesArtifact artifact.

    :param str uid: optional, uid which indicate that artifact already exists in repository service
    :param str name: optional, name of artifact
    :param str description: optional, description of artifact
    :param str version: version of libraries
    :param str  platform of libraries
    """
    def __init__(self, library=None,  uid=None, name=None, meta_props=MetaProps({})):
        super(LibrariesArtifact, self).__init__(uid, name, meta_props)
        self.library = library
        self.uid = uid
        self.meta_props = meta_props
        if meta_props.prop(MetaNames.LIBRARIES.PATCH_INPUT) is None:
            if meta_props.prop(MetaNames.LIBRARIES.NAME) is None:
                raise MetaPropMissingError('Value specified for "meta_props" does not contain value for ''"MetaNames.LIBRARIES.NAME"')
            self.name = meta_props.get()[MetaNames.LIBRARIES.NAME]

            if meta_props.prop(MetaNames.LIBRARIES.VERSION) is None:
                raise MetaPropMissingError('Value specified for "meta_props" does not contain value for' '"MetaNames.LIBRARIES.VERSION"')

            if meta_props.prop(MetaNames.LIBRARIES.PLATFORM) is None:
                raise MetaPropMissingError('Value specified for "meta_props" does not contain value for '
                                       '"MetaNames.LIBRIES.PLATFORM"')


        #if meta_props.prop(MetaNames.LIBRARIES.NAME) is not None and not isinstance(meta_props.prop(MetaNames.LIBRARIES.NAME), str):
        #    raise ValueError('Invalid type for MetaNames.LIBRARIES.NAME: {}'.
        #                     format(meta_props.prop(MetaNames.LIBRARIES.NAME).__class__.__name__))
#
#        if meta_props.prop(MetaNames.LIBRARIES.VERSION) is not None and not isinstance(meta_props.prop(MetaNames.LIBRARIES.VERSION), str):
#            raise ValueError('Invalid type for MetaNames.LIBRARIES.VERSION: {}'.
#                             format(meta_props.prop(MetaNames.LIBRARIES.VERSION).__class__.__name__))
#

    def reader(self):
        """
        Returns reader used for getting library content.

        :return: reader for LibrariesArtifact.library
        :rtype: LibrariesArtifactReader
        """
        try:
            return self._reader
        except:
            self._reader = LibrariesArtifactReader(self.library)
            return self._reader

    def _copy(self, uid=None, meta_props=None):
        if uid is None:
            uid = self.uid

        if meta_props is None:
            meta_props = self.meta

        return LibrariesArtifact(
            self.library,
            uid=uid,
            meta_props=meta_props
        )
