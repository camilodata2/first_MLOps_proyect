#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames, MetaProps, WmlRuntimesArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.runtimes_artifact_reader import RuntimesArtifactReader
from ibm_watson_machine_learning.libs.repo.util.exceptions import MetaPropMissingError


class RuntimesArtifact(WmlRuntimesArtifact):
    """
    Class representing RuntimesArtifact artifact.

    :param str uid: optional, uid which indicate that artifact already exists in repository service
    :param str name: optional, name of artifact
    :param str description: optional, description of artifact in meta
    :param str  platform of libraries in meta
    :param str  custom_libraries of libraries in meta
    """
    def __init__(self, uid=None, runtimespec_path=None, name=None, meta_props=MetaProps({})):
        super(RuntimesArtifact, self).__init__(uid, name, meta_props)
        self.uid = uid
        self.runtimespec_path = runtimespec_path
        self.meta_props = meta_props

        if meta_props.prop(MetaNames.RUNTIMES.PATCH_INPUT) is None:
            if meta_props.prop(MetaNames.RUNTIMES.NAME) is None:
                raise MetaPropMissingError('Value specified for "meta_props" does not contain value for ''"MetaNames.RUNTIMES.NAME"')
            self.name = meta_props.get()[MetaNames.RUNTIMES.NAME]

            if meta_props.prop(MetaNames.RUNTIMES.PLATFORM) is None:
                raise MetaPropMissingError('Value specified for "meta_props" does not contain value for '
                                           '"MetaNames.RUNTIMES.PLATFORM"')

    def reader(self):
        """
        Returns reader used for getting runtimes content.

        :return: reader for RuntimesArtifact.runtimespec_path
        :rtype: RuntimesArtifactReader
        """
        try:
            return self._reader
        except:
            self._reader = RuntimesArtifactReader(self.runtimespec_path)
            return self._reader

    def _copy(self, uid=None, name=None, meta_props=None):
        if uid is None:
            uid = self.uid

        if meta_props is None:
            meta_props = self.meta

        return RuntimesArtifact(
            uid=uid,
            name=name,
            meta_props=meta_props
        )
