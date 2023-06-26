#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging
import json
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames, MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact import LibrariesArtifact,  LibrariesArtifactReader, LibrariesLoader

logger = logging.getLogger('WmlLibrariesAdapter')


class WmlLibrariesAdapter(object):
    """
    Adapter creating libraries artifact using output from service.
    """

    def __init__(self, library_output, client):

        self.libraries_output = library_output
        self.client = client
        self.libraries_entity = library_output.entity
        self.libraries_metadata = library_output.metadata
        self.uid = library_output.metadata.guid
        self.name = None

    def artifact(self):
        libraries_artifact_builder = type(
            "LibrariesArtifact",
            (LibrariesLoader, LibrariesArtifact, LibrariesArtifactReader, object),
            {}
        )

        prop_map = {
            MetaNames.CREATION_TIME: self.libraries_metadata.created_at,
            MetaNames.LAST_UPDATED: self.libraries_metadata.modified_at,
            MetaNames.LIBRARIES.URL: self.libraries_metadata.url
        }

        if self.libraries_entity.get('name') is not None:
            prop_map[MetaNames.LIBRARIES.NAME] = self.libraries_entity.get('name')
            self.name = self.libraries_entity.get('name')

        if self.libraries_entity.get('description', None) is not None:
            prop_map[MetaNames.LIBRARIES.DESCRIPTION] = self.libraries_entity.get('description',None)

        if self.libraries_entity.get('version') is not None:
            prop_map[MetaNames.LIBRARIES.VERSION] = self.libraries_entity.get('version')

        platform_value = None

        if self.libraries_entity.get('platform') is not None:
            platform_value = self.libraries_entity.get('platform')

        prop_map[MetaNames.LIBRARIES.PLATFORM] = platform_value

        libraries_url = self.libraries_metadata.url
        prop_map[MetaNames.LIBRARIES.URL] = libraries_url
        libraries_content_url = libraries_url + "/content"
        prop_map[MetaNames.LIBRARIES.CONTENT_URL] = libraries_content_url

        local_meta = MetaProps(prop_map)

        libraries_artifact = libraries_artifact_builder(
            uid=self.uid,
            name=self.name,
            meta_props=local_meta
        )

        libraries_artifact.client = self.client
        libraries_artifact._content_href = libraries_content_url

        return libraries_artifact
