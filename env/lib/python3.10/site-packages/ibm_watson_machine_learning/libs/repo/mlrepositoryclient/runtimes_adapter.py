#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames, MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact import RuntimesArtifact, RuntimesArtifactReader, RuntimesArtifactLoader


logger = logging.getLogger('WmlRuntimesAdapter')


class WmlRuntimesAdapter(object):
    """
    Adapter creating runtimes artifact using output from service.
    """

    def __init__(self, runtimes_output, client):

        self.runtimes_output = runtimes_output
        self.client = client
        self.runtimes_entity = runtimes_output.entity
        self.runtimes_metadata = runtimes_output.metadata
        self.uid = runtimes_output.metadata.guid
        self.name = None

    def artifact(self):
        runtimes_artifact_builder = type(
            "RuntimesArtifact",
            (RuntimesArtifactLoader, RuntimesArtifact, RuntimesArtifactReader, object),
            {}
        )

        prop_map = {
            MetaNames.CREATION_TIME: self.runtimes_metadata.created_at,
            MetaNames.LAST_UPDATED: self.runtimes_metadata.modified_at,
            MetaNames.RUNTIMES.URL: self.runtimes_metadata.url
        }

        if self.runtimes_entity.get('name') is not None:
            prop_map[MetaNames.RUNTIMES.NAME] = self.runtimes_entity.get('name')
            self.name = self.runtimes_entity.get('name')

        if self.runtimes_entity.get('description', None) is not None:
            prop_map[MetaNames.RUNTIMES.DESCRIPTION] = self.runtimes_entity.get('description',None)

        if self.runtimes_entity.get('platform') is not None:
            prop_map[MetaNames.RUNTIMES.PLATFORM] = self.runtimes_entity.get('platform')

        if self.runtimes_entity.get('custom_libraries') is not None:
            prop_map[MetaNames.RUNTIMES.CUSTOM_LIBRARIES_URLS] = self.runtimes_entity.get('custom_libraries')

        runtimes_content_url = self.runtimes_metadata.url + "/content"

        prop_map[MetaNames.RUNTIMES.CONTENT_URL] = runtimes_content_url

        local_meta = MetaProps(prop_map)

        runtimes_artifact = runtimes_artifact_builder(
            uid=self.uid,
            name=self.name,
            meta_props=local_meta
        )

        runtimes_artifact.client = self.client
        runtimes_artifact._content_href = runtimes_content_url

        return runtimes_artifact
