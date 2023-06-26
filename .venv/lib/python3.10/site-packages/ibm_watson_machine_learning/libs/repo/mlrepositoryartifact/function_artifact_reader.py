#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging

from ibm_watson_machine_learning.libs.repo.mlrepository.artifact_reader import ArtifactReader

logger = logging.getLogger('FunctionArtifactReader')


class FunctionArtifactReader(ArtifactReader):
    def __init__(self, compressed_archive_path):
        self.archive_path = compressed_archive_path

    def read(self):
        return self._open_stream()

    # This is a no. op. for GenericTarGZReader as we do not want to delete the
    # archive file.
    def close(self):
        pass

    def _open_stream(self):
        return open(self.archive_path, 'rb')

