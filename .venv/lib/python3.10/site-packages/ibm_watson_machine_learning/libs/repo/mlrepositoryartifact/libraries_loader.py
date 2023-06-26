#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.libraries_artifact_loader import LibrariesArtifactLoader


class LibrariesLoader(LibrariesArtifactLoader):
    """
        Returns  Libraries instance associated with this library artifact.
        :return: library zip file
    """
    def download_library(self, file_path):
        return self.load(file_path)


