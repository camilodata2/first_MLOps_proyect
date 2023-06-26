#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate

import os, shutil


class LibrariesArtifactLoader(object):

    def load(self, file_path):
        return self.extract_content(file_path)

    def extract_content(self, file_path):

        if file_path is None:
            file_path = 'library_artifact'
            try:
                shutil.rmtree(file_path)
            except:
                pass
            os.makedirs(file_path)

        try:
            id_length = 20
            lib_generated_id = uid_generate(id_length)
            if self.library is not None:
                file_name = '{}/{}_{}.zip'.format(file_path, os.path.basename(self.library),lib_generated_id)
            else:
                file_name = '{}/{}_{}.zip'.format(file_path, self.name, lib_generated_id)
            input_stream = self._content_reader()
            file_content = input_stream.read()
            gz_f = open(file_name, 'wb+')
            gz_f.write(file_content)
            gz_f.close()
            return os.path.abspath(file_name)
        except Exception as ex:
            raise ex

    def _content_reader(self):
        if self._content_href is not None:
            if self._content_href.__contains__("libraries"):
                return self.client.repository_api.download_artifact_content(self._content_href, 'false', accept='application/gzip')
