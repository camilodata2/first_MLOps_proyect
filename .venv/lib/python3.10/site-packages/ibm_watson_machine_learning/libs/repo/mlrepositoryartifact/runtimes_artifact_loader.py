#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate

import os, shutil


class RuntimesArtifactLoader(object):

    def download_runtimes(self, file_path):
        return self.__extract_content(file_path)

    def __extract_content(self, file_path):

        if file_path is None:
            file_path = 'runtimes_artifact'
            try:
                shutil.rmtree(file_path)
            except:
                pass
            os.makedirs(file_path)

        try:
            id_length = 20
            runtimes_generated_id = uid_generate(id_length)
            if self.runtimespec_path is not None:
                file_name = '{}/{}_{}.yml'.format(file_path, os.path.basename(self.runtimespec_path),runtimes_generated_id)
            else:
                file_name = '{}/{}_{}.yml'.format(file_path, self.name, runtimes_generated_id)
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
            if self._content_href.__contains__("runtimes"):
                return self.client.repository_api.download_artifact_content(self._content_href, 'false', accept='text/plain')
