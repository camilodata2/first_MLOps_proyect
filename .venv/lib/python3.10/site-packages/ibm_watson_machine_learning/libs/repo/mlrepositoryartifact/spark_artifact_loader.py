#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.util.compression_util import CompressionUtil
import os, shutil
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate


class SparkArtifactLoader(object):
    def load(self):
        return self.extract_content(lambda content_dir: self.load_content(content_dir))

    def extract_content(self, callback):
        id_length = 8
        lib_generated_id = uid_generate(id_length)
        directory_name = 'artifact' + '_' + lib_generated_id

        try:
            shutil.rmtree(directory_name)
        except:
            pass

        try:
            tar_file_name = '{}/artifact_content.tar'.format(directory_name)
            gz_file_name = '{}/artifact_content.tar.gz'.format(directory_name)

            os.makedirs(directory_name)

            input_stream = self.reader().read()
            file_content = input_stream.read()
            gz_f = open(gz_file_name, 'wb+')
            gz_f.write(file_content)
            gz_f.close()
            self.reader().close()
            CompressionUtil.decompress_file_gzip(gz_file_name, tar_file_name)
            CompressionUtil.extract_tar(tar_file_name, directory_name)

            artifact_instance = callback(directory_name)

            shutil.rmtree(directory_name)
            return artifact_instance
        except Exception as ex:
            shutil.rmtree(directory_name)
            raise ex
