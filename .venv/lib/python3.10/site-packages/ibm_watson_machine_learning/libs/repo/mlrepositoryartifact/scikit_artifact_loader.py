#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.util.compression_util import CompressionUtil
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
import os, shutil

class ScikitArtifactLoader(object):
    def load(self,as_type):
        if self.meta.prop(MetaNames.TYPE).startswith('xgboost'):
            if (as_type == 'Booster') or (as_type is None):
                return self.extract_content(lambda content_dir: self.load_content(content_dir))._Booster
            elif as_type == 'XGBRegressor':
                return self.extract_content(lambda content_dir: self.load_content(content_dir))
            else:
                raise ValueError("Unknown 'as_type' value for xgboost model")
        return self.extract_content(lambda content_dir: self.load_content(content_dir))

    def extract_content(self, callback):
        directory_name = 'artifact'

        try:
            shutil.rmtree(directory_name)
        except:
            pass

        try:
            id_length = 20
            dir_id = uid_generate(id_length)
            model_dir_name = directory_name + dir_id
            tar_file_name = '{}/artifact_content.tar'.format(model_dir_name)
            gz_file_name = '{}/artifact_content.tar.gz'.format(model_dir_name)

            os.makedirs(model_dir_name)

            input_stream = self.reader().read()
            file_content = input_stream.read()
            gz_f = open(gz_file_name, 'wb+')
            gz_f.write(file_content)
            gz_f.close()
            self.reader().close()
            CompressionUtil.decompress_file_gzip(gz_file_name, tar_file_name)
            CompressionUtil.extract_tar(tar_file_name, model_dir_name)

            artifact_instance = callback(model_dir_name)

            shutil.rmtree(model_dir_name)
            return artifact_instance
        except Exception as ex:
            shutil.rmtree(model_dir_name)
            raise ex
