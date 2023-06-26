#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import shutil

from ibm_watson_machine_learning.libs.repo.mlrepository.artifact_reader import ArtifactReader
from ibm_watson_machine_learning.libs.repo.util.compression_util import CompressionUtil
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate


# TODO this part needs to be implemented
class SparkPipelineReader(ArtifactReader):
    def __init__(self, ml_pipeline, type_name):
        self.archive_path = None
        self.ml_pipeline = ml_pipeline
        self.type_name = type_name
        self.hummingbird_env = os.getenv('HB_RUNTIME_PROVIDER', "").upper()

    def read(self):
        return self._open_stream()

    def close(self):
        os.remove(self.archive_path)
        self.archive_path = None

    def _save_pipeline_archive(self):

        id_length = 20
        gen_id = uid_generate(id_length)
        temp_dir_name = '{}'.format(self.type_name + gen_id)
        if (self.hummingbird_env == 'HUMMINGBIRD') is True:
            temp_dir = os.path.join('/home/spark/shared/wml/repo/extract_', temp_dir_name)
        else:
            temp_dir = os.path.join('.', temp_dir_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        self.ml_pipeline.write().overwrite().save(temp_dir)
        archive_path = self._compress_artifact(temp_dir, gen_id)
        shutil.rmtree(temp_dir)
        return archive_path

    def _compress_artifact(self, compress_artifact, gen_id):
        tar_filename = '{}_content.tar'.format(self.type_name + gen_id)
        gz_filename = '{}.gz'.format(tar_filename)
        CompressionUtil.create_tar(compress_artifact, '.', tar_filename)
        CompressionUtil.compress_file_gzip(tar_filename, gz_filename)
        os.remove(tar_filename)
        return gz_filename

    def _open_stream(self):
        if self.archive_path is None:
            self.archive_path = self._save_pipeline_archive()
        return open(self.archive_path, 'rb')
