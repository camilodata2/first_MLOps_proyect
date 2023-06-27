#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import shutil
import logging

from ibm_watson_machine_learning.libs.repo.mlrepository.artifact_reader import ArtifactReader
from ibm_watson_machine_learning.libs.repo.util.compression_util import CompressionUtil
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

logger = logging.getLogger('TensorflowPipelineReader')


class TensorflowPipelineReader(ArtifactReader):
    def __init__(self, tensorflow_pipeline,
                 signature_def_map,
                 tags,
                 assets_collection,
                 legacy_init_op,
                 clear_devices,
                 main_op):
        self.archive_path = None
        self.tensorflow_pipeline = tensorflow_pipeline
        self.signature_def_map = signature_def_map
        self.tags = tags
        self.assets_collection = assets_collection
        self.legacy_init_op = legacy_init_op
        self.clear_devices = clear_devices
        self.main_op = main_op
        self.type_name = 'model'

    def read(self):
        return self._open_stream()

    def close(self):
        os.remove(self.archive_path)
        self.archive_path = None

    def _save_pipeline_archive(self):
        id_length = 20
        gen_id = uid_generate(id_length)
        temp_dir_name = '{}'.format(self.type_name + gen_id)
        temp_dir = os.path.join('.', temp_dir_name)
        self._save_tensorflow_model_to_dir(temp_dir)
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

    def _save_tensorflow_model_to_dir(self, path):
        lib_checker.check_lib(TENSORFLOW)
        import tensorflow as tf
        try:
            from tensorflow import logging
            logging.set_verbosity(logging.WARN)
            if '2.1.0' in tf.__version__ :
                builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(path)
            else:
                builder = tf.saved_model.builder.SavedModelBuilder(path)
            builder.add_meta_graph_and_variables(sess=self.tensorflow_pipeline,
                                                 tags=self.tags,
                                                 signature_def_map=self.signature_def_map,
                                                 assets_collection=self.assets_collection,
                                                 legacy_init_op=self.legacy_init_op,
                                                 clear_devices=self.clear_devices,
                                                 main_op=self.main_op)
            builder.save()
        except Exception as e:
            logMsg = "Tensorflow model Save failed with exception " + str(e)
            logger.info(logMsg)
            raise e

