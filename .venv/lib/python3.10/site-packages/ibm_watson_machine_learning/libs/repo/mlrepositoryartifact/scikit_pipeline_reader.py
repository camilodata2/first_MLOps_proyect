#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import shutil
import logging

from ibm_watson_machine_learning.libs.repo.mlrepository.artifact_reader import ArtifactReader
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.version_helper import ScikitModelBinary
from ibm_watson_machine_learning.libs.repo.util.compression_util import CompressionUtil
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

if lib_checker.installed_libs[SCIKIT]:
    import sklearn
    try:
        # note only up to scikit version 0.20.3
        from sklearn.externals import joblib
    except ImportError:
        # only for scikit 0.23.*
        import joblib

logger = logging.getLogger('ScikitPipelineReader')


class ScikitPipelineReader(ArtifactReader):
    def __init__(self, scikit_pipeline):
        self.archive_path = None
        self.scikit_pipeline = scikit_pipeline
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
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        self._save_scikit_pipeline_to_file(temp_dir)
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

    def _save_scikit_pipeline_to_file(self, path):
        try:
            full_file_name = os.path.join(path, ScikitModelBinary.model_bin_name())
            joblib.dump(self.scikit_pipeline, full_file_name)
        except Exception as e:
            logMsg = "ScikitPipelineReader Save to file failed with exception " + str(e)
            logger.info(logMsg)
            raise e
