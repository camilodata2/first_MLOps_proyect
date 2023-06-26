#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepository import ModelArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.tensorflow_pipeline_reader import TensorflowPipelineReader
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.generic_file_reader import GenericFileReader
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.version_helper import TensorflowVersionHelper
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *
from .python_version import PythonVersion
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate
from ibm_watson_machine_learning.libs.repo.util.exceptions import UnmatchedKerasVersion
import os, shutil, tarfile, json

class GenericArchivePipelineModelArtifact(ModelArtifact):
    """
    Class of  PMML,SPSS-MODELER,CAFFE,CAFFE2,PYTORCH,TORCH,MXNET,THEANO,BLUECONNECT and MXNET model artifacts created
    with MLRepositoryCLient.

    """
    def __init__(self,
                 generic_artifact,
                 uid=None,
                 name=None,
                 meta_props=MetaProps({}),):

        super(GenericArchivePipelineModelArtifact, self).__init__(uid, name, meta_props)

        self.ml_pipeline_model = generic_artifact
        self.ml_pipeline = None     # no pipeline or parent reference

    def pipeline_artifact(self):
        """
        Returns None. Pipeline is not implemented for archive model.
        """
        pass

    def reader(self):
        """
        Returns reader used for getting archive model content.

        :return: reader for TensorflowPipelineModelArtifact.pipeline.Pipeline
        :rtype: TensorflowPipelineReader
        """
        try:
            return self._reader
        except:
            self._reader = GenericFileReader(self.ml_pipeline_model)
            return self._reader

    def _copy(self, uid=None, meta_props=None):
        if uid is None:
            uid = self.uid

        if meta_props is None:
            meta_props = self.meta

        return GenericArchivePipelineModelArtifact(
            self.ml_pipeline_model,
            uid=uid,
            name=self.name,
            meta_props=meta_props
        )


    ## Below methods is to validate the caffe model archive

    def unzip_artifact(self):
        try:
            tar = tarfile.open(self.ml_pipeline_model, 'r:gz')
        except Exception as ex:
            raise IOError('Unable to read the compressed archive file in {0} due to '
                          'error "{1}". '
                          'Ensure a valid tar archive is compressed in gzip format.'
                          .format(self.ml_pipeline_model, ex))
        return tar

    def extract_tar_file(self):
        tar_url=self.ml_pipeline_model
        extract_dir =  uid_generate(5)
        tar = self.unzip_artifact()

        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(os.path.join("./", extract_dir))
        tar = tarfile.open(tar_url, 'r')
        for item in tar:
            tar.extract(item, extract_dir)

        return extract_dir





    def _check_multiple_ext_file(self, path, expected_ext):
        file_list = os.listdir(path)
        ext_count = 0
        for file_name in file_list:
            _, ext = os.path.splitext(os.path.join(path, file_name))
            if ext == expected_ext:
                ext_count += 1
            if ext_count > 1:
                return True
        return False

    def is_valid_caffe_archive(self,path):
        file_list = os.listdir(path)
        expected_files_ext = {'.json': False, '.prototxt': False, '.caffemodel': False}
        for file_name in file_list:
            name, ext = os.path.splitext(file_name)
            if ext in expected_files_ext:
                expected_files_ext[ext] = True

        for key in expected_files_ext:
            if not expected_files_ext[key]:
                shutil.rmtree(path)
                raise ValueError("Given archive doesn't have a file with extension {}".format(key))

        expected_file_name = "deployment-meta.json"
        try:
            with open(os.path.join(path, expected_file_name)) as data_file:
                json_dict = json.load(data_file)
        except Exception:
            shutil.rmtree(path)
            raise ValueError("The given archive doesn't have the file deployment-meta.json")


        if 'output_layers' not in json_dict:
            shutil.rmtree(path)
            raise ValueError("deployment-meta.json file does not have key output_layers")


        if 'network_definitions_file_name' in json_dict:
            nw_file_name = json_dict['network_definitions_file_name']
            if nw_file_name not in file_list:
                shutil.rmtree(path)
                raise ValueError("Given archive doesn't have a file defined for Key 'network_definitions_file_name' {}")

        if 'weights_file_name' in json_dict:
            weights_file_name = json_dict['weights_file_name']
            if weights_file_name not in file_list:
                shutil.rmtree(path)
                raise ValueError("Given archive doesn't have a file defined for Key 'weights_file_name' {}")

        if self._check_multiple_ext_file(path,'.prototxt'):
            if 'network_definitions_file_name' not in json_dict:
                shutil.rmtree(path)
                raise ValueError("Key 'network_definitions_file_name' expected in deployment-meta.json")

        if self._check_multiple_ext_file(path,'.caffemodel'):
            if 'weights_file_name' not in json_dict:
                shutil.rmtree(path)
                raise ValueError("Key 'weights_file_name' expected in deployment-meta.json")

        shutil.rmtree(path)

