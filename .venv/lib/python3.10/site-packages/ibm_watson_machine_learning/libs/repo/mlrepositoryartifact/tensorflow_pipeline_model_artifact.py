#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
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

lib_checker = LibraryChecker()

if lib_checker.installed_libs[TENSORFLOW]:
    import tensorflow as tf

class TensorflowPipelineModelArtifact(ModelArtifact):
    """
    Class of Tensorflow model artifacts created with MLRepositoryCLient.

    """
    def __init__(self,tensorflow_pipeline_model,
                 signature_def_map,
                 tags=None,
                 assets_collection=None,
                 legacy_init_op=None,
                 clear_devices=False,
                 main_op=None,
                 uid=None, name=None, meta_props=MetaProps({}),):
        lib_checker.check_lib(TENSORFLOW)
        super(TensorflowPipelineModelArtifact, self).__init__(uid, name, meta_props)

        if '2.1.0' in tf.__version__ and not isinstance(tensorflow_pipeline_model,tf.compat.v1.Session):
            raise TypeError("sess should be of type : %s" % tf.compat.v1.Session)
        elif not isinstance(tensorflow_pipeline_model,tf.Session):
            raise TypeError("sess should be of type : %s" % tf.Session)

        self.ml_pipeline_model = tensorflow_pipeline_model
        self.signature_def_map = signature_def_map
        self.tags = tags
        self.assets_collection = assets_collection
        self.legacy_init_op = legacy_init_op
        self.clear_devices = clear_devices
        self.main_op = main_op

        self.ml_pipeline = None     # no pipeline or parent reference

        if meta_props.prop(MetaNames.RUNTIMES) is None and meta_props.prop(MetaNames.RUNTIME_UID) is None and meta_props.prop(MetaNames.FRAMEWORK_RUNTIMES) is None:
            ver = PythonVersion.significant()
            runtimes = '[{"name":"python","version": "'+ ver + '"}]'
            self.meta.merge(
                MetaProps({MetaNames.FRAMEWORK_RUNTIMES: runtimes})
            )

        self.meta.merge(
            MetaProps({
                MetaNames.FRAMEWORK_NAME:   TensorflowVersionHelper.model_type(tensorflow_pipeline_model),
                MetaNames.FRAMEWORK_VERSION: TensorflowVersionHelper.model_version(tensorflow_pipeline_model)
            })
        )

    def pipeline_artifact(self):
        """
        Returns None. Pipeline is not implemented for Tensorflow model.
        """
        pass

    def reader(self):
        """
        Returns reader used for getting pipeline model content.

        :return: reader for TensorflowPipelineModelArtifact.pipeline.Pipeline
        :rtype: TensorflowPipelineReader
        """
        try:
            return self._reader
        except:
            self._reader = TensorflowPipelineReader(self.ml_pipeline_model,
                                                    self.signature_def_map,
                                                    self.tags,
                                                    self.assets_collection,
                                                    self.legacy_init_op,
                                                    self.clear_devices,
                                                    self.main_op)
            return self._reader

    def _copy(self, uid=None, meta_props=None):
        if uid is None:
            uid = self.uid

        if meta_props is None:
            meta_props = self.meta

        return TensorflowPipelineModelArtifact(
            self.ml_pipeline_model,
            self.signature_def_map,
            self.tags,
            self.assets_collection,
            self.legacy_init_op,
            self.clear_devices,
            self.main_op,
            uid=uid,
            name=self.name,
            meta_props=meta_props
        )


class TensorflowPipelineModelTarArtifact(ModelArtifact):
    """
    Class of serialized Tensorflow model artifacts in tar.gz format and created
    with MLRepositoryCLient.

    """
    def __init__(self,
                 tensorflow_tar_artifact,
                 uid=None,
                 name=None,
                 meta_props=MetaProps({}),):
        if not (lib_checker.installed_libs[TENSORFLOW]):
            raise NameError("Please install Tensorflow package and re-execute the command")
        super(TensorflowPipelineModelTarArtifact, self).__init__(uid, name, meta_props)

        self.ml_pipeline_model = tensorflow_tar_artifact
        self.ml_pipeline = None     # no pipeline or parent reference

    def pipeline_artifact(self):
        """
        Returns None. Pipeline is not implemented for Tensorflow model.
        """
        pass

    def reader(self):
        """
        Returns reader used for getting pipeline model content.

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

        return TensorflowPipelineModelTarArtifact(
            self.ml_pipeline_model,
            uid=uid,
            name=self.name,
            meta_props=meta_props
        )

    def unzip_artifact(self):
        try:
            tar = tarfile.open(self.ml_pipeline_model, 'r:gz')
        except Exception as ex:
            raise IOError('Unable to read the compressed archive file in {0} due to '
                          'error "{1}". '
                          'Ensure a valid tar archive is compressed in gzip format.'
                          .format(self.ml_pipeline_model, ex))
        return tar

    def get_keras_version(self):
        def extract_keras_version(extract_dir, h5_file_tar_info_name):
            import h5py
            keras_version_key = 'keras_version'
            file_name = os.path.join(extract_dir, h5_file_tar_info_name)
            try:
                with h5py.File(file_name) as f:
                    keras_version = f.attrs[keras_version_key] if keras_version_key in f.attrs else None
                    f.close()
            except Exception as ex:
                keras_version = None
                shutil.rmtree(extract_dir)
                raise Exception("Error while opening file to get Keras version. Error message - '{}' ".format(str(ex)))
            return keras_version.decode('utf-8')

        extract_dir = uid_generate(5)
        tar = self.unzip_artifact()

        # Return None if the artifact is not a valid keras archive
        if len(tar.getmembers()) != 1:
            return None

        h5_file_tar_info = tar.getmembers()[0]
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(os.path.join("./", extract_dir))
        tar.extract(h5_file_tar_info, extract_dir)
        keras_version = extract_keras_version(extract_dir, h5_file_tar_info.name)
        shutil.rmtree(extract_dir)
        return keras_version

    def is_valid_tf_archive(self):
        expected_file_name = "saved_model.pb"
        expected_file_with_path = os.path.join('.', expected_file_name)
        tar = self.unzip_artifact()
        is_valid_tf = True if expected_file_with_path in [file.name for file in tar.getmembers()] else False
        tar.close()
        return is_valid_tf

    def update_keras_version_meta(self, keras_version):
        KERAS_FRAMEWORK_NAME = "keras"
        # check if meta prop has already an entry for keras
        framewk_library_entry = {"name": KERAS_FRAMEWORK_NAME,
                                  "version": keras_version}
        framewk_library_entry_full = json.dumps([framewk_library_entry])

        if MetaNames.FRAMEWORK_LIBRARIES not in self.meta.meta:
            self.meta.add(MetaNames.FRAMEWORK_LIBRARIES, framewk_library_entry_full)
        else:
            lib_entry = json.loads(self.meta.meta[MetaNames.FRAMEWORK_LIBRARIES])
            for lib in lib_entry:
                if lib['name'] == KERAS_FRAMEWORK_NAME:
                    if lib['version'] != keras_version:
                        raise UnmatchedKerasVersion('Keras version specified as metadata and Keras version used to '
                                                    'train the model does not match. Keras version specified in '
                                                    'metadata is {0} . Keras version used to train the model is '
                                                    '{1}'.format(lib['version'], keras_version))
                    else:
                        return self
            lib_entry.append(framewk_library_entry)
            self.meta.meta.update({MetaNames.FRAMEWORK_LIBRARIES: json.dumps(lib_entry)})
        return self
