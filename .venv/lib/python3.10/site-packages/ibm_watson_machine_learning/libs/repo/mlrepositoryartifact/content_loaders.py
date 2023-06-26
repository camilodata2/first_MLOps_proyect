#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os

from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

if lib_checker.installed_libs[PYSPARK]:
    from pyspark.ml import Pipeline, PipelineModel

if lib_checker.installed_libs[SCIKIT]:
    from ..mlrepositoryartifact.version_helper import ScikitModelBinary
    try:
        # note only up to scikit version 0.20.3
        from sklearn.externals import joblib
    except ImportError:
        # only for scikit 0.23.*
        import joblib

    if lib_checker.installed_libs[XGBOOST]:
        from xgboost import Booster
        from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.version_helper import XGBoostModelBinary

def get_file_list(dir_path, ext):
    file_list = []
    for file in os.listdir(dir_path):
        if file.endswith(ext):
            file_list.append(os.path.join(dir_path, file))
    return file_list

class SparkPipelineContentLoader(object):
    @staticmethod
    def load_content(content_dir):
        lib_checker.check_lib(PYSPARK)
        return Pipeline.read().load(content_dir)


class MLPipelineContentLoader(object):
    @staticmethod
    def load_content(content_dir):
        lib_checker.check_lib(MLPIPELINE)
        from mlpipelinepy.mlpipeline import MLPipeline
        return MLPipeline.read().load(content_dir)


class IBMSparkPipelineContentLoader(object):
    @staticmethod
    def load_content(content_dir):
        lib_checker.check_lib(IBMSPARKPIPELINE)
        from pipeline import IBMSparkPipeline
        return IBMSparkPipeline.read().load(content_dir)


class SparkPipelineModelContentLoader(object):
    @staticmethod
    def load_content(content_dir):
        lib_checker.check_lib(PYSPARK)
        return PipelineModel.read().load(content_dir)


class MLPipelineModelContentLoader(object):
    @staticmethod
    def load_content(content_dir):
        lib_checker.check_lib(MLPIPELINE)
        from mlpipelinepy.mlpipeline import MLPipelineModel
        return MLPipelineModel.read().load(content_dir)


class IBMSparkPipelineModelContentLoader(object):
    @staticmethod
    def load_content(content_dir):
        lib_checker.check_lib(IBMSPARKPIPELINE)
        from pipeline import IBMSparkPipelineModel
        return IBMSparkPipelineModel.read().load(content_dir)


class ScikitPipelineModelContentLoader(object):
    INCOMPAT_MESSAGE = 'You may be trying to read with python 3 a joblib pickle generated with python 2. This feature is not supported by joblib.'

    @staticmethod
    def load_content(content_dir):
        lib_checker.check_lib(SCIKIT)
        extracted_files = get_file_list(content_dir, ScikitModelBinary.bin_ext())
        if(len(extracted_files)==0):
            extracted_files = get_file_list(content_dir, ScikitModelBinary.bin_ext_v0())
            if (len(extracted_files) == 0) or (extracted_files is None):
                raise Exception("No" + ScikitModelBinary.bin_ext() +
                                " file found in the saved model artifact")
        try:
            full_file_name = extracted_files[0]
            artifact_instance = joblib.load(full_file_name)
            return artifact_instance
        except ValueError as ve_ex:
            if ScikitPipelineModelContentLoader.INCOMPAT_MESSAGE in str(ve_ex):
                raise ValueError("Unable to load the model that was saved in Python 2.x runtime. Retry using the " +
                                 "model saved in Python 3.5 runtime.")
            else:
                raise ve_ex
        except Exception as ex:
            print ('Unable to load scikit model with sklearn.externals.joblib version ' + joblib.__version__)
            print ('Error message: ' + str(ex))
            raise ex



class TensorflowPipelineModelContentLoader(object):

    @staticmethod
    def load_content(content_dir,session,tags):
        lib_checker.check_lib(TENSORFLOW)
        import tensorflow as tf
        from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.version_helper import TensorflowModelBinary

        extracted_files = get_file_list(content_dir, TensorflowModelBinary.bin_ext())
        if len(extracted_files)==0:
            raise Exception("No " + TensorflowModelBinary.bin_ext() +
                            " file found in the saved model artifact")
        try:
            from tensorflow import logging
            logging.set_verbosity(logging.WARN)
            if '2.1.0' in tf.__version__:
                metagraphdef = tf.compat.v1.saved_model.loader.load(session, tags, content_dir)
            else:
                metagraphdef = tf.saved_model.loader.load(session, tags, content_dir)

            sig_def = list(metagraphdef.signature_def.values())[0]
            input_tensors = {}
            output_tensors = {}

            for input_key in list(sig_def.inputs.keys()):
                tensor_name = sig_def.inputs[input_key].name
                input_tensors[input_key] = session.graph.get_tensor_by_name(tensor_name)

            for output_key in list(sig_def.outputs.keys()):
                tensor_name = sig_def.outputs[output_key].name
                output_tensors[output_key] = session.graph.get_tensor_by_name(tensor_name)

            return TensorflowRuntimeArtifact(input_tensors, output_tensors,metagraphdef,session)
        except Exception as ex:
            print ('Unable to load Tensorflow model. Error: ' + str(ex))
            raise ex


class TensorflowRuntimeArtifact(object):
    def __init__(self,
                 input_tensors,
                 output_tensors,
                 metagraphdef,
                 session):
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.metagraphdef = metagraphdef
        self.session = session


    def get_input_tensors(self):
        return self.input_tensors

    def get_output_tensors(self):
        return self.output_tensors

    def get_metagraphdef(self):
        return self.metagraphdef

    def get_session(self):
        return self.session


