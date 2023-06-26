#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .spark_version import SparkVersion

from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()

if lib_checker.installed_libs[SCIKIT]:
    from sklearn.base import BaseEstimator

class VersionHelper(object):
    @staticmethod
    def significant():
        import pipeline
        version_parts = [int(version_part) for version_part in pipeline.__version__.split('.')]
        return "{}.{}".format(version_parts[0], version_parts[1])

    @staticmethod
    def getFrameworkVersion(ml_pipeline):
        canonical_name = ml_pipeline.__class__.__name__
        if canonical_name == 'Pipeline':
            return SparkVersion.significant()
        elif canonical_name =='PipelineModel':
            return SparkVersion.significant()
        elif canonical_name == 'MLPipeline':
            return '0.1'
        elif canonical_name =='MLPipelineModel':
            return '0.1'

    @staticmethod
    def pipeline_type(ml_pipeline):
        canonical_name = ml_pipeline.__class__.__name__
        if canonical_name == 'Pipeline':
            return 'mllib'
        elif canonical_name == 'MLPipeline':
            return 'wml'
        else:
            raise ValueError('Unsupported Pipeline class: {}'.format(canonical_name))

    @staticmethod
    def model_type(ml_pipeline_model):
        class_name = ml_pipeline_model.__class__.__name__
        if class_name == 'PipelineModel':
            return 'mllib'
        elif class_name == 'MLPipelineModel':
            return 'wml'
        else:
            raise ValueError('Unsupported PipelineModel class: {}'.format(class_name))


class ScikitVersionHelper(object):
    @staticmethod
    def model_type(scikit_pipeline_model):
        lib_checker.check_lib(SCIKIT)
        class_name = scikit_pipeline_model.__class__.__name__
        if issubclass(type(scikit_pipeline_model), BaseEstimator):
            return 'scikit-learn'
        else:
            raise ValueError('Unsupported ScikitPipelineModel class: {}'.format(class_name))

    @staticmethod
    def model_version(scikit_pipeline_model):
        class_name = scikit_pipeline_model.__class__.__name__
        if issubclass(type(scikit_pipeline_model), BaseEstimator):
            import sklearn
            sk_version = sklearn.__version__
            if len(sk_version.split('.'))>2:
                split_sk = sk_version.split('.')
                sk_version = split_sk[0]+'.' + split_sk[1]
            return format(sk_version)
        else:
            raise ValueError('Unsupported ScikitPipelineModel class: {}'.format(class_name))


class TensorflowVersionHelper(object):

    @staticmethod
    def model_type(tensorflow_pipeline_model):
        lib_checker.check_lib(TENSORFLOW)
        class_name = tensorflow_pipeline_model.__class__.__name__
        import tensorflow as tf
        if '2.1.0' in tf.__version__ :
            if isinstance(tensorflow_pipeline_model, tf.compat.v1.Session):
                return 'tensorflow'
            else:
                raise TypeError(
                    "Expecting object of type : %s" % tf.compat.v1.Session + " but got %s" % type(tensorflow_pipeline_model))

        else:
            if isinstance(tensorflow_pipeline_model, tf.Session):
                return 'tensorflow'
            else:
                raise TypeError("Expecting object of type : %s" % tf.Session + " but got %s" % type(tensorflow_pipeline_model))



    @staticmethod
    def model_version(tensorflow_pipeline_model):
        import tensorflow as tf
        class_name = tensorflow_pipeline_model.__class__.__name__
        if '2.1.0' in tf.__version__ :
            if isinstance(tensorflow_pipeline_model, tf.compat.v1.Session):
                tf_version = tf.__version__
                if len(tf_version.split('.')) > 2:
                    split_tf = tf_version.split('.')
                    tf_version = split_tf[0] + '.' + split_tf[1]
                return format(tf_version)
            else:
                raise TypeError(
                    "Expecting object of type : %s" % tf.compat.v1.Session + " but got %s" % type(tensorflow_pipeline_model))
        else:

            if isinstance(tensorflow_pipeline_model,tf.Session):
                tf_version = tf.__version__
                if len(tf_version.split('.'))>2:
                    split_tf = tf_version.split('.')
                    tf_version = split_tf[0]+'.' + split_tf[1]
                return format(tf_version)
            else:
                raise TypeError("Expecting object of type : %s" % tf.Session + " but got %s" % type(tensorflow_pipeline_model))



class XGBoostVersionHelper(object):
    @staticmethod
    def model_type(xgboost_model):
        lib_checker.check_lib(XGBOOST)
        from xgboost import Booster
        if isinstance(xgboost_model, Booster):
            return 'xgboost'
        else:
            raise ValueError('Unsupported XGBoost model class: {}'.format(xgboost_model.__class__.__name__))


    @staticmethod
    def model_version(xgboost_model):
        lib_checker.check_lib(XGBOOST)
        from xgboost import Booster
        import xgboost
        if isinstance(xgboost_model, Booster):
            xgb_version = xgboost.__version__
            return xgb_version

class ScikitModelBinary(object):
    @staticmethod
    def model_bin_name():
        return "scikit_model.pkl"

    @staticmethod
    def bin_ext():
        return ".pkl"

    @staticmethod
    def bin_ext_v0():
        return ".bin"

class XGBoostModelBinary(object):
    @staticmethod
    def model_bin_name():
        return "xgboost_model.pkl"

    @staticmethod
    def bin_ext():
        return ".pkl"

class TensorflowModelBinary(object):

    @staticmethod
    def bin_ext():
        return ".pb"
