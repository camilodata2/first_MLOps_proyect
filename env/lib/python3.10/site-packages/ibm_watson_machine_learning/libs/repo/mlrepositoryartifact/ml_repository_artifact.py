#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function

import json
from os import path, devnull
from ibm_watson_machine_learning.libs.repo.mlrepository.meta_names import MetaNames
from ibm_watson_machine_learning.libs.repo.base_constants import *
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.util.exceptions import MetaPropMissingError
lib_checker = LibraryChecker()
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps, PipelineArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.experiment_artifact import ExperimentArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.libraries_artifact import LibrariesArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.runtimes_artifact import RuntimesArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.generic_archive_pipeline_model_artifact import GenericArchivePipelineModelArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.function_artifact import FunctionArtifact
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.hybrid_pipeline_model_artifact import HybridPipelineModelArtifact


def MLRepositoryArtifact(ml_artifact= None,
                         training_data=None,
                         training_dataref=None,
                         training_target=None,
                         feature_names=None,
                         label_column_names=None,
                         pipeline_artifact=None,
                         name=None,
                         meta_props=MetaProps({}),
                         signature_def_map=None,
                         tags=None,
                         assets_collection=None,
                         legacy_init_op=None,
                         clear_devices=False,
                         main_op=None):
    """
    Returns SparkPipelineModelArtifact or SparkPipelineArtifact or ScikitPipelineModelArtifact depending on params.
    Except first param (ml_artifact) all other params should be named or in right order.

    :param ml_artifact: pyspark.ml.Pipeline or pyspark.ml.PipelineModel or subclass of sklearn.base.BaseEstimator or
    xgboost.Booster
    :param training_data: training data used to train the model. This parameter is mandatory for spark pipeline model
     and optional for Pipeline artifacts, Scikit-Learn and XGBoost models. This parameter has to be of type DataFrame
     for spark pipeline model and of type pandas.DataFrame, a list or numpy.ndarray for Scikit-Learn models
    :param training_dataref:
    :param training_target: Labels for the training data in case of Scikit-Learn models. This parameter is mandatory for
    Scikit-Learn models in cases where the training_data parameter is provided.
    Valid input types for this parameter are: pandas.DataFrame, pandas.Series, numpy.ndarray and list.
    :param feature_names: Optional argument containing the feature names for the training data in case of Scikit-Learn
    models. Valid types are numpy.ndarray and list
    :param label_column_names: Optional argument referring to the label column name of the training data pertaining to
    Scikit-Learn models. This is applicable only in the case where the training data is not of type - pandas.DataFrame.
    :param pipeline_artifact:
    :param name: Name for artifact
    :param meta_props: Properties of the artifact to be saved
    :return: Any of the following artifacts based on the type of the model: SparkPipelineArtifact,
    SparkPipelineModelArtifact, ScikitPipelineModelArtifact

    >>> model_artifact = MLRepositoryArtifact(model, name='test-model-name', training_data=training, meta_props=MetaProps({
    >>> MetaNames.EVALUATION_METRICS: json.dumps([{
    >>>        "name": "accuracy",
    >>>        "value": 0.95,
    >>>        "threshold": 0.9}]) }))
    >>> created_model_artifact = ml_repository_client.models.save(model_artifact)

    """
    if isinstance(meta_props, dict):
        meta_props = MetaProps(meta_props.copy())

    if name is not None and not isinstance(name, str):
        raise ValueError('Invalid type for name: {}'.format(name.__class__.__name__))

    # check if this is libraries artifact
    #if meta_props.get()[MetaNames.LIBRARIES.NAME] is not None:
    if MetaNames.LIBRARIES.PATCH_INPUT in meta_props.get():
        return _get_libraries_artifact(ml_artifact, name, meta_props)

    if MetaNames.LIBRARIES.PLATFORM in meta_props.get() and \
                    MetaNames.LIBRARIES.VERSION in meta_props.get():
        if meta_props.get()[MetaNames.LIBRARIES.PLATFORM] is not None and \
                        meta_props.get()[MetaNames.LIBRARIES.VERSION] is not None:

            if ml_artifact is None:
                return _get_libraries_artifact(ml_artifact, name, meta_props)
            else:
                if isinstance(ml_artifact, str):
                    if ("tar.gz" not in ml_artifact) and (".tgz" not in ml_artifact) and ("zip" not in ml_artifact):
                        raise TypeError('The artifact specified ( {} ) is not of type "tar.gz" or "tgz" or "zip". Only compressed archive of '
                                    '"tar.gz" or "tgz" format is supported.'.format(ml_artifact))

                    if not path.exists(ml_artifact):
                        raise IOError('The artifact specified ( {} ) does not exist.'.format(ml_artifact))
                    return _get_libraries_artifact(ml_artifact, name, meta_props)

    if ml_artifact is not None:
        if name is None:
            raise ValueError('Invalid input, name can not be None.')

    if MetaNames.RUNTIMES.PLATFORM in meta_props.get():
        return _get_runtimes_artifact(ml_artifact, name, meta_props)

    if MetaNames.RUNTIMES.PATCH_INPUT in meta_props.get():
        return _get_runtimes_artifact(ml_artifact, name, meta_props)

    if ml_artifact is None:
        if (MetaNames.FRAMEWORK_NAME in meta_props.get()) or (MetaNames.FRAMEWORK_VERSION in meta_props.get()):
            if meta_props.get()[MetaNames.FRAMEWORK_NAME] == HYBRID:
               if (MetaNames.CATEGORY in meta_props.get()) or (MetaNames.CATEGORY in meta_props.get()):
                   if (MetaNames.CONTENT_LOCATION in meta_props.get()) or (MetaNames.CONTENT_LOCATION in meta_props.get()):
                      return  _get_hybrid_pipeline_model_artifact(name=name, meta_props=meta_props)
                   else:
                       raise ValueError("Missing required meta_props: CONTENT_LOCATION for Hybrid model")
               else:
                   raise ValueError("Missing required meta_props: CATEGORY for Hybrid model")
            else:
                return _get_experiment_artifact(meta_props=meta_props)
        else:
            return _get_experiment_artifact(meta_props=meta_props)

    if training_dataref is not None:
        json_training_dataref = json.loads(training_dataref)
        meta_props.add('trainingDataRef', json_training_dataref)

    if not isinstance(meta_props, MetaProps):
        raise ValueError('Invalid type for meta_props: {}'.format(meta_props.__class__.__name__))

    if lib_checker.installed_libs[PYSPARK]:
        from pyspark.ml.pipeline import Pipeline, PipelineModel
        from pyspark.sql import DataFrame
        if ((training_data is not None and not isinstance(training_data, DataFrame)) and
                issubclass(type(ml_artifact), Pipeline)):
            raise ValueError('Invalid type for training_data: {}'.format(training_data.__class__.__name__))

        if pipeline_artifact is not None and not issubclass(type(pipeline_artifact), PipelineArtifact):
            raise ValueError('Invalid type for pipeline_artifact: {}'.format(pipeline_artifact.__class__.__name__))

        if issubclass(type(ml_artifact), Pipeline):
            return _get_pipeline(ml_artifact, name=name, meta_props=meta_props)
        elif training_data is not None and issubclass(type(ml_artifact), PipelineModel):
            return _get_pipeline_model(ml_artifact,
                                       training_data=training_data,
                                       pipeline_artifact=pipeline_artifact,
                                       name=name,
                                       meta_props=meta_props)

    if lib_checker.installed_libs[SCIKIT]:
        from sklearn.base import BaseEstimator
        if issubclass(type(ml_artifact), BaseEstimator):
            return _get_scikit_pipeline_model(ml_artifact,
                                              training_data,
                                              training_target,
                                              feature_names,
                                              label_column_names,
                                              name=name,
                                              meta_props=meta_props)
        if lib_checker.installed_libs[XGBOOST]:
            import sys
            try:
                sys.stderr = open(devnull, 'w')
                import xgboost as xgb
            except Exception as ex:
                print('Failed to import xgboost. Error: ' + str(ex))
            finally:
                sys.stderr.close()  # close /dev/null
                sys.stderr = sys.__stderr__

            if isinstance(ml_artifact, xgb.Booster):
                return _get_xgboost_model(ml_artifact,
                                          training_data,
                                          training_target,
                                          feature_names,
                                          label_column_names,
                                          name=name,
                                          meta_props=meta_props)

    if lib_checker.installed_libs[MLPIPELINE]:
        from mlpipelinepy.mlpipeline import MLPipeline, MLPipelineModel
        if issubclass(type(ml_artifact), MLPipeline):
            return _get_pipeline(ml_artifact, name=name, meta_props=meta_props)
        elif training_data is not None and issubclass(type(ml_artifact), MLPipelineModel):
            return _get_pipeline_model(ml_artifact, training_data=training_data, pipeline_artifact=pipeline_artifact,
                                       name=name, meta_props=meta_props)

    if lib_checker.installed_libs[TENSORFLOW]:
        import tensorflow as tf
        import sys
        if sys.version_info < (3, 7) and \
                (('2.1.0' in tf.__version__ and isinstance(ml_artifact,tf.compat.v1.Session) or
                  isinstance(ml_artifact,tf.Session))):
            if signature_def_map is None:
                raise ValueError("Missing required parameter: signature_def_map")
            elif signature_def_map is  not None:
                from tensorflow.core.protobuf import meta_graph_pb2
                if not isinstance(signature_def_map,dict):
                    raise TypeError("signature_def_map should be of type : %s" % dict)
                for key, value in signature_def_map.items():
                    if not isinstance(value,meta_graph_pb2.SignatureDef):
                        raise TypeError("signature_def_map value %s should be of "
                                    "type : %s" %(value,meta_graph_pb2.SignatureDef))

            if tags is None:
                if '2.1.0' in tf.__version__:
                    tags = [tf.compat.v1.saved_model.tag_constants.SERVING]
                else:
                    tags = [tf.saved_model.tag_constants.SERVING]
            elif tags is not None :
                if not isinstance(tags,list):
                    raise TypeError("tags should be of type : %s" % list)
                else:
                    if '2.1.0' in tf.__version__ and tf.compat.v1.saved_model.tag_constants.SERVING not in tags:
                        tags.append(tf.saved_model.tag_constants.SERVING)
                    elif tf.saved_model.tag_constants.SERVING not in tags:
                        tags.append(tf.saved_model.tag_constants.SERVING)

            if legacy_init_op is not None:
                from tensorflow.python.framework import ops
                if not isinstance(legacy_init_op, ops.Operation):
                    raise TypeError("legacy_init_op needs to be an type: %s" % ops.Operation)


            return _get_tensorflow_model(ml_artifact,
                                        name=name,
                                        meta_props=meta_props,
                                        signature_def_map= signature_def_map,
                                        tags=tags,
                                        assets_collection=assets_collection,
                                        legacy_init_op=legacy_init_op,
                                        clear_devices=clear_devices,
                                        main_op=main_op)

    # check if ml_artifact is a tar.gz of a tensorflow model
    if isinstance(ml_artifact, str):

        if (MetaNames.TYPE in meta_props.get()):
         if "tensorflow" in meta_props.get()[MetaNames.TYPE]:
            if ("tar.gz" not in ml_artifact) and (".tgz" not in ml_artifact):
                raise TypeError('The artifact specified ( {} ) is not of type "tar.gz" or "tgz". Only compressed archive of '
                             '"tar.gz" or "tgz" format is supported.'.format(ml_artifact))

         if not path.exists(ml_artifact):
            raise IOError('The artifact specified ( {} ) does not exist.'.format(ml_artifact))

         if MetaNames.TYPE not in meta_props.get():
            raise MetaPropMissingError('Value specified for "meta_props" does not contain value for '
                             '"MetaNames.TYPE"')

         # if MetaNames.FRAMEWORK_VERSION not in meta_props.get():
         #    if meta_props.get()[MetaNames.TYPE] == PMML:
         #        meta_props.add(MetaNames.FRAMEWORK_VERSION, "NA")
         #    else:
         #        raise MetaPropMissingError('Value specified for "meta_props" does not contain value for '
         #                     '"MetaNames.FRAMEWORK_VERSION"')

         if not MetaNames.is_supported_tar_framework(meta_props.get()[MetaNames.TYPE]):
            raise ValueError('Value specified for MetaNames.TYPE ( {} ) is not '
                             'supported.'.format(meta_props.get()[MetaNames.TYPE]))

         if "tensorflow" in meta_props.get()[MetaNames.TYPE]:
            return _get_tensorflow_model_tar(ml_artifact, name, meta_props)

         if MetaNames.is_archive_framework(meta_props.get()[MetaNames.TYPE]):
            if "pmml" in meta_props.get()[MetaNames.TYPE] and ".xml" not in ml_artifact:
               # check if the file is a xml file
               raise TypeError('The artifact specified ( {} ) is not an xml file.')
            return _get_generic_archive_model(ml_artifact, name, meta_props)
         else:
            return _get_function_artifact(ml_artifact, name, meta_props)

    raise ValueError('Invalid type for ml_artifact: {}'.format(ml_artifact.__class__.__name__))


def _get_pipeline(pipeline, name, meta_props):
    from .spark_pipeline_artifact import SparkPipelineArtifact
    return SparkPipelineArtifact(pipeline, name=name, meta_props=meta_props)

def _get_experiment_artifact(meta_props):
    return ExperimentArtifact(meta_props=meta_props)

def _get_hybrid_pipeline_model_artifact(name, meta_props):
    return HybridPipelineModelArtifact(name=name, meta_props=meta_props)

def _get_pipeline_model(pipeline_model, training_data, pipeline_artifact, name, meta_props):
    from .spark_pipeline_model_artifact import SparkPipelineModelArtifact
    return SparkPipelineModelArtifact(pipeline_model, training_data=training_data, pipeline_artifact=pipeline_artifact, name=name, meta_props=meta_props)


def _get_scikit_pipeline_model(scikit_pipeline_model, training_features, training_target, feature_names,
                               label_column_names, name, meta_props):
    from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.scikit_pipeline_model_artifact import ScikitPipelineModelArtifact
    return ScikitPipelineModelArtifact(scikit_pipeline_model,
                                       training_features,
                                       training_target,
                                       feature_names,
                                       label_column_names,
                                       name=name,
                                       meta_props=meta_props)


def _get_xgboost_model(xgboost_model, training_features, training_target, feature_names,
                       label_column_names, name, meta_props):
    from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.scikit_pipeline_model_artifact import ScikitPipelineModelArtifact
    return ScikitPipelineModelArtifact(xgboost_model,
                                       training_features,
                                       training_target,
                                       feature_names,
                                       label_column_names,
                                       name=name,
                                       meta_props=meta_props)


def _get_tensorflow_model(tensorflow_model, name, meta_props,signature_def_map,tags,assets_collection,legacy_init_op,clear_devices,main_op):
    from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.tensorflow_pipeline_model_artifact import TensorflowPipelineModelArtifact
    return TensorflowPipelineModelArtifact(tensorflow_model,
                                      name=name,
                                      meta_props=meta_props,
                                      signature_def_map=signature_def_map,
                                      tags=tags,
                                      assets_collection=assets_collection,
                                      legacy_init_op=legacy_init_op,
                                      clear_devices=clear_devices,
                                      main_op=main_op)


def _get_tensorflow_model_tar(tf_tar_artifact, name, meta_props):
    from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.tensorflow_pipeline_model_artifact import TensorflowPipelineModelTarArtifact
    return TensorflowPipelineModelTarArtifact(tf_tar_artifact,
                                              name=name,
                                              meta_props=meta_props)


def _get_generic_archive_model(generic_artifact, name, meta_props):
    return GenericArchivePipelineModelArtifact(generic_artifact,
                                     name=name,
                                     meta_props=meta_props)

def _get_function_artifact(generic_artifact, name, meta_props):
    return FunctionArtifact(generic_artifact,
                            name=name,
                            meta_props=meta_props)

def _get_libraries_artifact(generic_artifact, name, meta_props):
    return LibrariesArtifact(generic_artifact, name,
                             meta_props=meta_props)

def _get_runtimes_artifact(generic_artifact, name, meta_props):
    return RuntimesArtifact(generic_artifact, name,
                             meta_props=meta_props)
