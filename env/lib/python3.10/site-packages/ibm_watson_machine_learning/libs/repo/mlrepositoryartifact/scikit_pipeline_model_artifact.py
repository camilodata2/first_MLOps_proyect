#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepository import ModelArtifact
from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *
from .python_version import PythonVersion
import numpy as np
import json

lib_checker = LibraryChecker()

if lib_checker.installed_libs[SCIKIT]:
    from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.scikit_pipeline_reader import ScikitPipelineReader
    from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.xgboost_model_reader import XGBoostModelReader
    from sklearn.base import BaseEstimator
    from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.version_helper import ScikitVersionHelper, XGBoostVersionHelper

if lib_checker.installed_libs[XGBOOST]:
    from xgboost import XGBRegressor, XGBClassifier
    from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact.version_helper import ScikitVersionHelper
    import sys
    import os
    try:
        sys.stderr = open(os.devnull, 'w')
        import xgboost as xgb
    except Exception as ex:
        print ('Failed to import xgboost. Error: ' + str(ex))
    finally:
        sys.stderr.close()  # close /dev/null
        sys.stderr = sys.__stderr__


class ScikitPipelineModelArtifact(ModelArtifact):
    """
    Class of model artifacts created with MLRepositoryCLient.

    :param sklearn.pipeline.Pipeline scikit_pipeline_model: Pipeline Model which will be wrapped
    """
    def __init__(self, scikit_pipeline_model, training_features=None, training_target=None, feature_names=None,
                 label_column_names=None, uid=None, name=None, meta_props=MetaProps({})):
        lib_checker.check_lib(SCIKIT)
        super(ScikitPipelineModelArtifact, self).__init__(uid, name, meta_props)

        is_scikit, is_xgboost = False, False

        if issubclass(type(scikit_pipeline_model), BaseEstimator):
            is_scikit = True

        if not is_scikit and lib_checker.installed_libs[XGBOOST]:
            if isinstance(scikit_pipeline_model, xgb.Booster):
                is_xgboost = True

        if not (is_scikit or is_xgboost):
            raise ValueError('Invalid type for scikit ml_pipeline_model: {}'.
                             format(scikit_pipeline_model.__class__.__name__))

        self.ml_pipeline_model = scikit_pipeline_model
        self.ml_pipeline = None     # no pipeline or parent reference


        if meta_props.prop(MetaNames.RUNTIMES) is None and meta_props.prop(MetaNames.RUNTIME_UID) is None and meta_props.prop(MetaNames.FRAMEWORK_RUNTIMES) is None:
            ver = PythonVersion.significant()
            runtimes = '[{"name":"python","version": "'+ ver + '"}]'
            self.meta.merge(
                MetaProps({MetaNames.FRAMEWORK_RUNTIMES: runtimes})
            )

        if is_xgboost:
            self.meta.merge(
                MetaProps({
                    MetaNames.FRAMEWORK_NAME: XGBoostVersionHelper.model_type(scikit_pipeline_model),
                    MetaNames.FRAMEWORK_VERSION: XGBoostVersionHelper.model_version(scikit_pipeline_model)
                })
            )

            if(training_features is not None):
                if(training_target is None):
                    if not (isinstance(training_features, xgb.DMatrix)):
                        raise ValueError("Training target column has not been provided for the training data set")
                self.meta.merge(self._get_schema(training_features, training_target, feature_names, label_column_names))

            self._reader = XGBoostModelReader(self.ml_pipeline_model)
        else:
            if lib_checker.installed_libs[XGBOOST]:
                if (issubclass(type(scikit_pipeline_model), XGBClassifier) or issubclass(type(scikit_pipeline_model), XGBRegressor)):
                    if MetaNames.FRAMEWORK_LIBRARIES not in self.meta.meta:
                        framewk_library_entry = {"name": "xgboost",
                                                 "version": xgb.__version__}
                        framewk_library_entry_full = json.dumps([framewk_library_entry])
                        self.meta.merge(
                            MetaProps({
                                MetaNames.FRAMEWORK_LIBRARIES: framewk_library_entry_full
                            })
                        )
                else:
                    self._xgboost_in_model(scikit_pipeline_model)

            self.meta.merge(
                MetaProps({
                    MetaNames.FRAMEWORK_NAME: ScikitVersionHelper.model_type(scikit_pipeline_model),
                    MetaNames.FRAMEWORK_VERSION: ScikitVersionHelper.model_version(scikit_pipeline_model)
                })
            )

            if(training_features is not None):
                if(training_target is None):
                    raise ValueError("Training target column has not been provided for the training data set")
                self.meta.merge(self._get_schema(training_features, training_target, feature_names, label_column_names))

            if self.meta.prop(MetaNames.INPUT_DATA_SCHEMA) is None:
                try:
                    input_data_schema = {'fields': [f for f in self.meta.prop(MetaNames.TRAINING_DATA_SCHEMA)['fields'] if
                                                    f['name'] != self.meta.prop(MetaNames.LABEL_FIELD)], 'type': 'struct',
                                         'id': '1'}
                    self.meta.add(MetaNames.INPUT_DATA_SCHEMA, input_data_schema)
                except:
                    pass

            self._reader = ScikitPipelineReader(self.ml_pipeline_model)


    def _xgboost_in_model(self, model):
        import sklearn
        if '17' in sklearn.__version__:
            from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
        else:
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import VotingClassifier
        _estimators = [GridSearchCV, RandomizedSearchCV, VotingClassifier, OneVsRestClassifier, Pipeline]

        set_libraries = False

        for estimator in _estimators:
            if isinstance(model, estimator):
                if isinstance(model, GridSearchCV) or isinstance(model, RandomizedSearchCV):
                    estimator = model.best_estimator_
                    if isinstance(estimator, XGBRegressor) or isinstance(estimator, XGBClassifier):
                        set_libraries = True
                elif isinstance(model, Pipeline):
                    estimator = model.steps[-1][1]
                    if isinstance(estimator, XGBRegressor) or isinstance(estimator, XGBClassifier):
                        set_libraries = True
                else:
                    for est in model.estimators_:
                        if isinstance(est, XGBRegressor) or isinstance(est, XGBClassifier):
                            set_libraries = True

        if set_libraries:
            if MetaNames.FRAMEWORK_LIBRARIES not in self.meta.meta:
                framewk_library_entry = {"name": "xgboost",
                                         "version": xgb.__version__}
                framewk_library_entry_full = json.dumps([framewk_library_entry])
                self.meta.merge(
                    MetaProps({
                        MetaNames.FRAMEWORK_LIBRARIES: framewk_library_entry_full
                    })
                )

    def _get_schema(self, training_features, training_target, feature_names, label_column_names):
        is_xgboost_dmatrix = False

        if(training_target is None):           #training_target is None for Dmatrix
            training_props = {
                # "features": {"type": type(training_features).__name__, "fields": []},
                # "labels": {"type": type(training_features.get_label()).__name__, "fields": []}
                "type": type(training_features).__name__,
                "fields" : [],
                "labels": { "fields": []}
            }
        else:
            training_props = {
                "type": type(training_features).__name__,
                "fields": [],
                "labels": {"fields": []}
            }

        lib_checker.check_lib(PANDAS)
        import pandas as pd

        # Check feature types, currently supporting pandas df, numpy.ndarray, python lists and Dmatrix
        if(isinstance(training_features, pd.DataFrame)):
            for feature in training_features.dtypes.iteritems():
                training_props["fields"].append({"name": feature[0], "type": str(feature[1])})
        elif(isinstance(training_features, np.ndarray)):
            dims = training_features.shape
            if len(dims) == 1:
                if feature_names is None:
                    feature_names = 'f1'
                training_props["fields"].append({"name": feature_names, "type": type(training_features[0]).__name__})
            else:
                if feature_names is None:
                    feature_names = ['f' + str(i) for i in range(dims[1])]
                elif isinstance(feature_names, np.ndarray):
                    feature_names = feature_names.tolist()
                for i in range(dims[1]):
                    training_props["fields"].append({
                        "name": feature_names[i], "type": type(training_features.item(0, i)).__name__
                    })

        elif(isinstance(training_features, list)):
            if not isinstance(training_features[0], list):
                if feature_names is None:
                    feature_names = 'f1'
                training_props["fields"].append({"name": feature_names, "type": type(training_features[0]).__name__})
            else:
                if feature_names is None:
                    feature_names = ['f' + str(i) for i in range(len(training_features[0]))]
                elif isinstance(feature_names, np.ndarray):
                    feature_names = feature_names.tolist()
                for i in range(len(training_features[0])):
                    training_props["fields"].append({
                        "name": feature_names[i], "type": type(training_features[0][i]).__name__
                    })
        elif lib_checker.installed_libs[XGBOOST] and isinstance(training_features, xgb.DMatrix):
            for index, feature in enumerate(training_features.feature_names):   #start=0
                training_props["fields"].append({"name": feature, "type": str(training_features.feature_types[index])})
            is_xgboost_dmatrix = True
        else:
            raise ValueError("Unsupported training data type %s provided" % (type(training_features).__name__))

        #Check target or label data types
        if(isinstance(training_target, pd.DataFrame)):
            for feature in training_target.dtypes.iteritems():
                training_props["labels"]["fields"].append({"name": feature[0], "type": str(feature[1])})
        elif(isinstance(training_target, pd.Series)):
            training_props["labels"]["fields"].append({"name": training_target.name, "type": str(training_target.dtype)})
        elif(isinstance(training_target, np.ndarray)):
            dims = training_target.shape
            if len(dims) == 1:
                if label_column_names is None:
                    label_column_names = 'l1'
                elif isinstance(label_column_names, list) or isinstance(label_column_names, np.ndarray):
                    label_column_names = label_column_names[0]
                training_props["labels"]["fields"].append({"name": label_column_names, "type": type(training_target.item(0)).__name__})
            else:
                if label_column_names is None:
                    label_column_names = ['l' + str(i) for i in range(dims[1])]
                elif isinstance(label_column_names, np.ndarray):
                    label_column_names = label_column_names.tolist()
                for i in range(dims[1]):
                    training_props["labels"]["fields"].append({
                        "name": label_column_names[i], "type": type(training_target.item(0, i)).__name__
                    })
        elif(isinstance(training_target, list)):
            if not isinstance(training_target[0], list):
                if label_column_names is None:
                    label_column_names = 'l1'
                elif isinstance(label_column_names, list) or isinstance(label_column_names, np.ndarray):
                    label_column_names = label_column_names[0]
                training_props["labels"]["fields"].append({
                    "name": label_column_names, "type": type(training_target[0]).__name__
                })
            else:
                if label_column_names is None:
                    label_column_names = ['l' + str(i) for i in range(len(training_target[0]))]
                elif isinstance(label_column_names, np.ndarray):
                    label_column_names = label_column_names.tolist()
                for i in range(len(training_target[0])):
                    training_props["labels"]["fields"].append({
                        "name": label_column_names[i], "type": type(training_target[0][i]).__name__
                    })
        elif lib_checker.installed_libs[XGBOOST] and isinstance(training_features, xgb.DMatrix):  #For DMatrix labels is part of training_data and not training_target
            training_props["labels"]["fields"].append({"type": str(type(training_features.get_label()[0]).__name__)})
        else:
            raise ValueError("Unsupported label data type %s provided" % (type(training_target)))

        if is_xgboost_dmatrix:
            label = None
        else:
            label = training_props["labels"]["fields"][0]["name"]

        del training_props["labels"]
        return MetaProps({
            MetaNames.TRAINING_DATA_SCHEMA: training_props,
            MetaNames.LABEL_FIELD: label
        })

    def pipeline_artifact(self):
        """
        Returns None. Pipeline is not implemented for scikit model.
        """
        pass

    def reader(self):
        """
        Returns reader used for getting pipeline model content.

        :return: reader for sklearn.pipeline.Pipeline
        :rtype: ScikitPipelineReader
        """

        return self._reader

    def _copy(self, uid=None, meta_props=None):
        if uid is None:
            uid = self.uid

        if meta_props is None:
            meta_props = self.meta

        return ScikitPipelineModelArtifact(
            self.ml_pipeline_model,
            uid=uid,
            name=self.name,
            meta_props=meta_props
        )