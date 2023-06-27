#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import traceback

from lale.operators import TrainablePipeline
from lale.lib.lale import Hyperopt
from lale.operators import TrainedPipeline
from lale import wrap_imported_operators
from lale.helpers import import_from_sklearn_pipeline

from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.neighbors import KNeighborsClassifier as KNN

# from lale.lib.sklearn import KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression



from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService

from ibm_watson_machine_learning.experiment.autoai.autoai import PipelineTypes
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, is_cp4d



class TestLale(unittest.TestCase):
    """
    Test Case Description:

    1. Get historical optimizer described by run_id or if run_id is None -> get last autoai optimizer.
    2. Get all generated pipelines as lale type and predict.
    3. Pipeline refinery with lale:
        * Remove last Pipeline stage.
        * Add three estimator choices as a last Pipeline stage.
        * Use HyperOpt to find best Pipeline (best last step estimator)
    4. Deploy lale pipeline
    """

    wml_credentials = None
    project_id = None
    space_id = None

    experiment = None
    lale_pipeline = None
    new_pipeline = None
    prefix = None
    hist_opt = None
    hyperopt_pipelines = None
    new_sklearn_pipeline = None
    service: 'WebService' = None
    
    X_data = None
    y_data = None

    run_id = None

    optimizer_name = None


    @classmethod
    def setUp(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        wrap_imported_operators()
        cls.wml_credentials = get_wml_credentials()

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        if is_cp4d():
            TestLale.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                                 project_id=self.project_id,
                                                 space_id=self.space_id)
        else:
            TestLale.experiment = AutoAI(wml_credentials=self.wml_credentials)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_get_run_id(self):
        if not self.run_id:
            # IF run id not set get last complete autoai run.
            runs_df = self.experiment.runs(filter=self.optimizer_name).list()

            self.assertGreater(len(runs_df), 0,
                               msg="No runs found for the WML instance. Please run any autoai experiment (or test) with the wml instance before rerun the test")

            for run in runs_df.values:
                if run[2] == 'completed':
                    TestLale.run_id = run[1]
                    break
            self.assertIsNotNone(TestLale.run_id,
                               msg="No completed runs found for the WML instance. Please run any autoai experiment (or test) with the wml instance before rerun the test")

        print("AutoAI Run Id: ", TestLale.run_id)

        self.assertIsNotNone(TestLale.run_id)
        self.assertIsInstance(TestLale.run_id, str)

    def test_03_get_optimizer(self):
        TestLale.hist_opt = self.experiment.runs.get_optimizer(run_id=self.run_id)
        print(f"Optimizer params: {TestLale.hist_opt.get_params()}")

        data = TestLale.hist_opt.get_data_connections()[0].read(with_holdout_split=False)
        TestLale.X_data = data.drop(TestLale.hist_opt.get_params()['prediction_column'], axis=1)
        TestLale.y_data = data[TestLale.hist_opt.get_params()['prediction_column']]

        self.assertIsNotNone(TestLale.X_data, msg="Source data set is None/empty")
        self.assertGreater(len(TestLale.X_data), 0, msg="Source data set is empty")

        print(TestLale.X_data.head(3))

    def test_04_get_default_pipeline_astype_lale(self):
        TestLale.lale_pipeline = self.hist_opt.get_pipeline()

        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")
        self.assertIsInstance(self.lale_pipeline, TrainablePipeline)

        predictions = self.lale_pipeline.predict(
            X=self.X_data.values[:5])

        print(predictions)
        self.assertGreater(len(predictions), 0,  msg="Returned prediction are empty")

    def test_05_get_all_pipelines_as_lale(self):
        summary = self.hist_opt.summary()
        print(summary)
        failed_pipelines = []
        for pipeline_name in summary.reset_index()['Pipeline Name']:
            print(f"Getting pipeline: {pipeline_name}")
            lale_pipeline = None
            try:
                lale_pipeline = self.hist_opt.get_pipeline(pipeline_name=pipeline_name)
                self.assertIsInstance(lale_pipeline, TrainablePipeline)
                predictions = lale_pipeline.predict(
                    X=self.X_data.values[:1])
                print(predictions)
                self.assertGreater(len(predictions), 0, msg=f"Returned prediction for {pipeline_name} are empty")
            except:
                print(f"Failure: {pipeline_name}")
                failed_pipelines.append(pipeline_name)
                traceback.print_exc()

            if not TestLale.lale_pipeline:
                TestLale.lale_pipeline = lale_pipeline
                print(f"{pipeline_name} loaded for next test cases")

        self.assertEqual(len(failed_pipelines), 0, msg=f"Some pipelines failed. Full list: {failed_pipelines}")

    def test_06_pipeline_to_script__lale__pretty_print(self):
        pipeline_code = self.lale_pipeline.pretty_print()
        try:
            exec(pipeline_code)

        except Exception as exception:
            self.assertIsNone(exception, msg="Pretty print from lale pipeline was not successful")

    ################################
    # Pipeline refinery with lale  #
    ################################

    def test_07_remove_last_freeze_trainable_prefix_returned(self):
        TestLale.prefix = self.lale_pipeline.remove_last().freeze_trainable()
        self.assertIsInstance(self.prefix, TrainablePipeline,
                              msg="Prefix pipeline is not of TrainablePipeline instance.")

    def test_08_add_estimator(self):
        TestLale.new_pipeline = self.prefix >> (LR | Tree | KNN)

    def test_09_hyperopt_fit_new_pipepiline(self):
        hyperopt = Hyperopt(estimator=TestLale.new_pipeline, cv=3, max_evals=2)
        try:
            TestLale.hyperopt_pipelines = hyperopt.fit(self.X_data.values, self.y_data.values)
        except ValueError as e:
            print(f"ValueError message: {e}")
            traceback.print_exc()
            hyperopt_results = hyperopt._impl._trials.results
            print(hyperopt_results)
            self.assertIsNone(e, msg="hyperopt fit was not successful")

    def test_10_get_pipeline_from_hyperopt(self):
        from sklearn.pipeline import Pipeline
        new_pipeline_model = TestLale.hyperopt_pipelines.get_pipeline()
        print(f"Hyperopt_pipeline_model is type: {type(new_pipeline_model)}")
        TestLale.new_sklearn_pipeline = new_pipeline_model.export_to_sklearn_pipeline()
        self.assertIsInstance(TestLale.new_sklearn_pipeline, Pipeline,
                              msg=f"Incorect Sklearn Pipeline type after conversion. Current: {type(TestLale.new_sklearn_pipeline)}")

    def test_11_predict_refined_pipeline(self):
        predictions = TestLale.new_sklearn_pipeline.predict(
            X=self.X_data.values[:1])
        print(predictions)
        self.assertGreater(len(predictions), 0, msg=f"Returned prediction for refined pipeline are empty")

    ############################
    # Deploy refined pipeline  #
    ############################

    def test_20_deployment_setup_and_preparation(self):
        if is_cp4d():
            TestLale.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
                                          source_project_id=self.project_id,
                                          source_space_id=self.space_id)
        else:
            TestLale.service = WebService(source_wml_credentials=self.wml_credentials)

    def test_21__deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
        deployment_name = TestLale.hist_opt.get_params()['name'] + " Deployment"
        model_to_deploy = import_from_sklearn_pipeline(self.new_sklearn_pipeline) \
            if self.new_sklearn_pipeline else self.lale_pipeline
        self.service.create(
            experiment_run_id=self.run_id,
            model=model_to_deploy,
            deployment_name=deployment_name)

    def test_22_score_deployed_model(self):
        nb_records = 5
        predictions = self.service.score(payload=self.X_data[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_23_list_deployments(self):
        self.service.list()
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_24_delete_deployment(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()


if __name__ == '__main__':
    unittest.main()
