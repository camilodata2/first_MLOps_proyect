#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from sklearn.externals.joblib import load
from lale.helpers import import_from_sklearn_pipeline
from lale.lib.sklearn import RandomForestClassifier, KNeighborsClassifier
from lale.lib.xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from lale.lib.lale import HyperoptCV
from utils.assertions import is_lale_pipeline_type
from lale.json_operator import to_json


class TestLalePipeline(unittest.TestCase):
    """
    Test Case Description:

    1. Load pipeline model from file, prepared by autoai service in Watson Machine Learning.
    2. Convert loaded pipeline model to LALE Pipeline.
    3. Remove last Pipeline stage.
    4. Add three estimator choices as a last Pipeline stage.
    5. Use HyperOpt to find best Pipeline (best last step estimator)

    """
    train_data = None
    train_target = None
    pipeline_model_path = f"./artifacts/BreastCancer/model.pickle"
    pipeline_model = None
    lale_pipeline = None
    new_planned_pipeline = None

    @classmethod
    def setUpClass(cls):
        """Setups training_data, training_target."""
        breast_cancer_data = load_breast_cancer()
        cls.train_data = breast_cancer_data.data
        cls.train_target = breast_cancer_data.target

    def test_01__load_auto_ai_pipeline_model__read_model_from_disk__model_correctly_loaded(self):
        TestLalePipeline.pipeline_model = load(TestLalePipeline.pipeline_model_path)

        print(f"Loaded Pipeline Model Steps: {TestLalePipeline.pipeline_model.steps}")

    def test_02__import_from_sklearn_pipeline__convert_loaded_model_to_lale_pipeline__lale_pipeline_created(self):
        TestLalePipeline.lale_pipeline = import_from_sklearn_pipeline(TestLalePipeline.pipeline_model)

        is_lale_pipeline_type(pipeline=TestLalePipeline.lale_pipeline)

        print(f"Converted Lale Pipeline Steps: {to_json(TestLalePipeline.lale_pipeline)}")

    def test_03__remove_last_pipeline_step__deleting_step__last_step_deleted(self):
        number_of_pipeline_steps = len(TestLalePipeline.lale_pipeline._steps)
        TestLalePipeline.lale_pipeline._steps.remove(TestLalePipeline.lale_pipeline._steps[-1])

        self.assertEqual(number_of_pipeline_steps - 1, len(TestLalePipeline.lale_pipeline._steps),
                         msg="Last step of the pipeline was not correctly removed.")

    def test_04__freeze_trainable__add_estimator_choices_as_a_last_pipeline_step__new_pipeline_created(self):
        TestLalePipeline.new_planned_pipeline = TestLalePipeline.lale_pipeline.freeze_trainable() >> (
                RandomForestClassifier | KNeighborsClassifier | XGBClassifier
        )

        print(f"New planned Pipeline info: "
              f"{to_json(TestLalePipeline.new_planned_pipeline.steps()[-1])}")

        self.assertIn('RandomForestClassifier', str(to_json(TestLalePipeline.new_planned_pipeline.steps()[-1])),
                      msg="RandomForestClassifier is not in the pipeline last step.")
        self.assertIn('KNeighborsClassifier', str(to_json(TestLalePipeline.new_planned_pipeline.steps()[-1])),
                      msg="KNeighborsClassifier is not in the pipeline last step.")
        self.assertIn('XGBClassifier', str(to_json(TestLalePipeline.new_planned_pipeline.steps()[-1])),
                      msg="XGBClassifier is not in the pipeline last step.")

    def test_05__auto_configure__use_hyepopt_to_find_best_pipeline__best_pipeline_found(self):
        TestLalePipeline.new_best_pipeline = TestLalePipeline.new_planned_pipeline.auto_configure(
            X=TestLalePipeline.train_data, y=TestLalePipeline.train_target,
            optimizer=HyperoptCV, scoring='roc_auc', cv=3, max_evals=2
        )

        print(f"Best lale pipeline: {to_json(TestLalePipeline.new_best_pipeline)}")


if __name__ == '__main__':
    unittest.main()
