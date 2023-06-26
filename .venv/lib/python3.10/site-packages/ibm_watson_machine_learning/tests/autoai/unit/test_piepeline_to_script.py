#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

try:
    from sklearn.externals import joblib
except ImportError:
    import joblib

from ibm_watson_machine_learning.helpers import pipeline_to_script


class TestPipelineToScript(unittest.TestCase):
    model_path = './autoai/artifacts/bank/bank_model_333.pickle'
    pipeline = None

    @classmethod
    def setUp(cls) -> None:
        cls.pipeline = joblib.load(cls.model_path)

    def test_01__create_python_script(self):
        pipeline_to_script(pipeline=self.pipeline)

        script = None
        try:
            with open("pipeline_script.py") as f:
                script = f.read()

        except FileNotFoundError:
            self.assertIsNone(script, msg="pipeline_script.py file not found!")

        print(script)


if __name__ == '__main__':
    unittest.main()
