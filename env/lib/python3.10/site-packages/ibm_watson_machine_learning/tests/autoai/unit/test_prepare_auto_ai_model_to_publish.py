#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment.autoai.runs import AutoPipelinesRuns
from ibm_watson_machine_learning.experiment.autoai.engines import WMLEngine
from ibm_watson_machine_learning.utils.autoai.utils import prepare_auto_ai_model_to_publish
from ibm_watson_machine_learning.helpers import get_credentials_from_config


class TestPrepareAutoaiModelToPublish(unittest.TestCase):
    cloud_run_details = None
    cp4d_run_details = None

    client_cloud = None
    client_cp4d = None

    wml_credentials_cloud = None
    wml_credentials_cp4d = None

    project_id = '6f6f3aab-1633-4ffa-8d95-1146dbfeb2e4'

    run_id_cp4d = '3c4f1e02-3af7-4b7e-8bab-c088acdf79d5'
    run_id_cloud = '0b8ccd16-7eab-43d6-bc03-26d53b94d4f7'

    model_cp4d = None
    model_cloud = None

    @classmethod
    def setUp(cls) -> None:
        cls.wml_credentials_cloud = get_credentials_from_config('CLOUD_PROD_AM', 'wml_credentials')
        cls.wml_credentials_cp4d = get_credentials_from_config('CLOUD_DEV_AM', 'wml_credentials')

        cls.client_cloud = APIClient(cls.wml_credentials_cloud)
        cls.client_cp4d = APIClient(cls.wml_credentials_cp4d, cls.project_id)

        cls.historical_runs_cloud = AutoPipelinesRuns(WMLEngine(wml_client=cls.client_cloud))
        cls.historical_runs_cp4d = AutoPipelinesRuns(WMLEngine(wml_client=cls.client_cp4d))

        cls.model_cloud = cls.historical_runs_cloud.get_optimizer(run_id=cls.run_id_cloud).get_pipeline(astype='sklearn')
        cls.model_cp4d = cls.historical_runs_cp4d.get_optimizer(run_id=cls.run_id_cp4d).get_pipeline(astype='sklearn')

    def test_01_cloud_model_preparation(self):
        cos_model_path = prepare_auto_ai_model_to_publish(
            pipeline_model=self.model_cloud,
            run_params=self.client_cloud.training.get_details(training_uid=self.run_id_cloud),
            run_id=self.run_id_cloud
        )
        print(cos_model_path)

    def test_02_cp4d_model_preparation(self):
        schema, artifact_name = prepare_auto_ai_model_to_publish(
            pipeline_model=self.model_cp4d,
            run_params=self.client_cp4d.training.get_details(training_uid=self.run_id_cp4d),
            run_id=self.run_id_cp4d,
            wml_client=self.client_cp4d
        )
        print(schema)
        print(artifact_name)

        self.assertEqual(artifact_name, 'artifact_auto_ai_model.tar.gz', msg="Name of the tar.gz file is different.")


if __name__ == '__main__':
    unittest.main()
