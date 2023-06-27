#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import io
import abc
import time
import contextlib

from ibm_watson_machine_learning.tests.base.abstract.abstract_deployment_test import AbstractDeploymentTest



class AbstractBatchDeploymentTest(AbstractDeploymentTest, abc.ABC):
    """
    Abstract class implementing scoring using batch deployment jobs.
    """
    def create_deployment_props(self):
        return {
            self.wml_client.deployments.ConfigurationMetaNames.NAME: self.deployment_name,
            self.wml_client.deployments.ConfigurationMetaNames.BATCH: {},
            self.wml_client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {"name": "S", "num_nodes": 1}
        }

    def test_09_download_deployment(self):
        pass

    def test_10_score_deployments(self):
        job_payload = self.create_scoring_payload()
        job_details = self.wml_client.deployments.create_job(self.deployment_id, job_payload)

        AbstractBatchDeploymentTest.job_id = self.wml_client.deployments.get_job_uid(job_details)
        self.assertIsNotNone(self.job_id)

    def test_10b_list_jobs(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.wml_client.deployments.list_jobs()
            jobs_list = buf.getvalue()

        self.assertIn(self.job_id, jobs_list)

    def test_10b_get_job_details(self):
        status = None
        elapsed_time = 0
        wait_time = 5
        max_wait_time = 500
        while status not in ['completed', 'failed'] and elapsed_time < max_wait_time:
            time.sleep(wait_time)
            elapsed_time += wait_time
            status = self.wml_client.deployments.get_job_status(self.job_id).get('state')
        self.assertEqual(status, "completed")
