#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import abc
import uuid
import time
from os import environ

import unittest

from sklearn.pipeline import Pipeline

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.utils.deployment.errors import ServingNameNotAvailable
from ibm_watson_machine_learning.workspace import WorkSpace
from ibm_watson_machine_learning.helpers.connections import DataConnection
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, get_space_id,
                                                     is_cp4d)
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup, delete_model_deployment
from ibm_watson_machine_learning.utils.autoai.enums import TShirtSize, PredictionType

from ibm_watson_machine_learning.tests.utils.assertions import get_and_predict_all_pipelines_as_lale


class AbstractTestWebservice(abc.ABC):
    """
    The abstract tests which covers:
    - deployment with lale pipeline
    - deployment deletion
    In order to execute test connection definitions must be provided
    in inheriting classes.
    """
    SPACE_ONLY = True

    DEPLOYMENT_NAME = "SDK tests Deployment"

    wml_client: 'APIClient' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    service: 'WebService' = None

    space_id = None
    project_id = None
    target_space_id = None

    X_df = None
    uid = str(uuid.uuid4())[:12].replace('-', "")
    initial_deployment_serving_name = 'depl_serving_name_init_' + uid
    deployment_serving_name = 'depl_serving_name_' + uid
    deployment_serving_name_patch = 'depl_serving_name_patch_' + uid

    get_all_result = 0

    @abc.abstractmethod
    def test_00a_space_cleanup(self):
        pass

    ###########################################################
    #      DEPLOYMENT SECTION    tests numbers start from 31  #
    ###########################################################

    def test_31_deployment_setup_and_preparation(self):
        # note: if target_space_id is not set, use the space_id
        if self.target_space_id is None:
            self.target_space_id = self.space_id
        # end note

        if self.SPACE_ONLY:
            AbstractTestWebservice.service = WebService(source_wml_credentials=self.wml_credentials,
                                                        source_space_id=self.space_id,
                                                        target_wml_credentials=self.wml_credentials,
                                                        target_space_id=self.target_space_id)
        else:
            AbstractTestWebservice.service = WebService(source_wml_credentials=self.wml_credentials,
                                                        source_project_id=self.project_id,
                                                        target_wml_credentials=self.wml_credentials,
                                                        target_space_id=self.target_space_id)

        self.wml_client.set.default_space(self.space_id)
        delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)

        self.assertIsInstance(AbstractTestWebservice.service, WebService, msg="Deployment is not of WebService type.")
        self.assertIsInstance(AbstractTestWebservice.service._source_workspace, WorkSpace, msg="Workspace set incorrectly.")
        self.assertEqual(AbstractTestWebservice.service.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(AbstractTestWebservice.service.name, None, msg="Deployment name initialized incorrectly")

    def test_32__deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
        best_pipeline = self.remote_auto_pipelines.summary()._series['Enhancements'].keys()[0]
        print('Deploying', best_pipeline)

        self.assertTrue(self.wml_client.deployments.is_serving_name_available(
            serving_name=AbstractTestWebservice.initial_deployment_serving_name))

        AbstractTestWebservice.service.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model=best_pipeline,
            deployment_name=self.DEPLOYMENT_NAME,
            serving_name=AbstractTestWebservice.initial_deployment_serving_name
        )

        self.assertFalse(self.wml_client.deployments.is_serving_name_available(
            serving_name=AbstractTestWebservice.initial_deployment_serving_name))

        with self.assertRaises(ServingNameNotAvailable):
            service = WebService(source_wml_credentials=self.wml_credentials,
                                 source_space_id=self.space_id,
                                 target_wml_credentials=self.wml_credentials,
                                 target_space_id=self.space_id)

            service.create(
                experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
                model=best_pipeline,
                deployment_name=self.DEPLOYMENT_NAME,
                serving_name=AbstractTestWebservice.initial_deployment_serving_name
            )

        self.assertIsNotNone(AbstractTestWebservice.service.id, msg="Online Deployment creation - missing id")
        self.assertIsNotNone(AbstractTestWebservice.service.name, msg="Online Deployment creation - name not set")
        self.assertIsNotNone(AbstractTestWebservice.service.scoring_url,
                             msg="Online Deployment creation - mscoring url  missing")

    def test_32b_serving_name_test(self):
        self.target_space_id = self.space_id
        self.wml_client.set.default_space(self.target_space_id)

        time.sleep(1) #wait for deployment update
        deployment_details = self.wml_client.deployments.get_details(serving_name='non_existing')
        self.assertTrue(len(deployment_details['resources']) == 0)

        self.wml_client.deployments.update(AbstractTestWebservice.service.deployment_id, {
            self.wml_client.deployments.ConfigurationMetaNames.SERVING_NAME: self.deployment_serving_name})

        time.sleep(1)  # wait for deployment update
        deployment_details = self.wml_client.deployments.get_details(serving_name=self.deployment_serving_name)
        self.assertTrue(len(deployment_details['resources']) == 1, f"Found more deployment for the serving name: {deployment_details['resources']}")

        self.assertTrue(
            str(self.wml_client.deployments.get_serving_href(deployment_details['resources'][0])).endswith(
                f'/{self.deployment_serving_name}/predictions'))

        self.wml_client.deployments.update(AbstractTestWebservice.service.deployment_id,
                                           {'serving_name': self.deployment_serving_name_patch})

        time.sleep(1)  # wait for deployment update
        deployment_details = self.wml_client.deployments.get_details(serving_name=self.deployment_serving_name)
        self.assertTrue(len(deployment_details['resources']) == 0)

        deployment_details = self.wml_client.deployments.get_details(
            serving_name=self.deployment_serving_name_patch)
        self.assertTrue(len(deployment_details['resources']) == 1)

        self.assertTrue(
            str(self.wml_client.deployments.get_serving_href(deployment_details['resources'][0])).endswith(
                f'/{self.deployment_serving_name_patch}/predictions'))

    def test_32b_get_details_as_generator(self):
        limit = 2
        deployments_details = []
        for entry in self.wml_client.deployments.get_details(limit=limit, asynchronous=True):
            #self.assertLessEqual(len(entry['resources']), 2)
            deployments_details.extend(entry['resources'])

        AbstractTestWebservice.get_all_result = len(deployments_details)

    def test_32b_get_details_as_generator_get_all(self):
        limit = 2
        deployments_details = []
        for entry in self.wml_client.deployments.get_details(limit=limit, asynchronous=True, get_all=True):
            #self.assertLessEqual(len(entry['resources']), 2)
            deployments_details.extend(entry['resources'])

        self.assertEqual(self.get_all_result, len(deployments_details))

    def test_32b_get_details_get_all_with_limit(self):
        limit = 2
        deployments_details = self.wml_client.deployments.get_details(limit=limit, get_all=True)
        #self.assertLessEqual(len(entry['resources']), 2)

        self.assertEqual(self.get_all_result, len(deployments_details['resources']))

    def test_32b_get_details_get_all(self):
        deployments_details = self.wml_client.deployments.get_details(get_all=True)
        self.assertEqual(self.get_all_result, len(deployments_details['resources']))

    def test_33_score_deployed_model(self):
        nb_records = 5
        predictions = AbstractTestWebservice.service.score(payload=self.X_df[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_34_list_deployments(self):
        AbstractTestWebservice.service.list()
        params = AbstractTestWebservice.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_35_delete_deployment(self):
        print("Delete current deployment: {}".format(AbstractTestWebservice.service.deployment_id))
        AbstractTestWebservice.service.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(AbstractTestWebservice.service.asset_id)
        self.wml_client.set.default_project(self.project_id) if is_cp4d() else None
        self.assertEqual(AbstractTestWebservice.service.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(AbstractTestWebservice.service.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(AbstractTestWebservice.service.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")
