#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import io
import os
import abc
import time
import contextlib

from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.tests.base.abstract.abstract_client_test import AbstractClientTest


class AbstractDeploymentTest(AbstractClientTest, abc.ABC):
    """
    Abstract class implementing interface for a test of
    a deployment scenario using python-client
    """
    deployment_type = None
    deployment_name = None
    model_name = None
    software_specification_name = None
    model_props = None
    deployment_props = None
    model_path = "model.gz"

    model_id = None
    deployment_id = None

    IS_MODEL = True
    SPACE_ONLY = True

    @abc.abstractmethod
    def get_model(self):
        """
        Returns model which is going to be stored.
        It can be either:
         - A path to a saved object.
         - An object loaded to a current runtime.
        """
        pass

    @abc.abstractmethod
    def create_model_props(self):
        """
        Creates model_props required for model storing.
        """
        pass

    @abc.abstractmethod
    def create_deployment_props(self):
        """
        Creates deployment_props required for deployment creation.
        """
        pass

    @abc.abstractmethod
    def create_scoring_payload(self):
        """
        Creates payload for scoring which is performed on deployed model.
        """
        pass

    def try_delete_downloaded_model(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.model_path)

    # --- STORE + DEPLOY + SCORE MODEL ---

    def test_01_store_model(self):
        model_props = self.create_model_props()
        AbstractDeploymentTest.model = self.get_model()

        if self.IS_MODEL:
            model_details = self.wml_client.repository.store_model(
                meta_props=model_props,
                model=self.model
            )
            AbstractDeploymentTest.model_id = self.wml_client.repository.get_model_id(model_details)
            AbstractDeploymentTest.model_href = self.wml_client.repository.get_model_href(model_details)
        else:
            function_details = self.wml_client.repository.store_function(
                function=self.model,
                meta_props=model_props
            )
            AbstractDeploymentTest.model_id = self.wml_client.repository.get_function_id(function_details)
            AbstractDeploymentTest.model_href = self.wml_client.repository.get_function_href(function_details)
        self.assertIsNotNone(self.model_href)
        self.assertIsNotNone(self.model_id)

    def test_02_get_model_details(self):
        model_details = self.wml_client.repository.get_details(self.model_id)
        self.assertIsNotNone(model_details)
        self.assertEqual(self.model_name, model_details["metadata"]["name"])

        if model_details['entity'].get('schemas'):
            if model_details['entity'].get('label_column') and model_details['entity']['schemas'].get('input'):
                self.assertNotIn(model_details['entity']['label_column'], str(model_details['entity']['schemas']['input']))

    def test_03_list_repository(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.wml_client.repository.list()
            repository_list = buf.getvalue()

        self.assertIn(self.model_id, repository_list)

    def test_04_download_model(self):
        self.try_delete_downloaded_model()
        self.wml_client.repository.download(self.model_id, filename=self.model_path)
        self.try_delete_downloaded_model()

    def test_05_load_model(self):
        AbstractDeploymentTest.artifact_type = self.wml_client.repository._check_artifact_type(self.model_id)

        # TODO check loading a model for given libraries
        if False and self.artifact_type['model']:
            model = self.wml_client.repository.load(self.model_id)
            self.assertIsNotNone(model)

    def test_06_create_deployment(self):
        deployment_props = self.create_deployment_props()

        deployment_details = self.wml_client.deployments.create(
            artifact_uid=self.model_id,
            meta_props=deployment_props
        )
        AbstractDeploymentTest.deployment_id = self.wml_client.deployments.get_id(deployment_details)
        self.assertIsNotNone(self.deployment_id)

    def test_07_get_deployments_details(self):
        deployment_details = self.wml_client.deployments.get_details(self.deployment_id)
        self.assertIsNotNone(deployment_details)
        self.assertEqual(self.model_id, deployment_details["entity"]["asset"]["id"])

    def test_08_list_deployments(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.wml_client.deployments.list()
            deployments_list = buf.getvalue()

        self.assertIn(self.deployment_id, deployments_list)

    @abc.abstractmethod
    def test_09_download_deployment(self):
        """
        Method only supported for CoreML deployments.
        """
        pass

    @abc.abstractmethod
    def test_10_score_deployments(self):
        """
        Scoring is implemented separately  for different types of deployments.
        """
        pass

    # --- UPDATE MODEL + DEPLOYMENT ---

    def test_11_save_model_revision(self):
        if self.artifact_type['model']:
            new_model_revision = self.wml_client.repository.create_model_revision(self.model_id)
        elif self.artifact_type['function']:
            new_model_revision = self.wml_client.repository.create_function_revision(self.model_id)
        AbstractDeploymentTest.rev_id = new_model_revision['metadata'].get('rev')
        self.assertIsNotNone(self.rev_id)

    def test_12_list_revision(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            if self.artifact_type['model']:
                self.wml_client.repository.list_models_revisions(self.model_id, limit=50)
            elif self.artifact_type['function']:
                self.wml_client.repository.list_functions_revisions(self.model_id, limit=50)

            rev_list = buf.getvalue()

        self.assertIn(self.rev_id, rev_list)

    def test_13_update_model(self):
        updated_description = " Updated description"

        if self.artifact_type['model']:
            details = self.wml_client.repository.update_model(
                self.model_id,
                {
                    self.wml_client.repository.ModelMetaNames.DESCRIPTION: updated_description
                }
            )
        elif self.artifact_type['function']:
            details = self.wml_client.repository.update_function(
                self.model_id,
                {
                    self.wml_client.repository.FunctionMetaNames.DESCRIPTION: updated_description
                }
            )
        self.assertIsNotNone(details)

    def test_14_update_deployment(self):
        self.wml_client.deployments.update(
            AbstractDeploymentTest.deployment_id,
            {
                self.wml_client.deployments.ConfigurationMetaNames.ASSET: {
                    "id": self.model_id,
                    "rev": self.rev_id
                }
            }
        )

        status = None
        elapsed_time = 0
        wait_time = 5
        max_wait_time = 500
        while status not in ['ready', 'failed'] and elapsed_time < max_wait_time:
            time.sleep(wait_time)
            elapsed_time += 5
            deployment_details = self.wml_client.deployments.get_details(self.deployment_id)
            status = deployment_details['entity']['status'].get('state')
        self.assertEqual(status, "ready")
        self.assertEqual(self.rev_id, deployment_details['entity']['asset']['rev'])

        time.sleep(15)

    # --- DELETE DEPLOYMENT + MODEL ---

    def test_15_delete_deployment(self):
        self.wml_client.deployments.delete(self.deployment_id)

        # When getting details of non-existing deployment exception should be raised.
        with self.assertRaises(WMLClientError):
            self.wml_client.deployments.get_details(self.deployment_id)

    def test_16_delete_model(self):
        self.wml_client.repository.delete(self.model_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.repository.get_details(self.model_id)
