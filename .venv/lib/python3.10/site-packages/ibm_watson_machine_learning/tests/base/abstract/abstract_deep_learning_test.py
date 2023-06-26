#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import io
import os
import abc
import contextlib

import ibm_boto3

from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.tests.utils import bucket_exists, create_bucket
from ibm_watson_machine_learning.tests.base.abstract.abstract_client_test import AbstractClientTest


class AbstractDeepLearningTest(AbstractClientTest, abc.ABC):
    """
    Class implementing interface of test checking a scenario
    of deep learning training.
    """
    model_definition_name = None
    training_name = None
    training_description = None
    software_specification_name = None
    execution_command = None

    training_id = None
    training_props = None
    data_location = None
    data_cos_path = None

    # If Test case is composed of multiple models pass the paths inside the \
    # List object [even if one model is required it should be passed wrapped in List].
    model_paths = None
    model_definition_ids = []

    SPACE_ONLY = False

    def create_model_definition_payload(self):
        """
        Creates payload for model definition.
        """
        return {
            self.wml_client.model_definitions.ConfigurationMetaNames.NAME: self.model_definition_name,
            self.wml_client.model_definitions.ConfigurationMetaNames.COMMAND: self.execution_command,
            self.wml_client.model_definitions.ConfigurationMetaNames.PLATFORM: {"name": "python", "versions": ["3.9"]},
            self.wml_client.model_definitions.ConfigurationMetaNames.VERSION: "2.0",
        }

    def create_training_payload(self):
        """
        Creates payload for training creation.
        """
        return {
            self.wml_client.training.ConfigurationMetaNames.NAME: self.training_name,
            self.wml_client.training.ConfigurationMetaNames.DESCRIPTION: self.training_description,
            self.wml_client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [
                {
                    "type": "connection_asset",
                    "connection": {
                        "id": self.connection_id,
                    },
                    "location": {
                        "bucket": self.bucket_name,
                        "file_name": "."
                    }
                }
            ],
            self.wml_client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE:  {
                "type": "connection_asset",
                "connection": {
                    "id": self.connection_id,
                },
                "location": {
                    "bucket": self.bucket_name,
                    "file_name": "."
                },
            },
            self.wml_client.training.ConfigurationMetaNames.MODEL_DEFINITION: {
                "id": self.definition_id,
                "hardware_spec": {
                  "name": "K80",
                  "nodes": 1
                },
                "software_spec": {
                  "name": self.software_specification_name
                },
            }
        }

    def test_01_prepare_COS_instance(self):
        AbstractDeepLearningTest.cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # Prepare bucket
        if not bucket_exists(self.cos_resource, self.bucket_name):
            AbstractDeepLearningTest.bucket_name = create_bucket(self.cos_resource, self.bucket_name)

            self.assertIsNotNone(self.bucket_name)
            self.assertTrue(bucket_exists(self.cos_resource, AbstractDeepLearningTest.bucket_name))

        if os.path.isfile(self.data_location):
            self.cos_resource.Bucket(self.bucket_name).upload_file(
                self.data_location,
                self.data_cos_path
            )
        else:
            for file in os.listdir(self.data_location):
                self.cos_resource.Bucket(self.bucket_name).upload_file(
                    os.path.join(self.data_location, file),
                    self.data_cos_path
                )

    def test_02_prepare_connection_to_COS(self):
        connection_details = self.wml_client.connections.create({
            'datasource_type': self.wml_client.connections.get_datasource_type_uid_by_name('bluemixcloudobjectstorage'),
            'name': 'Connection to COS for tests',
            'properties': {
                'bucket': self.bucket_name,
                'access_key': self.cos_credentials['cos_hmac_keys']['access_key_id'],
                'secret_key': self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                'iam_url': self.wml_client.service_instance._href_definitions.get_iam_token_url(),
                'url': self.cos_endpoint
            }
        })

        AbstractDeepLearningTest.connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.connection_id, str)

    def test_03_create_model_definition(self):
        for model_path in self.model_paths:
            meta_props = self.create_model_definition_payload()
            model_definitions_details = self.wml_client.model_definitions.store(
                model_path, meta_props
            )
            model_definition_id = self.wml_client.model_definitions.get_id(
                model_definitions_details
            )
            self.assertIsNotNone(model_definition_id)
            self.model_definition_ids.append(model_definition_id)

        self.assertEqual(len(self.model_definition_ids), len(self.model_paths))

    def test_04_get_model_definition_details(self):
        for id_ in self.model_definition_ids:
            details = self.wml_client.model_definitions.get_details(id_)
            self.assertIsNotNone(details)
            self.assertEqual(self.model_definition_name, details["metadata"]["name"])

    def test_05_list_definitions(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.wml_client.model_definitions.list()
            definitions_list = buf.getvalue()

        self.assertTrue(all(id_ in definitions_list for id_ in self.model_definition_ids))

    def test_06_create_definition_revision(self):
        AbstractDeepLearningTest.definition_id = self.model_definition_ids[0]
        revision = self.wml_client.model_definitions.create_revision(self.definition_id)
        AbstractDeepLearningTest.rev_id = revision['metadata'].get('revision_id')
        self.assertIsNotNone(self.rev_id)

    def test_07_list_definition_revisions(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.wml_client.model_definitions.list_revisions(self.definition_id)
            rev_list = buf.getvalue()

        self.assertIn(str(self.rev_id), rev_list)

    def test_08_update_model_definition(self):
        details = self.wml_client.model_definitions.update(
            self.definition_id,
            {
                self.wml_client.model_definitions.ConfigurationMetaNames.NAME: "Updated model definition"
            }
        )
        self.assertIsNotNone(details)


    def test_09_download_model_definition_for_given_rev_id(self):
        definition_path = 'model.tar.gz'
        with contextlib.suppress(FileNotFoundError):
            os.remove(definition_path)

            self.wml_client.model_definitions.download(
                model_definition_uid=self.definition_id, filename=definition_path, rev_id=self.rev_id
            )

            os.remove(definition_path)

    def test_10_run_traininng(self):
        meta_props = self.create_training_payload()
        training_details = self.wml_client.training.run(meta_props, asynchronous=False)

        AbstractDeepLearningTest.training_id = self.wml_client.training.get_id(training_details)
        self.assertIsNotNone(self.training_id)

        state = self.wml_client.training.get_status(self.training_id).get('state')
        self.assertEqual(state, 'completed')

    def test_11a_get_training_details(self):
        details = self.wml_client.training.get_details(self.training_id)
        self.assertIsNotNone(details)
        self.assertEqual(self.training_name, details["metadata"]["name"])

    def test_11b_get_trainings_details(self):
        data = self.wml_client.training.get_details(asynchronous=False, get_all=True, limit=10)
        for synch_entity, asynch_entity in zip(
                data, self.wml_client.training.get_details(asynchronous=False, get_all=True, limit=10)):
            print(asynch_entity)
            self.assertIsNotNone(asynch_entity)
            self.assertEqual(synch_entity, asynch_entity)

    def test_12_get_training_list(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.wml_client.training.list(limit=200)
            training_list = buf.getvalue()

        # TODO check listing
        # self.assertIn(self.training_id, training_list)

    def test_13_hard_delete_training(self):
        self.wml_client.training.cancel(self.training_id, hard_delete=True)

    def test_14_delete_model_definitions(self):
        for id_ in self.model_definition_ids:
            self.wml_client.model_definitions.delete(id_)

        with self.assertRaises(WMLClientError):
            self.wml_client.model_definitions.get_details(self.definition_id)


class AbstractDeepLearningExperimentTest(AbstractDeepLearningTest, abc.ABC):
    """
    Class implementing interface of test checking a scenario
    of deep learning training using experiment.
    """
    experiment_name = None

    def create_experiment_payload(self):
        """
        Creates payload for experiment.
        """
        return {
            self.wml_client.repository.ExperimentMetaNames.NAME: self.experiment_name,
            self.wml_client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                {
                    "model_definition": {
                        "id": self.definition_id,
                        "hardware_spec": {
                            "name": "K80",
                            "nodes": 1
                        },
                        "software_spec": {
                            "name": self.software_specification_name
                        }
                    }
                }
            ]
        }

    def create_training_payload(self):
        """
        Overrides payload for training creation.
        It's using experiment instead of model definition.
        """
        return {
            self.wml_client.training.ConfigurationMetaNames.NAME: self.training_name,
            self.wml_client.training.ConfigurationMetaNames.DESCRIPTION: self.training_description,
            self.wml_client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [
                {
                    "type": "connection_asset",
                    "connection": {
                        "id": self.connection_id,
                    },
                    "location": {
                        "bucket": self.bucket_name,
                        "file_name": "."
                    }
                }
            ],
            self.wml_client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE:  {
                "type": "connection_asset",
                "connection": {
                    "id": self.connection_id,
                },
                "location": {
                    "bucket": self.bucket_name,
                    "file_name": "."
                },
            },
            self.wml_client.training.ConfigurationMetaNames.EXPERIMENT: {
                "id": self.experiment_id
            }
        }

    def test_09a_store_experiment(self):
        meta_props = self.create_experiment_payload()
        experiment_details = self.wml_client.experiments.store(meta_props)
        AbstractDeepLearningExperimentTest.experiment_id = \
            self.wml_client.experiments.get_id(experiment_details)

    def test_09b_get_experiment_details(self):
        details = self.wml_client.experiments.get_details(self.experiment_id)
        self.assertIsNotNone(details)
        self.assertEqual(self.experiment_name, details["metadata"]["name"])

    def test_09c_list_experiments(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.wml_client.experiments.list()
            experiments_list = buf.getvalue()

        self.assertIn(self.experiment_id, experiments_list)

    def test_09d_create_experiment_revision(self):
        revision = self.wml_client.experiments.create_revision(self.experiment_id)
        AbstractDeepLearningTest.rev_id = revision['metadata'].get('rev')
        self.assertIsNotNone(self.rev_id)

    def test_09e_list_experiment_revisions(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            self.wml_client.experiments.list_revisions(self.experiment_id)
            rev_list = buf.getvalue()

        self.assertIn(self.rev_id, rev_list)

    def test_09f_update_experiment(self):
        updated_name = "Updated DL experiment"
        details = self.wml_client.experiments.update(
            self.experiment_id,
            {
                self.wml_client.experiments.ConfigurationMetaNames.NAME: updated_name
            }
        )

        self.assertIsNotNone(details)
        self.assertEqual(updated_name, details["metadata"]["name"])
