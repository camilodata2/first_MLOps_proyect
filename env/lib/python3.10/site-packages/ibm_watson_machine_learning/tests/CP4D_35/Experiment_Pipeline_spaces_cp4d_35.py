#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import sys
import io
import logging
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *



class TestWMLClientCloudWithExperiment(unittest.TestCase):
    deployment_id = None
    model_id = None
    scoring_url = None
    library_id = None
    library_url = None
    pipeline_id = None
    pipeline_url = None
    trained_model_id = None
    experiment_id = None
    experiment_run_id = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientCloudWithExperiment.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()




        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'client_space_' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description'
        }

        self.space = self.client.spaces.store(meta_props=metadata)

        TestWMLClientCloudWithExperiment.space_id = self.client.spaces.get_id(self.space)
        print("space_id: ", TestWMLClientCloudWithExperiment.space_id)
        self.client.set.default_space(TestWMLClientCloudWithExperiment.space_id)

    # @classmethod
    # def tearDownClass(self):
    #     clean_cos(self.cos_resource, self.bucket_names)


    def test_01_save_pipeline(self):
        TestWMLClientCloudWithExperiment.logger.info("Save pipelines ...")
        doc= {
             "doc_type": "pipeline",
             "version": "2.0",
             "primary_pipeline": "dlaas_only",
             "pipelines": [
               {
                 "id": "dlaas_only",
                 "runtime_ref": "hybrid",
                 "nodes": [
                   {
                     "id": "training",
                     "type": "model_node",
                     "op": "dl_train",
                     "runtime_ref": "DL",
                     "inputs": [
                     ],
                     "outputs": [],
                     "parameters": {
                       "name": "tf-mnist",
                       "description": "Simple MNIST model implemented in TF",
                       "command": "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000",
                     #  "training_lib_href": TestWMLClientCloudWithExperiment.library_url,
                       "compute_configuration_name": "k80",
                       "compute_configuration_nodes": 1,
                       "target_bucket": "wml-dev-results"
                     }
                   }
                 ]
               }
             ],
             "schemas": [
               {
                 "id": "schema1",
                 "fields": [
                   {
                     "name": "text",
                     "type": "string"
                   }
                 ]
               }
             ],
             "runtimes": [
               {
                 "id": "DL",
                 "name": "tensorflow",
                 "version": "1.13-py3.6"
                 }
             ]
            }

        metadata = {
            self.client.repository.PipelineMetaNames.NAME: "my_pipeline",
            self.client.repository.PipelineMetaNames.DOCUMENT: doc,
            self.client.repository.PipelineMetaNames.DESCRIPTION: "python client pipeline test",
            self.client.repository.PipelineMetaNames.TAGS : ["t1","t2"],
            self.client.repository.PipelineMetaNames.CUSTOM: {"value1": "custom test"}
            }


        definition_details = self.client.repository.store_pipeline( meta_props=metadata)
        print(definition_details)
        TestWMLClientCloudWithExperiment.pipeline_id = self.client.repository.get_pipeline_id(definition_details)
        TestWMLClientCloudWithExperiment.logger.info("Saved pipeline uid: " + str(TestWMLClientCloudWithExperiment.pipeline_id))
        print(definition_details)

    def test_02_get_pipeline_details(self):
        TestWMLClientCloudWithExperiment.logger.info("Getting pipeline details ...")
        details_1 = self.client.repository.get_pipeline_details(TestWMLClientCloudWithExperiment.pipeline_id)
        TestWMLClientCloudWithExperiment.logger.info(details_1)
        print(details_1)
        self.assertTrue('my_pipeline' in str(details_1))

    def test_03_list_pipelines(self):
        TestWMLClientCloudWithExperiment.logger.info("List pipelines")
        self.client.repository.list_pipelines()

    def test_04_get_id_url(self):
        def_details = self.client.repository.get_pipeline_details(TestWMLClientCloudWithExperiment.pipeline_id)
        uid = self.client.repository.get_pipeline_id(def_details)
        url = self.client.repository.get_pipeline_href(def_details)
        self.assertIsNotNone(uid)
        self.assertIsNotNone(url)

    def test_05_create_revision(self):
        details_1 = self.client.repository.create_pipeline_revision(TestWMLClientCloudWithExperiment.pipeline_id)
        TestWMLClientCloudWithExperiment.logger.info(details_1)
        print(details_1)
        self.assertTrue('my_pipeline' in str(details_1))

    def test_06_list_revision(self):
        details_1 = self.client.repository.list_pipelines_revisions(TestWMLClientCloudWithExperiment.pipeline_id)
        TestWMLClientCloudWithExperiment.logger.info(details_1)

    def test_07_get_revision_details(self):
        details_1 = self.client.repository.get_pipeline_revision_details(TestWMLClientCloudWithExperiment.pipeline_id,1)
        TestWMLClientCloudWithExperiment.logger.info(details_1)
        self.assertTrue('my_pipeline' in str(details_1))

    def test_08_update_pipeline(self):
        metadata = {
            self.client.repository.PipelineMetaNames.NAME: "updated_name",
            self.client.repository.PipelineMetaNames.DESCRIPTION: "python client pipeline updated description",
            self.client.repository.PipelineMetaNames.TAGS: ["t3"],
            self.client.repository.PipelineMetaNames.CUSTOM: {"value2": "custom test updated"}
        }

        updated_details = self.client.repository.update_pipeline(TestWMLClientCloudWithExperiment.pipeline_id,
                                                                      metadata)
        self.assertTrue('updated_name' in str(updated_details))
        TestWMLClientCloudWithExperiment.logger.info(updated_details)
        print(updated_details)
        self.assertTrue('value1' not in str(updated_details))

    def test_09_save_experiment(self):
        metadata = {
                    self.client.repository.ExperimentMetaNames.NAME: "xxx",
                    self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [{
                        "pipeline": {
                            "id": TestWMLClientCloudWithExperiment.pipeline_id,
                           # "id" : "acf894fd-c132-487d-9b42-7e417848bf5d"
                           # "href" : "/v4/pipelines/acf894fd-c132-487d-9b42-7e417848bf5d?rev=e07994ba-3d11-49b9-a6fa-9e64a23c34f8"
                        }
                    }
                    ]
                }


        experiment_details = self.client.repository.store_experiment(meta_props=metadata)
        print(experiment_details)

        TestWMLClientCloudWithExperiment.experiment_id = self.client.experiments.get_id(experiment_details)
        url = self.client.experiments.get_id(experiment_details)

        experiment_specific_details = self.client.experiments.get_details(TestWMLClientCloudWithExperiment.experiment_id)
        self.assertTrue(TestWMLClientCloudWithExperiment.experiment_id in str(experiment_specific_details))
        self.assertIsNotNone(url)

    def test_10_get_experiment_id_url(self):
        def_exp_details = self.client.repository.get_experiment_details(TestWMLClientCloudWithExperiment.experiment_id)
        uid = self.client.repository.get_experiment_id(def_exp_details)
        url = self.client.repository.get_experiment_href(def_exp_details)
        self.assertIsNotNone(uid)
        self.assertIsNotNone(url)

    def test_11_create_experiment_revision(self):
        details_1 = self.client.repository.create_experiment_revision(TestWMLClientCloudWithExperiment.experiment_id)
        TestWMLClientCloudWithExperiment.logger.info(details_1)
        print(details_1)
        self.assertTrue('xxx' in str(details_1))


    def test_12_update_experiment(self):
        metadata = {
            self.client.repository.ExperimentMetaNames.NAME: "my_experiment",
            self.client.repository.ExperimentMetaNames.DESCRIPTION: "mnist best model",
        }

        experiment_details = self.client.repository.update_experiment(TestWMLClientCloudWithExperiment.experiment_id, metadata)
        self.assertTrue('my_exp' in str(experiment_details))
        TestWMLClientCloudWithExperiment.logger.info(experiment_details)
        print(experiment_details)
        self.assertTrue('xxx' not in str(experiment_details))

    def test_13_list_experiment_revision(self):
        details_1 = self.client.repository.list_experiments_revisions(TestWMLClientCloudWithExperiment.experiment_id)
        TestWMLClientCloudWithExperiment.logger.info(details_1)

    def test_14_get_experiment_revision_details(self):
        details_1 = self.client.repository.get_experiment_revision_details(TestWMLClientCloudWithExperiment.experiment_id,1)
        TestWMLClientCloudWithExperiment.logger.info(details_1)
        self.assertTrue('xxx' in str(details_1))

    def test_15_list_experiment(self):
        # stdout_ = sys.stdout
        # captured_output = io.StringIO()  # Create StringIO object
        # sys.stdout = captured_output  # and redirect stdout.
        # self.client.experiments.list()# Call function.
        # sys.stdout = stdout_  # Reset redirect.
        # print(captured_output)
        # self.assertTrue(TestWMLClientCloudWithExperiment.experiment_id in captured_output.getvalue())
        self.client.experiments.list()  # Just to see values.

    # def test_07_save_training(self):
    #     metadata = {
    #
    #         self.client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {"name": "name123",
    #                                                                                 "connection": {
    #                                                                                     "endpoint_url": "https://s3-api.us-geo.objectstorage.softlayer.net",
    #                                                                                     "access_key_id": "zfho4HT7pUIStZvSkDsl",
    #                                                                                     "secret_access_key": "21q66Vvxkhr4uPDacTf8F9fnzMjSUIzsZRtxrYbx"
    #                                                                                 },
    #                                                                                 "location": {
    #                                                                                     "bucket": "fvt-training-results"
    #                                                                                 },
    #                                                                                 "type": "s3"
    #                                                                                 },
    #
    #         self.client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES:
    #             [{
    #                 "name": "training_input_data",
    #                 "type": "s3",
    #                 "connection": {
    #                     "endpoint_url": "https://s3-api.us-geo.objectstorage.softlayer.net",
    #                     "access_key_id": "zfho4HT7pUIStZvSkDsl",
    #                     "secret_access_key": "21q66Vvxkhr4uPDacTf8F9fnzMjSUIzsZRtxrYbx"
    #                 },
    #                 "location": {
    #                     "bucket": "wml-dev"
    #                 },
    #                 "schema": {
    #                     "id": "id123_schema",
    #                     "fields": [
    #                         {
    #                             "name": "text",
    #                             "type": "string"
    #                         }
    #                     ]
    #                 }
    #             }]
    #         ,
    #         self.client.training.ConfigurationMetaNames.EXPERIMENT_UID: TestWMLClientCloudWithExperiment.experiment_id
    #
    #     }
    #
    #
    #     training_details = self.client.training.run(meta_props=metadata)
    #
    #     TestWMLClientCloudWithExperiment.trained_model_id = self.client.training.get_id(training_details)
    #     url = self.client.training.get_id(training_details)
    #
    #     training_details = self.client.training.get_details(TestWMLClientCloudWithExperiment.trained_model_id)
    #     self.assertTrue(TestWMLClientCloudWithExperiment.trained_model_id in str(training_details))
    #     self.assertIsNotNone(url)
    #
    # def test_09_list_trainings(self):
    #     stdout_ = sys.stdout
    #     captured_output = io.StringIO()  # Create StringIO object
    #     sys.stdout = captured_output  # and redirect stdout.
    #     self.client.training.list()# Call function.
    #     sys.stdout = stdout_  # Reset redirect.
    #     self.assertTrue(TestWMLClientCloudWithExperiment.trained_model_id in captured_output.getvalue())
    #     self.client.training.list()  # Just to see values.
    #
    # def test_10_delete_training(self):
    #     self.client.training.delete(TestWMLClientCloudWithExperiment.trained_model_id)
    #

    def test_16_delete_experiment(self):
        self.client.repository.delete(TestWMLClientCloudWithExperiment.experiment_id)

    def test_17_delete_pipeline_library(self):
        self.client.repository.delete(TestWMLClientCloudWithExperiment.pipeline_id)


if __name__ == '__main__':
    unittest.main()
