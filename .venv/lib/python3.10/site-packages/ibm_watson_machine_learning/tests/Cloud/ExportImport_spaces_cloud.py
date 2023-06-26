#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from ibm_watson_machine_learning.tests.Cloud.preparation_and_cleaning import *

class TestExportImport(unittest.TestCase):
    model_id1 = None
    model_id2 = None

    export_space_id = None
    import_space_id = None

    model_id_for_deployment = None
    deployment_id = None
    scoring_url = None

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestExportImport.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.cos_credentials = get_cos_credentials()
        self.cos_resource_crn = self.cos_credentials['resource_instance_id']
        self.instance_crn = get_instance_crn()

        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'space_export_' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description',
                     self.client.spaces.ConfigurationMetaNames.STORAGE: {
                                                                           "resource_crn": self.cos_resource_crn
                                                                        },
                     self.client.spaces.ConfigurationMetaNames.COMPUTE: {
                         "name": "existing_instance_id",
                         "crn": self.instance_crn
            }
        }

        self.space = self.client.spaces.store(meta_props=metadata, background_mode=False)

        TestExportImport.export_space_id = self.client.spaces.get_id(self.space)
        print("export space_id: ", TestExportImport.export_space_id)

        self.space_name = str(uuid.uuid4())

        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'space_import_' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description',
                     self.client.spaces.ConfigurationMetaNames.STORAGE: {
                                                                           "resource_crn": self.cos_resource_crn
                                                                        },
                     self.client.spaces.ConfigurationMetaNames.COMPUTE: {
                         "name": "existing_instance_id",
                         "crn": self.instance_crn
            }
        }

        self.space = self.client.spaces.store(meta_props=metadata, background_mode=False)

        TestExportImport.import_space_id = self.client.spaces.get_id(self.space)
        print("import space_id: ", TestExportImport.import_space_id)

    def test_01_publish_local_model_in_repository(self):

        self.client.set.default_space(TestExportImport.export_space_id)

        self.client.repository.ModelMetaNames.show()

        # model_content_path = 'artifacts/mnistCNN.h5.tgz'
        model_content_path = 'artifacts/tf_model_fvt_test.tar.gz'

        base_sw_spec_id = self.client.software_specifications.get_uid_by_name('default_py3.7')
        # base_sw_spec_id = self.client.software_specifications.get_id_by_name("ai-function_0.1-py3.6")

        print(base_sw_spec_id)

        meta_prop_sw_spec = {
            self.client.software_specifications.ConfigurationMetaNames.NAME: "test_sw_spec_" + str(uuid.uuid4()),
            self.client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for test",
            self.client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_id}
        }

        sw_spec_details = self.client.software_specifications.store(meta_props=meta_prop_sw_spec)

        print(sw_spec_details)

        TestExportImport.sw_spec_id = self.client.software_specifications.get_id(sw_spec_details)

        model_details = self.client.repository.store_model(model=model_content_path,
                                                           meta_props={self.client.repository.ModelMetaNames.NAME: "test_keras1",
                                                                  # self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: '2b73a275-7cbf-420b-a912-eae7f436e0bc',
                                                                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: TestExportImport.sw_spec_id,
                                                                       self.client.repository.ModelMetaNames.TYPE: 'tensorflow_2.1'})
        # self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: TestExportImport.sw_spec_id,
        #                                                           self.client.repository.ModelMetaNames.TYPE: 'keras_2.2.5'})
        print("model_id1: ", model_details)
        TestExportImport.model_id1 = self.client.repository.get_model_uid(model_details)

        print(TestExportImport.model_id1)

        self.assertIsNotNone(TestExportImport.model_id1)

        model_details = self.client.repository.store_model(model=model_content_path,
                                                           meta_props={self.client.repository.ModelMetaNames.NAME: "test_keras2",
                                                                  # self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: '2b73a275-7cbf-420b-a912-eae7f436e0bc',
                                                                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: TestExportImport.sw_spec_id,
                                                                       self.client.repository.ModelMetaNames.TYPE: 'tensorflow_2.1'})
        # self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: TestExportImport.sw_spec_id,
        #                                                           self.client.repository.ModelMetaNames.TYPE: 'keras_2.2.5'})
        print("model_id2: ", model_details)
        TestExportImport.model_id2 = self.client.repository.get_model_uid(model_details)

        print(TestExportImport.model_id2)

        self.assertIsNotNone(TestExportImport.model_id2)

    def test_02_deployment_of_model(self):

        deployment_details = self.client.deployments.create(TestExportImport.model_id1,
                                                            meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test keras to be exported deployment",
                                                                        self.client.deployments.ConfigurationMetaNames.ONLINE:{}})

        print(deployment_details)

        TestExportImport.deployment_id = self.client.deployments.get_id(deployment_details)
        TestExportImport.scoring_url = self.client.deployments.get_scoring_href(deployment_details)

        self.assertTrue('online' in str(deployment_details))

        deployment_details = self.client.deployments.get_details(deployment_uid=TestExportImport.deployment_id)

        print(deployment_details)

        self.client.deployments.delete(TestExportImport.deployment_id)


    def test_03_export(self):
        metadata = { self.client.export_assets.ConfigurationMetaNames.NAME: "export_model",
                     # self.client.export_assets.ConfigurationMetaNames.ASSET_TYPES: ["wml_function"],
                     self.client.export_assets.ConfigurationMetaNames.ASSET_IDS: [TestExportImport.model_id1,
                                                                                  TestExportImport.model_id2]
                     # self.client.export_assets.ConfigurationMetaNames.ASSET_TYPES: ["wml_model"]
         # self.client.export_assets.ConfigurationMetaNames.ALL_ASSETS: True
                     }
        # self.client.export_assets.ConfigurationMetaNames.ASSET_TYPES: ["wml_model"],
        # self.client.export_assets.ConfigurationMetaNames.ASSET_IDS: ["13a53931-a8c0-4c2f-8319-c793155e7517",
        #                                                             "13a53931-a8c0-4c2f-8319-c793155e7518"]
        #                                                            >> >}
        details = self.client.export_assets.start(meta_props=metadata, space_id=TestExportImport.export_space_id)
        print(details)

        TestExportImport.export_job_id = details[u'metadata'][u'id']

        import time

        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 10 * 60:
            time.sleep(3)
            response = self.client.export_assets.get_details(TestExportImport.export_job_id, space_id=TestExportImport.export_space_id)
            state = response[u'entity'][u'status'][u'state']
            print(state)
            if state == 'completed' or state == 'error' or state == 'failed':
                break
            diff_time = time.time() - start_time

        print(response)

        self.assertTrue(state == 'completed')

        details = self.client.export_assets.get_exported_content(TestExportImport.export_job_id,
             space_id = TestExportImport.export_space_id,
             file_path = '/Users/mitun_vb/export_import/data/my_exported_content.zip')

        print(details)

    def test_04_import(self):
        self.client.set.default_space(TestExportImport.import_space_id)
        self.client.repository.list_models()
        self.client.software_specifications.list()

        details = self.client.import_assets.start(file_path='/Users/mitun_vb/export_import/data/my_exported_content.zip',
                                                  space_id=TestExportImport.import_space_id)
        print(details)

        TestExportImport.import_job_id = details[u'metadata'][u'id']

        import time

        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 10 * 60:
            time.sleep(3)
            response = self.client.import_assets.get_details(TestExportImport.import_job_id,
                                                             space_id=TestExportImport.import_space_id)
            state = response[u'entity'][u'status'][u'state']
            print(state)
            if state == 'completed' or state == 'error' or state == 'failed':
                break
            diff_time = time.time() - start_time

        print(response)

        self.client.repository.list_models()
        self.client.software_specifications.list()
        details = self.client.repository.get_model_details()
        print(details)


        for obj in details[u'resources']:
            if obj[u'metadata'][u'name'] == "test_keras1":
                TestExportImport.model_id_for_deployment = obj[u'metadata'][u'id']

        print(TestExportImport.model_id_for_deployment)

    def test_05_deployment_of_imported_model(self):

        deployment_details = self.client.deployments.create(TestExportImport.model_id_for_deployment,
                                                            meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test keras imported deployment",
                                                                        self.client.deployments.ConfigurationMetaNames.ONLINE:{}})

        print(deployment_details)

        TestExportImport.deployment_id = self.client.deployments.get_id(deployment_details)
        TestExportImport.scoring_url = self.client.deployments.get_scoring_href(deployment_details)

        self.assertTrue('online' in str(deployment_details))

        deployment_details = self.client.deployments.get_details(deployment_uid=TestExportImport.deployment_id)

        print(deployment_details)

        self.client.deployments.delete(TestExportImport.deployment_id)

    # def test_05_delete_model(self):
    #     TestExportImport.logger.info("Delete model")
    #     self.client.repository.delete(TestExportImport.model_id1)
    #     self.client.repository.delete(TestExportImport.model_id2)

    def test_06_delete_space(self):
        TestExportImport.logger.info("Delete space and jobs")
        self.client.export_assets.list(space_id=TestExportImport.export_space_id)
        self.client.import_assets.list(space_id=TestExportImport.import_space_id)

        self.client.export_assets.delete(TestExportImport.export_job_id, space_id=TestExportImport.export_space_id)
        self.client.import_assets.delete(TestExportImport.import_job_id, space_id=TestExportImport.import_space_id)

        self.client.spaces.delete(TestExportImport.export_space_id)
        self.client.spaces.delete(TestExportImport.import_space_id)

if __name__ == '__main__':
    unittest.main()
