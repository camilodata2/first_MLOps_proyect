#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import datetime,time
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *
import logging
from ibm_watson_machine_learning.spaces import Spaces
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.ICP.models_preparation import *


class TestWMLClientWithSpace(unittest.TestCase):
    space_uid = None
    space_href=None
    member_uid = None
    member_href = None

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSpace.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        # reload(site)
        self.client = get_client()
        version = self.client.version
        print(version)


    # def test_01_service_instance_details(self):
    #     TestWMLClientWithSpace.logger.info("Check client ...")
    #     self.assertTrue(self.client.__class__.__name__ == 'APIClient')
    #
    #     TestWMLClientWithSpace.logger.info("Getting instance details ...")
    #     details = self.client.service_instance.get_details()
    #     TestWMLClientWithSpace.logger.debug(details)
    #
    #     self.assertTrue("published_models" in str(details))
    #     self.assertEqual(type(details), dict)

    #create pipeline first
    def test_01_save_space(self):
        metadata = {
                    self.client.repository.SpacesMetaNames.NAME: "exports_space_01"
                }

        space_details = self.client.repository.store_space(meta_props=metadata)

        TestWMLClientWithSpace.space_uid = self.client.repository.get_space_uid(space_details)
        TestWMLClientWithSpace.space_href = self.client.repository.get_space_href(space_details)

        space_specific_details = self.client.repository.get_space_details(TestWMLClientWithSpace.space_uid)
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(space_specific_details))
        print(self.client.version)

    def test_02_set_space(self):
        set_space = self.client.set.default_space(TestWMLClientWithSpace.space_uid)


        self.assertTrue("SUCCESS" in str(set_space))

    def test_03_create_ai_function(self):

        self.client.repository.FunctionMetaNames.show()
        sw_spec_uid = self.client.software_specifications.get_uid_by_name("ai-function_0.1-py3.6")

        function_props = {
            self.client.repository.FunctionMetaNames.NAME: 'sample_function_with_sw',
            self.client.repository.FunctionMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid

        }

        def score(payload):
            payload = payload['input_data'][0]
            values = [[row[0] * row[1]] for row in payload['values']]
            return {'predictions': [{'fields': ['multiplication'], 'values': values}]}

        ai_function_details = self.client.repository.store_function(score, function_props)

        TestWMLClientWithSpace.function_uid = self.client.repository.get_function_uid(ai_function_details)
        function_url = self.client.repository.get_function_href(ai_function_details)
        TestWMLClientWithSpace.logger.info("AI function ID:" + str(TestWMLClientWithSpace.function_uid))
        TestWMLClientWithSpace.logger.info("AI function URL:" + str(function_url))
        self.assertIsNotNone(TestWMLClientWithSpace.function_uid)
        self.assertIsNotNone(function_url)

    def test_04_publish_scikit_model(self):
        # space = self.client.spaces.store({self.client.spaces.ConfigurationMetaNames.NAME: "test_case"})
        # space_id = self.client.spaces.get_uid(space)
        # self.client.set.default_space(space_id)
        # TestWMLClientWithScikitLearn.space_id = space_id
        print("The space is" + TestWMLClientWithSpace.space_uid)
        TestWMLClientWithSpace.logger.info("Creating scikit-learn model ...")

        model_data = create_scikit_learn_model_data()
        predicted = model_data['prediction']

        TestWMLClientWithSpace.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        sw_spec_uid = self.client.software_specifications.get_uid_by_name("scikit-learn_0.20-py3.6")

        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
                       self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
                       self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid
                       }
        published_model_details = self.client.repository.store_model(model=model_data['model'], meta_props=model_props,
                                                                     training_data=model_data['training_data'],
                                                                     training_target=model_data['training_target'])
        TestWMLClientWithSpace.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithSpace.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithSpace.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithSpace.model_url))
        self.assertIsNotNone(TestWMLClientWithSpace.model_uid)
        self.assertIsNotNone(TestWMLClientWithSpace.model_url)


    def test_05_get_space_details(self):
        details = self.client.repository.get_space_details()
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(details))

        details2 = self.client.repository.get_space_details(TestWMLClientWithSpace.space_uid)
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(details2))


    def test_06_create_export_for_space(self):
        self.logger.info("export space ...")

        assets = {"wml_function": [TestWMLClientWithSpace.model_uid],
                  "wml_model": [TestWMLClientWithSpace.function_uid]
                  }
        print(assets)

        export_spaces_props = {self.client.spaces.ExportMetaNames.NAME: "Export space " + TestWMLClientWithSpace.space_uid + " " ,
                       self.client.spaces.ExportMetaNames.ASSETS: assets

                       }

        space_exports_details = self.client.spaces.exports(TestWMLClientWithSpace.space_uid, meta_props=export_spaces_props)
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(space_exports_details))

        TestWMLClientWithSpace.space_exports_id = self.client.spaces.get_exports_uid(space_exports_details)
        self.assertIsNotNone(TestWMLClientWithSpace.space_exports_id)

        #TestWMLClientWithSpace.status = self.client.spaces.get_exports_details(TestWMLClientWithSpace.space_uid,TestWMLClientWithSpace.space_exports_id)
    def test_07_get_status(self):
        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 10 * 60:
            time.sleep(3)
            response = self.client.spaces.get_exports_details(TestWMLClientWithSpace.space_uid,TestWMLClientWithSpace.space_exports_id)
            if response['entity']['status']['state'] == 'completed' or response['entity']['status']['state'] == 'error' or response['entity']['status']['state'] == 'canceled':
                break

            diff_time = time.time() - start_time

        self.assertIsNotNone(response)
#        self.assertTrue(response['entity']['status']['state'] == 'completed')


    def test_08_download_export_for_space(self):

        # If the exports is completed and then download

        self.logger.info("download exported space ...")

        TestWMLClientWithSpace.filename = "wml_space_" + TestWMLClientWithSpace.space_uid + ".zip"

        self.client.spaces.download(TestWMLClientWithSpace.space_uid,
                                                               TestWMLClientWithSpace.space_exports_id,
                                                               filename=TestWMLClientWithSpace.filename)


    def test_09_import_downloaded_zip_to_new_space(self):
        metadata = {
            self.client.repository.SpacesMetaNames.NAME: "import_space_01"
        }

        imp_space_details = self.client.spaces.store(meta_props=metadata)

        TestWMLClientWithSpace.imp_space_uid = self.client.spaces.get_uid(imp_space_details)
        TestWMLClientWithSpace.imp_space_href = self.client.spaces.get_href(imp_space_details)

        space_specific_details = self.client.spaces.get_details(TestWMLClientWithSpace.imp_space_uid)
        self.assertTrue(TestWMLClientWithSpace.imp_space_uid in str(space_specific_details))


        import_space_details =  self.client.spaces.imports(TestWMLClientWithSpace.imp_space_uid,file_path=TestWMLClientWithSpace.filename)
        self.assertTrue(TestWMLClientWithSpace.imp_space_uid in str(import_space_details))

        TestWMLClientWithSpace.imports_uid = self.client.spaces.get_imports_uid(import_space_details)

        self.logger.info("Imported Space ID:" + str(TestWMLClientWithSpace.imports_uid))

        self.assertIsNotNone(TestWMLClientWithSpace.imports_uid)



    def test_10_check_if_assets_are_imported_correctly(self):

        set_space = self.client.set.default_space(TestWMLClientWithSpace.imp_space_uid)

        self.assertTrue("SUCCESS" in str(set_space))

        details_func = self.client.repository.get_function_details()

        if details_func['resources'] == []:
            print("functions not imported properly")
            self.assertIsNotNone(details_func['resources'])

        details_models = self.client.repository.get_model_details()
        if details_models['resources'] == []:
            print("models not imported properly")
            self.assertIsNotNone(details_models['resources'])
