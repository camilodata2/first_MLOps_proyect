#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from models_preparation import *
from preparation_and_cleaning import *
import logging


class TestWMLClientWithScikitLearn(unittest.TestCase):
    model_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithScikitLearn.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.project_id = os.environ['PROJECT_ID']
        self.client.set.default_project(self.project_id)

    def test_01_publish_model(self):
        TestWMLClientWithScikitLearn.logger.info("Creating scikit-learn model ...")

        model_data = create_scikit_learn_model_data()
        predicted = model_data['prediction']

        TestWMLClientWithScikitLearn.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "ScikitModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "scikit-learn_0.20-py3.6"
                       }
        published_model_details = self.client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], training_target=model_data['training_target'])
        TestWMLClientWithScikitLearn.model_uid = self.client.repository.get_model_uid(published_model_details)
       # TestWMLClientWithScikitLearn.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithScikitLearn.model_uid))
        #self.logger.info("Published model URL:" + str(TestWMLClientWithScikitLearn.model_url))
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_uid)
        #self.assertIsNotNone(TestWMLClientWithScikitLearn.model_url)

    def test_03_download_model(self):
        TestWMLClientWithScikitLearn.logger.info("Download model")
        try:
            os.remove('download_test_url')
        except OSError:
            pass

        try:
            file = open('download_test_uid', 'r')
        except IOError:
            file = open('download_test_uid', 'w')
            file.close()

        self.client.repository.download(TestWMLClientWithScikitLearn.model_uid, filename='download_test_url')
        try:
            os.remove('test_ai_function.tar.gz')
        except:
            pass
    def test_04_get_details(self):
        TestWMLClientWithScikitLearn.logger.info("Get model details")
        details = self.client.repository.get_details(self.model_uid)
        TestWMLClientWithScikitLearn.logger.debug("Model details: " + str(details))
        self.assertTrue("ScikitModel" in str(details))

    def test_07_delete_model(self):
        TestWMLClientWithScikitLearn.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithScikitLearn.model_uid)
       # self.client.spaces.delete(TestWMLClientWithScikitLearn.space_id)


if __name__ == '__main__':
    unittest.main()