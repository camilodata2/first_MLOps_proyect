#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

import ibm_boto3
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import DataConnection, ContainerLocation
from ibm_watson_machine_learning.tests.utils import is_cp4d, save_data_to_container
from ibm_watson_machine_learning.tests.autoai.abstract_tests_classes import AbstractTestWebservice, \
    AbstractTestAutoAIAsync, AbstractTestBatch

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, Metrics


class TestAutoAIRemote(AbstractTestAutoAIAsync, AbstractTestWebservice, AbstractTestBatch, unittest.TestCase):
    """
    The test can be run on CLOUD, and CPD
    """

    cos_resource = None
    data_location = './autoai/data/Corona_NLP_test_utf8.csv'

    data_cos_path = 'data/Corona_NLP_test_utf8.csv'

    SPACE_ONLY = True

    OPTIMIZER_NAME = "Corona NLP Text Transformer test sdk"

    batch_payload_location = './autoai/data/scoring_payload/Corona_NLP_scoring_payload.csv'
    batch_payload_cos_location = "scoring_payload/Corona_NLP_scoring_payload.csv"

    BATCH_DEPLOYMENT_WITH_CA = False
    BATCH_DEPLOYMENT_WITH_CDA = False
    BATCH_DEPLOYMENT_WITH_DA = True
    BATCH_DEPLOYMENT_WITH_DF = True

    target_space_id = None

    experiment_info = dict(
        name=OPTIMIZER_NAME,
        prediction_type=PredictionType.MULTICLASS,
        prediction_column='Sentiment',
        scoring=Metrics.F1_SCORE_MACRO,
        holdout_size=0.1,
        text_processing=True,
        text_columns_names=['OriginalTweet', 'Location'],
        word2vec_feature_number=4,
        max_number_of_estimators=1,
        daub_give_priority_to_runtime=3.0,
        retrain_on_holdout=False
    )

    def test_00b_write_data_to_container(self):
        if self.SPACE_ONLY:
            self.wml_client.set.default_space(self.space_id)
        else:
            self.wml_client.set.default_project(self.project_id)

        save_data_to_container(self.data_location, self.data_cos_path, self.wml_client)

    def test_02_data_reference_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            location=ContainerLocation(path=self.data_cos_path
                                       ))
        TestAutoAIRemote.results_connection = None

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    def test_06_get_train_data(self):
        X_train, X_holdout, y_train, y_holdout = self.remote_auto_pipelines.get_data_connections()[0].read(
            with_holdout_split=True)

        print("train data sample:")
        print(X_train)
        print(y_train)
        print("holdout data sample:")
        print(X_holdout)
        print(y_holdout)

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_holdout), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_holdout), len(y_holdout))

        AbstractTestAutoAIAsync.X_df = X_holdout
        AbstractTestAutoAIAsync.X_values = AbstractTestAutoAIAsync.X_df.values
        AbstractTestAutoAIAsync.y_values = y_holdout

    # def test_10b_check_holdout_split_scores(self):
    #     from sklearn.metrics import get_scorer
    #
    #     scorer = get_scorer(self.experiment_info['scoring'])
    #
    #     summary = self.remote_auto_pipelines.summary()
    #     print(summary)
    #
    #     holdout_scoring_column = 'holdout_'
    #
    #     failed_pipelines = []
    #     error_messages = []
    #
    #     for pipeline_name in summary.reset_index()['Pipeline Name']:
    #         print(f"Getting pipeline: {pipeline_name}")
    #         try:
    #             pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name=pipeline_name)
    #             score = scorer(pipeline, self.X_values, self.y_values)
    #
    #             print(predictions)
    #             test_case.assertGreater(len(predictions), 0, msg=f"Returned prediction for {pipeline_name} are empty")
    #         except:
    #             print(f"Failure: {pipeline_name}")
    #             failed_pipelines.append(pipeline_name)
    #             error_message = traceback.format_exc()
    #             print(error_message)
    #             error_messages.append(error_message)
    #
    #     test_case.assertEqual(len(failed_pipelines), 0, msg=f"Some pipelines failed. Full list: {failed_pipelines} \n "
    #                                                         f"Errors: {error_messages}")

    def test_08_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        print(parameters)
        self.assertIsNotNone(parameters)

    def test_08a_get_feature_importance(self):
        feature_importance_df = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_1').get('features_importance')
        self.assertIsNotNone(feature_importance_df)

        str_feature_importance_index_list = str(list(feature_importance_df.sort_index().index))

        text_columns = self.experiment_info['text_columns_names']

        for column in text_columns:
            self.assertIn(f'word2vec({column})', str_feature_importance_index_list,
                          msg=f"word2vec({column}) is not in features importance table. Full table: {feature_importance_df}")

        self.assertIn('NewTextFeature_0', str_feature_importance_index_list,
                      msg="Text features were numerated incorrectly.")
        self.assertIn(f"NewTextFeature_{self.experiment_info['word2vec_feature_number']*len(text_columns)-1}",
                      str_feature_importance_index_list, msg="Text features were numerated incorrectly.")





if __name__ == '__main__':
    unittest.main()
