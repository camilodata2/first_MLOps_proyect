#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from sklearn.pipeline import Pipeline
from pprint import pprint
import pandas as pd
import traceback
from os import environ

from ibm_watson_machine_learning import APIClient

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import Batch
from ibm_watson_machine_learning.preprocessing import DataJoinGraph
from ibm_watson_machine_learning.preprocessing.data_join_pipeline import OBMPipelineGraphBuilder

from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, FSLocation, \
    DeploymentOutputAssetLocation, AssetLocation

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines

from ibm_watson_machine_learning.tests.utils import print_test_separators
from ibm_watson_machine_learning.utils.autoai.errors import WrongDataJoinGraphNodeName, OBMMainNodeAlreadySet

from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d, bucket_exists, \
    create_bucket, get_space_id

from ibm_watson_machine_learning.tests.utils.cleanup import delete_model_deployment, space_cleanup



from ibm_watson_machine_learning.tests.utils.cleanup import delete_model_deployment


get_step_details = OBMPipelineGraphBuilder.get_step_details
get_join_extractors = OBMPipelineGraphBuilder.get_join_extractors
get_extractor_columns = OBMPipelineGraphBuilder.get_extractor_columns
get_extractor_transformations = OBMPipelineGraphBuilder.get_extractor_transformations


class TestOBMAndKB(unittest.TestCase):
    data_join_graph: 'DataJoinGraph' = None
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    historical_opt: 'RemoteAutoPipelines' = None
    service: 'Batch' = None
    data_join_pipeline: 'DataJoinPipeline' = None

    data_location = './autoai/data/temperature_dataset/'

    trained_pipeline_details = None
    run_id = None

    data_connections = []
    data_connections_space_only = []
    main_conn = None
    temp_2_conn = None
    temp_3_conn = None
    temp_4_conn = None

    results_connection = None

    prediction_column = 'temp>=31'

    train_data = None

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-obm-qa")
    pod_version = environ.get('KB_VERSION', None)
    space_name = environ.get('SPACE_NAME', 'regression_tests_sdk_space')

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    cos_resource_instance_id = None
    data_cos_path = 'data/'

    results_cos_path = 'results_wml_autoai'

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None

    OPTIMIZER_NAME = 'OBM + KB test'
    DEPLOYMENT_NAME = "OBM + KB deployment test"

    job_id = None


    # note: notebook cos version:
    metadata = None
    local_optimizer_cos = None
    model_location = None
    training_status =None
    pipeline_name =None
    # --- end note

    asset_id = None
    space_id = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()

        if not is_cp4d():
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())
        # cls.project_id = cls.wml_credentials.get('project_id')

    @print_test_separators
    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestOBMAndKB.space_id = get_space_id(self.wml_client, self.space_name,
                                     cos_resource_instance_id=self.cos_resource_instance_id)

        self.wml_client.set.default_space(self.space_id)

    @print_test_separators
    def test_00b_prepare_COS_instance(self):
        if self.wml_client.ICP or self.wml_client.WSD:
            self.skipTest("Prepare COS is available only for Cloud")

        import ibm_boto3
        cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # Prepare bucket
        if not bucket_exists(cos_resource, self.bucket_name):
            TestOBMAndKB.bucket_name = create_bucket(cos_resource, self.bucket_name)

            self.assertIsNotNone(TestOBMAndKB.bucket_name)
            self.assertTrue(bucket_exists(cos_resource, TestOBMAndKB.bucket_name))

        print(f"Using COS bucket: {TestOBMAndKB.bucket_name}")

    @print_test_separators
    def test_01_create_multiple_data_connections__connections_created(self):
        print("Creating multiple data connections...")
        if self.wml_client.ICP:
            self.wml_client.set.default_space(self.space_id)
            for dataset_no in range(1, 5):
                conn = DataConnection(
                    data_join_node_name=f"temp_{dataset_no}",
                    location=FSLocation(path=self.data_location + f'temp_{dataset_no}.csv'))

                TestOBMAndKB.data_connections.append(conn)

                # additional data assets for scoring in space should be also created

                asset_details = self.wml_client.data_assets.create(
                    name=f'temp_{dataset_no}.csv',
                    file_path=self.data_location + f'temp_{dataset_no}.csv')
                asset_id = asset_details['metadata']['guid']

                conn_space = DataConnection(
                    data_join_node_name=f'temp_{dataset_no}',
                    location=AssetLocation(asset_id=asset_id))

                TestOBMAndKB.data_connections_space_only.append(conn_space)


            TestOBMAndKB.results_connection = None
        else:
            TestOBMAndKB.main_conn = DataConnection(
                data_join_node_name="temp_1",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'temp_1.csv'))

            TestOBMAndKB.temp_2_conn = DataConnection(
                data_join_node_name="temp_2",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'temp_2.csv'))

            TestOBMAndKB.temp_3_conn = DataConnection(
                data_join_node_name="temp_3",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'temp_3.csv'))

            TestOBMAndKB.temp_4_conn = DataConnection(
                data_join_node_name="temp_4",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'temp_4.csv'))

            TestOBMAndKB.results_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.results_cos_path)
            )

            TestOBMAndKB.data_connections = [self.main_conn,
                                             self.temp_2_conn,
                                             self.temp_3_conn,
                                             self.temp_4_conn]

    @print_test_separators
    def test_02a_create_data_join_graph__graph_created(self):
        print("Defining DataJoinGraph...")
        data_join_graph = DataJoinGraph()
        data_join_graph.node(name="temp_2")
        data_join_graph.node(name="temp_1", csv_separator=",", main=True)
        data_join_graph.node(name="temp_3")
        data_join_graph.node(name="temp_4")
        data_join_graph.edge(from_node="temp_1", to_node="temp_2",
                             from_column=["device"], to_column=["device"])
        data_join_graph.edge(from_node="temp_3", to_node="temp_1",
                             from_column=["device"], to_column=["device"])
        data_join_graph.edge(from_node="temp_4", to_node="temp_3",
                             from_column=["device"], to_column=["device"])

        TestOBMAndKB.data_join_graph = data_join_graph

        print(f"data_join_graph: {data_join_graph}")

    @print_test_separators
    def test_02b_create_data_join_graph__more_main_nodes__creation_failed(self):
        print("Defining DataJoinGraph...")
        data_join_graph = DataJoinGraph()
        data_join_graph.node(name="temp_2")
        data_join_graph.node(name="temp_1", csv_separator=",", main=True)
        with self.assertRaises(OBMMainNodeAlreadySet, msg="Check of OBM main node failed!"):
            data_join_graph.node(name="temp_3", main=True)

    # @print_test_separators
    # def test_03_data_join_graph_visualization(self):
    #     print("Visualizing data_join_graph...")
    #     self.data_join_graph.visualize()

    @print_test_separators
    def test_04_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        print("Initializing AutoAI experiment...")
        TestOBMAndKB.experiment = AutoAI(wml_credentials=self.wml_credentials,
                                         space_id=self.space_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    @print_test_separators
    def test_05_save_remote_data_and_DataConnection_setup(self):
        print("Writing multiple data files to COS...")
        if is_cp4d():
            pass

        else:  # for cloud and COS
            self.main_conn.write(data=self.data_location + 'temp_1.csv',
                                 remote_name=self.data_cos_path + 'temp_1.csv')
            self.temp_2_conn.write(data=self.data_location + 'temp_2.csv',
                                      remote_name=self.data_cos_path + 'temp_2.csv')
            self.temp_3_conn.write(data=self.data_location + 'temp_3.csv',
                                         remote_name=self.data_cos_path + 'temp_3.csv')
            self.temp_4_conn.write(data=self.data_location + 'temp_4.csv',
                                      remote_name=self.data_cos_path + 'temp_4.csv')

    @print_test_separators
    def test_06_initialize_optimizer(self):
        print("Initializing optimizer with data_join_graph...")
        TestOBMAndKB.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            prediction_type=AutoAI.PredictionType.BINARY,
            prediction_column=self.prediction_column,
            scoring=AutoAI.Metrics.ACCURACY_SCORE,
            max_number_of_estimators=1,
            data_join_graph=self.data_join_graph,
            t_shirt_size=self.experiment.TShirtSize.L,
            holdout_size=0.1,
            drop_duplicates=False,
            autoai_pod_version=self.pod_version
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    @print_test_separators
    def test_07_get_configuration_parameters_of_remote_auto_pipeline(self):
        print("Getting experiment configuration parameters...")
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_08a_check_data_join_node_names(self):
        if self.wml_client.ICP:
            main_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'temp_1.csv'))

            temp_2_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'temp_2.csv'))

            temp_3_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'temp_3.csv'))

            temp_4_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'temp_4.csv'))
        else:
            main_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'temp_1.csv'))

            temp_2_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'temp_2.csv'))

            temp_3_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'temp_3.csv'))

            temp_4_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'temp_4.csv'))


        data_connections = [main_conn,
                            temp_2_conn,
                            temp_3_conn,
                            temp_4_conn]

        with self.assertRaises(WrongDataJoinGraphNodeName, msg="Training started with wrong data join node names"):
            self.remote_auto_pipelines.fit(
                training_data_reference=data_connections,
                training_results_reference=self.results_connection,
                background_mode=False)

    @print_test_separators
    def test_08b_fit_run_training_of_auto_ai_in_wml(self):
        print("Scheduling OBM + KB training...")
        TestOBMAndKB.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=self.data_connections,
            training_results_reference=self.results_connection,
            background_mode=False)

        TestOBMAndKB.run_id = self.trained_pipeline_details['metadata']['id']

        TestOBMAndKB.pipeline_name = 'Pipeline_4'

        run_status = self.remote_auto_pipelines.get_run_status()
        print(f"Run ended with status:  {run_status}")

        self.assertEqual(run_status, "completed", msg=f"Obm training has not finished succesfully. More info: {self.trained_pipeline_details['entity'].get('status')}")

        TestOBMAndKB.model_location = self.trained_pipeline_details['entity']['status']['metrics'][-1]['context']['intermediate_model']['location']['model']
        TestOBMAndKB.training_status = self.trained_pipeline_details['entity']['results_reference']['location']['training_status']

        for connection in self.data_connections:
            self.assertIsNotNone(
                connection.auto_pipeline_params,
                msg=f'DataConnection auto_pipeline_params was not updated for connection: {connection.id}')

    @print_test_separators
    def test_09_download_original_training_data(self):
        print("Downloading each training file...")
        for connection in self.remote_auto_pipelines.get_data_connections():
            train_data = connection.read()

            print(f"Connection: {connection.id} - train data sample:")
            print(train_data.head())
            self.assertGreater(len(train_data), 0)

    @print_test_separators
    def test_10_download_preprocessed_obm_training_data(self):
        print("Downloading OBM preprocessed training data ...")
        TestOBMAndKB.train_data = (
            self.remote_auto_pipelines.get_preprocessed_data_connection().read())

        print("OBM train data sample:")
        print(self.train_data.head())
        self.train_data.to_csv("temp_preprocessed_data.csv", index=False)
        self.assertGreater(len(self.train_data), 0)


    @print_test_separators
    def test_11_visualize_obm_pipeline(self):
        print("Visualizing OBM model ...")
        TestOBMAndKB.data_join_pipeline = self.remote_auto_pipelines.get_preprocessing_pipeline()
        assert isinstance(self.data_join_pipeline._pipeline_json, list)
        str_graph = self.data_join_pipeline._graph.__str__()
        assert '[label="temp>=31"]' in self.data_join_pipeline._graph.__str__()
    #     self.data_join_pipeline.visualize()

    @print_test_separators
    def test_11b_check_if_data_join_pipeline_graph_correct(self):
        pipeline_json = self.data_join_pipeline._pipeline_json
        graph = self.data_join_pipeline._graph_json

        step_types = [message['feature_engineering_components']['obm'][0]['step_type'] for message in pipeline_json]
        last_non_join_iteration = step_types.index('join')
        selection_iteration = step_types.index('feature selection') + 1
        join_iterations = [i + 1 for i, x in enumerate(step_types) if x == "join"]

        for message in pipeline_json:
            name, iteration, _ = get_step_details(message)
            self.assertTrue(str(iteration) in graph['nodes'])

            if 1 < iteration <= 2:
                self.assertTrue(str(iteration) in graph['edges'][str(iteration - 1)])
            elif iteration in join_iterations:
                self.assertTrue(str(iteration) in graph['edges'][str(last_non_join_iteration)])
                extractors = get_join_extractors(message)

                if extractors is None:
                    continue
                for ext, i in zip(extractors, range(len(extractors))):
                    ext_index = str(iteration) + str(i)
                    self.assertTrue(ext_index in graph['nodes'] and
                                    ext_index in ext_index in graph['edges'][str(iteration)])

                    columns = get_extractor_columns(extractors[ext])
                    transformations = get_extractor_transformations(extractors[ext])
                    for j, column in enumerate(columns):
                        col_index = str(iteration) + str(i) + str(j)
                        self.assertTrue(col_index in graph['nodes'] and col_index in graph['edges'][str(ext_index)])

                        for transformation in transformations:
                            self.assertTrue(transformation in graph['edges'][str(col_index)])
                            self.assertTrue(str(selection_iteration) in graph['edges'][str(transformation)])

            elif iteration > selection_iteration:
                self.assertTrue(str(iteration) in graph['edges'][str(iteration - 1)])

    @print_test_separators
    def test_12_get_run_status(self):
        print("Getting training status...")
        status = self.remote_auto_pipelines.get_run_status()
        print(status)
        self.assertEqual("completed", status, msg="AutoAI run didn't finished successfully. Status: {}".format(status))

    @print_test_separators
    def test_13_get_run_details(self):
        print("Getting training details...")
        parameters = self.remote_auto_pipelines.get_run_details()
        print(parameters)
        self.assertIsNotNone(parameters)

    def test_13b_get_metrics(self):
        self.wml_client.training.get_metrics(self.run_id)

    @print_test_separators
    def test_14_predict_using_fitted_pipeline(self):
        print("Make predictions on best pipeline...")
        predictions = self.remote_auto_pipelines.predict(X=self.train_data.drop([self.prediction_column], axis=1).values[:5])
        print(predictions)
        self.assertGreater(len(predictions), 0)

    @print_test_separators
    def test_15_summary_listing_all_pipelines_from_wml(self):
        print("Getting pipelines summary...")
        pipelines_details = self.remote_auto_pipelines.summary()
        print(pipelines_details)

    @print_test_separators
    def test_16__get_data_connections__return_a_list_with_data_connections_with_optimizer_params(self):
        print("Getting all data connections...")
        data_connections = self.remote_auto_pipelines.get_data_connections()
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    @print_test_separators
    def test_17_get_pipeline_params_specific_pipeline_parameters(self):
        print("Getting details of Pipeline_1...")
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_1')
        print(pipeline_params)

    @print_test_separators
    def test_18__get_pipeline_params__fetch_best_pipeline_parameters__parameters_fetched_as_dict(self):
        print("Getting details of the best pipeline...")
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(best_pipeline_params)

    ########
    # LALE #
    ########

    @print_test_separators
    def test_19__get_pipeline__load_lale_pipeline__pipeline_loaded(self):
        print("Download and load Pipeline_4 as lale...")
        TestOBMAndKB.lale_pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name='Pipeline_4')
        self.assertWarnsRegex(Warning, "OBM",
                              msg="Warning (OBM pipeline support only for inspection and deployment) was not raised")
        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")

        # self.assertIsInstance(self.lale_pipeline, TrainablePipeline,
        #                       msg="Fetched pipeline is not of TrainablePipeline instance.")

    @print_test_separators
    def test_19b_get_all_pipelines_as_lale(self):
        print("Fetching all pipelines as lale...")
        summary = self.remote_auto_pipelines.summary()
        print(summary)
        failed_pipelines = []
        for pipeline_name in summary.reset_index()['Pipeline Name']:
            print(f"Getting pipeline: {pipeline_name}")
            lale_pipeline = None
            try:
                lale_pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name=pipeline_name)
                self.assertWarnsRegex(Warning, "OBM",
                                      msg="Warning (OBM pipeline support only for inspection and deployment) was not raised")
                from lale.operators import TrainablePipeline
                self.assertIsInstance(lale_pipeline, TrainablePipeline)
                predictions = lale_pipeline.predict(X=self.train_data.drop([self.prediction_column], axis=1).values[:1])
                print(predictions)
                self.assertGreater(len(predictions), 0, msg=f"Returned prediction for {pipeline_name} are empty")
            except:
                print(f"Failure: {pipeline_name}")
                failed_pipelines.append(pipeline_name)
                traceback.print_exc()

            if not TestOBMAndKB.lale_pipeline:
                TestOBMAndKB.lale_pipeline = lale_pipeline
                print(f"{pipeline_name} loaded for next test cases")

        self.assertEqual(len(failed_pipelines), 0, msg=f"Some pipelines failed. Full list: {failed_pipelines}")

    @unittest.skip("Skipped lale pretty print")
    def test_20__pretty_print_lale__checks_if_generated_python_pipeline_code_is_correct(self):
        pipeline_code = self.lale_pipeline.pretty_print()
        try:
            exec(pipeline_code)

        except Exception as exception:
            self.assertIsNone(
                exception,
                msg=f"Pretty print from lale pipeline was not successful \n\n Full pipeline code:\n {pipeline_code}")

    @print_test_separators
    def test_20a_remove_last_freeze_trainable_prefix_returned(self):
        print("Prepare pipeline to refinery...")
        from lale.lib.sklearn import KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression
        prefix = self.lale_pipeline.remove_last().freeze_trainable()
        # self.assertIsInstance(prefix, TrainablePipeline,
        #                       msg="Prefix pipeline is not of TrainablePipeline instance.")
        TestOBMAndKB.new_pipeline = prefix >> (KNeighborsClassifier | DecisionTreeClassifier | LogisticRegression)

    @print_test_separators
    def test_20b_hyperopt_fit_new_pipepiline(self):
        from lale.lib.lale import Hyperopt

        print("Do a hyperparameters optimization...")
        hyperopt = Hyperopt(estimator=TestOBMAndKB.new_pipeline, cv=3, max_evals=2)
        try:
            TestOBMAndKB.hyperopt_pipelines = hyperopt.fit(self.train_data.drop([self.prediction_column], axis=1).values,
                                                           self.train_data[self.prediction_column].values)
        except ValueError as e:
            print(f"ValueError message: {e}")
            traceback.print_exc()
            hyperopt_results = hyperopt._impl._trials.results
            print(hyperopt_results)
            self.assertIsNone(e, msg="hyperopt fit was not successful")

    @print_test_separators
    def test_20c_get_pipeline_from_hyperopt(self):
        print("Get locally optimized pipeline...")
        new_pipeline_model = TestOBMAndKB.hyperopt_pipelines.get_pipeline()
        self.assertWarnsRegex(Warning, "OBM",
                              msg="Warning (OBM pipeline support only for inspection and deployment) was not raised")
        print(f"Hyperopt_pipeline_model is type: {type(new_pipeline_model)}")
        TestOBMAndKB.new_sklearn_pipeline = new_pipeline_model.export_to_sklearn_pipeline()
        self.assertIsInstance(
            TestOBMAndKB.new_sklearn_pipeline,
            Pipeline,
            msg=f"Incorect Sklearn Pipeline type after conversion. Current: {type(TestOBMAndKB.new_sklearn_pipeline)}")

    @print_test_separators
    def test_21_predict_refined_pipeline(self):
        print("Make predictions on locally optimized pipeline...")
        predictions = TestOBMAndKB.new_sklearn_pipeline.predict(
            X=self.train_data.drop([self.prediction_column], axis=1).values[:1])
        print(predictions)
        self.assertGreater(len(predictions), 0, msg=f"Returned prediction for refined pipeline are empty")

    @print_test_separators
    def test_22_get_pipeline__load_specified_pipeline__pipeline_loaded(self):
        print("Loading pipeline as sklearn...")
        TestOBMAndKB.pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name='Pipeline_4',
                                                                        astype=self.experiment.PipelineTypes.SKLEARN)
        print(f"Fetched pipeline type: {type(self.pipeline)}")
        self.assertWarnsRegex(Warning, "OBM",
                              msg="Warning (OBM pipeline support only for inspection and deployment) was not raised")

        self.assertIsInstance(self.pipeline, Pipeline,
                              msg="Fetched pipeline is not of sklearn.Pipeline instance.")

    @print_test_separators
    def test_23_predict__do_the_predict_on_sklearn_pipeline__results_computed(self):
        print("Make predictions on sklearn pipeline...")
        predictions = self.pipeline.predict(self.train_data.drop([self.prediction_column], axis=1).values)
        print(predictions)


    @print_test_separators
    def test_24_get_historical_optimizer_with_data_join_graph(self):
        print("Fetching historical optimizer with OBM and KB...")
        TestOBMAndKB.historical_opt = self.experiment.runs.get_optimizer(run_id=self.remote_auto_pipelines._engine._current_run_id)
        # TestOBMAndKB.historical_opt = self.experiment.runs.get_optimizer(run_id='ead25ed8-0218-4d13-b6ad-c6218f910154')
        self.assertIsInstance(self.historical_opt.get_params()['data_join_graph'], DataJoinGraph,
                              msg="data_join_graph was incorrectly loaded")

        TestOBMAndKB.pipeline = self.historical_opt.get_pipeline()
        self.assertWarnsRegex(Warning, "OBM",
                              msg="Warning (OBM pipeline support only for inspection and deployment) was not raised")

    @print_test_separators
    def test_25_get_obm_training_data_from_historical_optimizer(self):
        print("Download historical preprocessed data from OBM stage...")
        train_data = (
            self.historical_opt.get_preprocessed_data_connection().read())

        print("OBM train data sample:")
        print(train_data.head())
        self.assertGreater(len(train_data), 0)

    #################################
    #      DEPLOYMENT SECTION       #
    #################################

    @print_test_separators
    def test_26_batch_deployment_setup_and_preparation(self):
        TestOBMAndKB.service = Batch(source_wml_credentials=self.wml_credentials,
                                     source_space_id=self.space_id,
                                     target_wml_credentials=self.wml_credentials,
                                     target_space_id=self.space_id)

        self.wml_client.set.default_space(self.space_id) if self.wml_client.ICP else None

    @print_test_separators
    def test_27_deploy__deploy_best_computed_pipeline_from_autoai_on_wml(self):
        self.service.create(
            experiment_run_id=self.remote_auto_pipelines._engine._current_run_id,
            model="Pipeline_3",
            deployment_name=self.DEPLOYMENT_NAME)

    @print_test_separators
    def test_28_score_deployed_model(self):
        if is_cp4d():
            payload_reference = self.data_connections_space_only
            output_data_reference = DataConnection(location=DeploymentOutputAssetLocation(name="batch_output_obm.csv"))
        else:
            payload_reference = self.data_connections
            output_data_reference = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path="prediction_results.csv")
            )

        scoring_params = self.service.run_job(
            payload=payload_reference,
            output_data_reference=output_data_reference,
            background_mode=False)
        print(scoring_params)

    @print_test_separators
    def test_29_list_deployments(self):
        deployments = self.service.list()
        print(deployments)
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    @print_test_separators
    def test_30_delete_deployment(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(self.service.asset_id)

    #################################
    #      NOTEBOOK SECTION       #
    #################################

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_31_prepare_notebook_version(self):
        if is_cp4d():
            try:
                print(type(self.remote_auto_pipelines))
                result_path = self.remote_auto_pipelines.get_run_details()['entity']['results_reference']['location'].get('path')
            except KeyError as e:
                print(self.remote_auto_pipelines.get_run_details())
                raise e
            self.assertIsNotNone(result_path)
            metadata_training_result_ref = DataConnection(
                                            location=FSLocation(
                                                path=result_path))
        else:
            metadata_training_result_ref = DataConnection(
                                            connection=S3Connection(endpoint_url=self.cos_endpoint,
                                                                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                                                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                                            location=S3Location(bucket=self.bucket_name,
                                                                path=self.results_cos_path,
                                                                model_location=self.model_location,
                                                                training_status=self.training_status
                                                                )
                                        )

        TestOBMAndKB.metadata = dict(
            training_data_reference=self.data_connections,
            training_result_reference=metadata_training_result_ref,
            prediction_type=self.experiment.PredictionType.BINARY,
            prediction_column=self.prediction_column,
            holdout_size=0.1,
            scoring=self.experiment.Metrics.ACCURACY_SCORE,
            max_number_of_estimators=1,
        )

        print("HMAC: Initializing AutoAI local scenario with metadata...")

        TestOBMAndKB.local_optimizer_cos = AutoAI().runs.get_optimizer(
            metadata=self.metadata
        )

    # @print_test_separators
    # def test_32_data_join_graph_visualization_notebook(self):
    #     optimizer_params = self.local_optimizer_cos.get_params()
    #     data_join_graph = optimizer_params['data_join_graph']
    #     print("Visualizing data_join_graph...")
    #     data_join_graph.visualize()

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_33_download_preprocessed_obm_training_data_notebook(self):
        print("Downloading OBM preprocessed training data...")
        TestOBMAndKB.train_data = (
            self.local_optimizer_cos.get_preprocessed_data_connection().read())

        print("OBM train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

    # @print_test_separators
    # def test_34_visualize_obm_pipeline_notebook(self):
    #     print("Visualizing OBM model ...")
    #     data_join_pipeline = self.local_optimizer_cos.get_preprocessing_pipeline()
    #     assert isinstance(data_join_pipeline._pipeline_json, list)
    #     assert '40 [label=NumberSet]' in data_join_pipeline._graph.__str__()
    # #     data_join_pipeline.visualize()

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_35_batch_deployment_setup_and_preparation_notebook(self):
        TestOBMAndKB.service = Batch(self.wml_credentials,
                                     source_space_id=self.space_id,
                                     target_wml_credentials=self.wml_credentials,
                                     target_space_id=self.space_id)

        self.wml_client.set.default_space(self.space_id) if self.wml_client.ICP else None

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_36__deploy__deploy_best_computed_pipeline_from_autoai_on_wml_notebook(self):
        self.service.create(
            metadata=self.metadata,
            model=self.pipeline_name,
            deployment_name=self.DEPLOYMENT_NAME)

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_37a_score_deployed_model_notebook(self):
        if is_cp4d():
            payload_reference = self.data_connections_space_only
            output_data_reference = DataConnection(
                location=DeploymentOutputAssetLocation(name="batch_output_obm.csv"))
        else:
            payload_reference = self.data_connections
            output_data_reference = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path="prediction_results.csv")
            )

        scoring_params = self.service.run_job(
            payload=payload_reference,
            output_data_reference=output_data_reference,
            background_mode=False)
        print(scoring_params)

        TestOBMAndKB.job_id = scoring_params['metadata']['id']

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_37b_list_jobs_notebook(self):
        jobs = self.service.list_jobs()
        print(jobs)

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_37c_rerun_job_notebook(self):
        scoring_params = self.service.rerun_job(self.job_id, background_mode=False)
        print(scoring_params)

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_37d_get_job_result_notebook(self):
        result = self.service.get_job_result(self.job_id)
        print(result)

    @print_test_separators
    @unittest.skipIf( is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_38_list_deployments_notebook(self):
        deployments = self.service.list()
        print(deployments)
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    @print_test_separators
    @unittest.skipIf(is_cp4d(), "Flow with notebook metadata works only on WS")
    def test_39_delete_deployment_notebook(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(self.service.asset_id)

if __name__ == '__main__':
    unittest.main()
