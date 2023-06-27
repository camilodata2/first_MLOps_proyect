#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import threading
import time
import unittest

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.data_loaders.experiment import ExperimentDataLoader
from ibm_watson_machine_learning.utils.autoai.enums import SamplingTypes, PredictionType
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_db_credentials
from ibm_watson_machine_learning.helpers.connections import DataConnection, DatabaseLocation
from ibm_watson_machine_learning.data_loaders.datasets.experiment import ExperimentIterableDataset


class DataLoaderTestHelper:
    """
    Class setting up all the properties required to benchmark
    subsampling.
    """
    BATCH_EXPECTED_SIZE: int = 1000000000
    TOLERANCE: int = 200000000
    database_type: str = "db2"
    schema_name: str = "DB2INST1"
    # table_name: str = "EVENTS"

    def __init__(self):
        self.connection_id = None

        self.wml_credentials = get_wml_credentials()
        self.project_id = self.wml_credentials["project_id"]

        self.client = APIClient(self.wml_credentials)
        self.client.set.default_project(self.project_id)

        self.create_connection_to_database()

    def create_connection_to_database(self) -> None:
        db_credentials = get_db_credentials("db")
        connection_details = self.client.connections.create({
            'datasource_type': self.client.connections.get_datasource_type_uid_by_name(self.database_type),
            'name': 'Connection to DB for data loader tests',
            'properties': db_credentials
        })
        self.connection_id = self.client.connections.get_uid(connection_details)

    def initialize_data_connection(self, table_name) -> DataConnection:
        data_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=DatabaseLocation(
                schema_name=self.schema_name,
                table_name=table_name
            )
        )
        data_connection.set_client(self.client)

        return data_connection

    def delete_connection(self) -> None:
        self.client.connections.delete(self.connection_id)


data_loader_test_helper = DataLoaderTestHelper()

TABLE_NAME = "HEART"

class TestExperimentIterableDatasetIssues(unittest.TestCase):
    '''This test case has non regression tests for TestExperimentIterableDatasetIssues
    '''

    def testSlowinessIssue27723(self):
        '''In https://github.ibm.com/NGP-TWC/ml-planning/issues/27723 :

        While reading data with small batch rows, it looked like there was some 3 sec. delays
        between batches, which is not correct.

        The original issue uses batchs of 10k but in tests, 10k is too large for
        issues to start. So to reproduce, use very small batch rows like 1000:
        
        Using HEART on DB2INST1 of autoai-db-db2-master.fyre.ibm.com + ENV=CPD_4_0
        
        ```
        [...]
        5 1649942298.0624645 18000 2 (1000, 18)
        6 1649942298.0705998 18000 2 (1000, 18)
        7 1649942298.2112653 18000 2 (1000, 18)
        8 1649942298.2448256 18000 2 (1000, 18)
        9 1649942298.2528276 18000 2 (1000, 18)
        10 1649942301.2568488 18000 2 (1000, 18)
        11 1649942301.2780516 18000 2 (1000, 18)
        12 1649942301.3005335 18000 2 (1000, 18)
        13 1649942301.3097115 18000 2 (1000, 18)
        14 1649942304.3099859 18000 2 (1000, 18)
        15 1649942304.338603 18000 2 (1000, 18)
        16 1649942304.346808 18000 2 (1000, 18)
        [...]
        Number of batches :  78
        Time taken to download in seconds:
        0.0 68.97492933273315 68.97492933273315
        ```

        We see the 3 sec delay kicking in (here at step 9-10 and 13-14).
        And it stops early (78 batchs vs 320 normally: there are 319796 records)

        The problem was in:
        ```
        data_rows = len(data) * self.q.qsize()
        if data_rows > self.number_of_batch_rows and not self.stop_reading:
           time.sleep(3)
        ```
        while we want to break the wait as soon as:
           - (data_rows > self.number_of_batch_rows) when so when q.qsize() decrease
           - stop_reading toggles to true
           
        Using a threading.Condition to track changes to q.size() and stop_reading:

        ```
        1 1649947374.3512974 18000 2 (1000, 18)
        2 1649947374.5184145 18000 2 (1000, 18)
        3 1649947374.6937237 18000 2 (1000, 18)
        4 1649947374.7069578 18000 2 (1000, 18)
        5 1649947374.7316537 18000 2 (1000, 18)
        6 1649947374.8649504 18000 2 (1000, 18)
        7 1649947374.8862677 18000 2 (1000, 18)
        8 1649947374.9075136 18000 2 (1000, 18)
        9 1649947374.9166982 18000 2 (1000, 18)
        10 1649947374.924368 18000 2 (1000, 18)
        11 1649947375.0355093 18000 2 (1000, 18)
        12 1649947375.0679495 18000 2 (1000, 18)
        13 [...]
        19 1649947375.1788774 18000 2 (1000, 18)
        20 1649947375.1870837 18000 2 (1000, 18)
        [...]
        Number of batches :  320
        Time taken to download in seconds:
        0.0 15.019081115722656 15.019081115722656
        ```
     
        (actual time varies between 15 and 20ish seconds, depending on network status)


        Note that we are just interested by the fact the values changes, so we don't
        need to be owner the lock, we just acquire it for notification.
        '''
        dataset = ExperimentIterableDataset(
            connection=data_loader_test_helper.initialize_data_connection(TABLE_NAME),
            experiment_metadata={
                "project_id": data_loader_test_helper.project_id
            },
            flight_parameters={"num_partitions": 1},
            number_of_batch_rows=1000
        )

        start_time = time.time()

        data_loader = ExperimentDataLoader(dataset)

        connect_time = time.time()

        i=0
        for data in data_loader:
            i += 1
            print(i, time.time(), data.size,data.ndim, data.shape)
            # data.to_csv(f"data{i}.csv", mode='a', index=False, header=False)

        print("Number of batches : ",i)

        end_time = time.time()
        connect_time_sec = connect_time - start_time
        rw_time_sec = end_time - connect_time
        elapsed_time_sec = end_time - start_time
        print("Time taken to download in seconds: ")
        print(connect_time_sec, rw_time_sec, elapsed_time_sec)

    def testSlowinessIssue27723_flightClient(self):
        '''Direct reading with the flightClient. With the HEART dataset:
        Result is:
        ```
        elapsed: 22.79
        ```

        (actual time varies depending on network status.
        the time above was recorded when using `iterable_read` took 16 sec on average sec)
        '''
        dataset = ExperimentIterableDataset(
            connection=data_loader_test_helper.initialize_data_connection(TABLE_NAME),
            experiment_metadata={
                "project_id": data_loader_test_helper.project_id
            },
            flight_parameters={"num_partitions": 1},
            number_of_batch_rows=1000
        )
        connection = dataset.connection
        readClient = connection.flight_client

        start_time = time.time()

        for i, endpoints in enumerate(connection.get_endpoints()):
            for j, endpoint in enumerate(endpoints):
                reader = readClient.do_get(endpoint.ticket)
                df = reader.read_pandas()
                print('x')
                # df.to_csv(f'direct_flight{i}_{j}.csv',index=False)
        read_time = time.time()
        print(f"elapsed: {read_time-start_time}")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()