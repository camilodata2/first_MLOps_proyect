#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


import os
import time
import unittest
import uuid

import numpy as np
import pandas as pd

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials
from ibm_watson_machine_learning.helpers.connections import DataConnection

from ibm_watson_machine_learning.tests.utils.profiling import PSProfiler


def generate_random_data(count, data_filename):
    # we have 10 5 digit numbers + a comma for each column
    # that's 60 bytes per line
    # so this is around 6k for 1000
    # for 1GB we need 17,895,697 lines
    target = count * 1024 * 1024
    line_count = int(target / 60)
    print("Generating dataset with %s lines to match %s MB" % (line_count, target))
    df = pd.DataFrame(np.random.randint(0,100000,size=(line_count, 10)), columns=list('ABCDEFGHIJ'))
    df.to_csv(data_filename, index=False)


class TestDataConnection(unittest.TestCase):
    SPACE_PREFIX = "test_data_connection"
    SPACE_NAME = f"{SPACE_PREFIX}_{uuid.uuid5(uuid.NAMESPACE_DNS, SPACE_PREFIX)}"
    ASSET_NAME = "test_1G_random_10_fields"

    def get_or_create_space_id(self, client, space_name):
        cos_credentials = get_cos_credentials()
        cos_resource_crn = cos_credentials['resource_instance_id']
        # find our space
        our_space_id = None
        for det in client.spaces.get_details(asynchronous=True, get_all=True):
            for space_d in det.get('resources'):
                if space_d['entity']['name'] == space_name:
                    our_space_id = space_d['metadata']['id']

        # create our space
        if our_space_id is None:
            metadata = {
                client.spaces.ConfigurationMetaNames.NAME: TestDataConnection.SPACE_NAME,
                client.spaces.ConfigurationMetaNames.DESCRIPTION: TestDataConnection.SPACE_NAME + ' description',
                client.spaces.ConfigurationMetaNames.STORAGE: {
                                                               "type": "bmcos_object_storage",
                                                               "resource_crn": cos_resource_crn
                                                              }
            }

            our_space_d = client.spaces.store(meta_props=metadata, background_mode=False)
            our_space_id = our_space_d['metadata']['id']

        return our_space_id

    def get_or_create_asset_id(self, client, asset_name):
        our_asset_id = None

        client.data_assets.list()

        assets = client.data_assets.get_details()
        for asset in assets['resources']:
            if asset['metadata']['name'] == asset_name:
                our_asset_id = asset['metadata']['asset_id']
                print("Found existing data, asset id = %s" % our_asset_id)

        # upload the data if needed
        if our_asset_id is None:
            data_source = os.getenv("TEST_DATA_CONNECTION_SOURCE", None)
            if data_source is None:
                data_source = "testMemoryIssues27588.input_data.csv"
                generate_random_data(1024, data_source)
            print(f"Uploading data = {data_source}")
            asset_meta_props = {
                client.data_assets.ConfigurationMetaNames.NAME: asset_name,
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: asset_name,
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: data_source
            }
            asset_details = client.data_assets.store(asset_meta_props)
            our_asset_id = asset_details['metadata']['asset_id']

        return our_asset_id

    def testMemoryIssues27588(self):
        '''This is a non regression test in the context of:
        https://github.ibm.com/NGP-TWC/ml-planning/issues/27588

        This test deals with the memory issue raised at 
        https://github.ibm.com/NGP-TWC/ml-planning/issues/27588#issuecomment-42721847

        In this issue, OP sees a problem when reading 1GB of data with parameter
        'read_to_file': the python process ends up at 1GB of memory footprint, while
        we expect the memory footprint to less.

        After investigation, it looks like the default number of batchs is 10,000.
        The default data size for binary load is 32768. So each chunk is 10k times
        chunksize = about 300Mb. The inner loop for reading binary data does this:

        ```
            reader = self.flight_client.do_get(endpoint.ticket)
                try:
                    while True:
                        mini_batch, metadata = reader.read_chunk()
                        if read_to_file:
                            sink.write(b''.join(mini_batch.columns[0].tolist()))
        ```

        This is correct, but mini_batch.columns[0] is 10k elements of 32K.

        If we log the memory usage for a 1 GB load, we can see memory increases by 300M,
        when reading first chunk, then another +300M at second chunk, then GC kicks in
        and reduce memory by 300M, then it increases again by 300M.

        After adding the number_of_batch_rows to DataConnection.read() and setting this
        to 1000 (we will have chunks of 1000*32k = 30M), we can see a more steady memory
        around 200M for the process and this does not change.

        With number_of_batch_rows = 100, the memory stays stable around 150M, but the read
        time are longer. Probably more time spent in gc / context switching.

        This test is to be run with YPQA env. It will create and upload a 1 GB csv asset
        if needed. It generates a `test_data_connection.testMemoryIssues27588.csv` with
        memory profile.

        This test is run as manual test, as completion (upload of 1GB + download of 1GB)
        can be pretty long.
        '''
        wml_credentials = get_wml_credentials()

        client = APIClient(wml_credentials=wml_credentials)

        # find our space
        our_space_id = self.get_or_create_space_id(client, self.SPACE_NAME)
        print(our_space_id)

        client.set.default_space(our_space_id)

        # find our data
        our_asset_id = self.get_or_create_asset_id(client, self.ASSET_NAME)

        data_connection_read = DataConnection(
            data_asset_id=our_asset_id
        )

        data_connection_read._wml_client = client

        flight_parameters = {
            "num_partitions": 1
        }

        print("Starting flight-download from COS:")
        with PSProfiler() as profiler:
            file_path = "input_data.csv"
            start_time = time.time()
            binary_data = data_connection_read.read(use_flight=True,
                                                    raw=True,
                                                    flight_parameters=flight_parameters,
                                                    binary=True,
                                                    read_to_file=file_path,
                                                    number_of_batch_rows=1000)
            read_time = time.time()
            elapsed_time_sec = read_time - start_time
            print(f"Time to download in seconds: {elapsed_time_sec}")
        profiler.save_csv("test_data_connection.testMemoryIssues27588.csv")


if __name__ == "__main__":
    unittest.main()
