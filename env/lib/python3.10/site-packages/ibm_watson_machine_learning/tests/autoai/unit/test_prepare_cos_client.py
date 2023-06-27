#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest

from ibm_watson_machine_learning.helpers import DataConnection, S3Connection, S3Location, AssetLocation
from ibm_watson_machine_learning.utils.autoai.utils import prepare_cos_client
from ibm_boto3 import resource, client
from ibm_botocore.client import Config


cos_resource = resource(
    service_name='s3',
    ibm_api_key_id='...',
    ibm_auth_endpoint='...',
    config=Config(signature_version="oauth"),
    endpoint_url='https://s3-api.us-geo.objectstorage.softlayer.net'
)


class TestPrepareCOSClient(unittest.TestCase):
    def test_01_check_clients_types(self):
        training_data_reference = [
            DataConnection(
                connection=S3Connection(
                    access_key_id='...',
                    secret_access_key='...',
                    endpoint_url='https://s3-api.us-geo.objectstorage.softlayer.net'
                ),
                location=S3Location(
                    bucket='some_bucket',
                    path='some_path'
                )
            ),
            DataConnection(
                connection=S3Connection(
                    api_key='...',
                    auth_endpoint='...',
                    endpoint_url='https://s3-api.us-geo.objectstorage.softlayer.net'
                ),
                location=S3Location(
                    bucket='some_bucket_2',
                    path='some_path_2'
                )
            )
        ]

        training_result_reference = DataConnection(
            connection=S3Connection(
                api_key='...',
                auth_endpoint='...',
                endpoint_url='https://s3-api.us-geo.objectstorage.softlayer.net'
            ),
            location=S3Location(
                bucket='some_bucket_2',
                path='some_path_2'
            )
        )

        data_clients, result_client = prepare_cos_client(training_data_references=training_data_reference,
                                                         training_result_reference=training_result_reference)

        self.assertIsInstance(data_clients, list, msg="data_clients is not a list!")
        self.assertIsInstance(data_clients[0][0], DataConnection,
                              msg='First variable from data_clients[0] tuple should be always a DataConnection')
        self.assertIsInstance(data_clients[1][0], DataConnection,
                              msg='First variable from data_clients[1] tuple should be always a DataConnection')
        self.assertIsInstance(result_client[0], DataConnection,
                              msg='First variable from result_client tuple should be always a DataConnection')
        self.assertEqual(str(type(data_clients[0][1])), str(type(cos_resource)),
                         msg="Wrong COS client type in [0] of data_clients")
        self.assertEqual(str(type(data_clients[1][1])), str(type(cos_resource)),
                         msg="Wrong COS client type in [1] of data_clients")
        self.assertEqual(str(type(result_client[1])), str(type(cos_resource)),
                         msg="Wrong COS client type in result_client")

    def test_02_wrong_connections(self):
        training_data_reference = [
            DataConnection(
                location=AssetLocation(
                    asset_id='...'
                )
            ),
            DataConnection(
                location=AssetLocation(
                    asset_id='...'
                )
            )
        ]

        training_result_reference = DataConnection(
            location=AssetLocation(
                asset_id='...'
            )
        )

        data_clients, result_client = prepare_cos_client(training_data_references=training_data_reference,
                                                         training_result_reference=training_result_reference)

        self.assertIsInstance(data_clients, list, msg="data_clients is not a list!")
        self.assertEqual(data_clients, [], msg="data_clients should be an empty list!")
        self.assertIsNone(result_client, msg="result_client should be None")


if __name__ == '__main__':
    unittest.main()
