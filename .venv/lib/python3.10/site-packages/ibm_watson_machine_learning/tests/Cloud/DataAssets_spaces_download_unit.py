#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from unittest import mock
from unittest.mock import Mock

import requests

from assets import Assets
from href_definitions import HrefDefinitions
from ibm_watson_machine_learning import APIClient
from instance_new_plan import ServiceInstanceNewPlan


class TestDataAssetsDownloadUnittests(unittest.TestCase):

    def setUp(self):
        client = Mock(spec=APIClient)
        client.wml_credentials = {"url": "the-wml-url"}
        client.WSD = None
        client.ICP = True
        client.ICP_PLATFORM_SPACES = None
        client.wml_token = ''
        client.service_instance = Mock(spec=ServiceInstanceNewPlan)
        client.service_instance._href_definitions = Mock(spec=HrefDefinitions)
        self.assets = Assets(client)

    def build_mock_get_for_download(self, expected_content):
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        # There will be 3 requests. The first two return some JSON, and we
        # provide some fake responses for these.
        mock_response.json.side_effect = [
            {
                "attachments": [{'id': 'attachment-id'}],
            },
            {
                "url": "attachment_signed_url"
            },
            {}
        ]
        # The response to the third request is supposed to have the file content.
        # But we don't check that the content is actually retrieved only
        # from the third response: all three returned responses will have
        # the same content.
        mock_response.content = expected_content
        mock_response.url = 'the-url'
        mock_response.request = Mock()
        mock_get = Mock()
        mock_get.return_value = mock_response
        return mock_get

    def test_download_accepts_and_stores_filename(self):
        expected_content = 'the file content'
        mock_open = mock.mock_open()
        with mock.patch('requests.get', self.build_mock_get_for_download(expected_content)):
            with mock.patch('builtins.open', mock_open):
                self.assets.download('asset-id', 'filename')
        mock_open.assert_called_once_with('filename', 'wb')
        mock_open().write.assert_called_once_with(expected_content)

    def test_get_content_returns_content_without_storing(self):
        expected_content = 'content'
        mock_open = mock.mock_open()
        with mock.patch('requests.get', self.build_mock_get_for_download(expected_content)):
            with mock.patch('builtins.open', mock_open):
                content = self.assets.get_content('asset-id')
        self.assertEqual(0, mock_open.call_count)
        self.assertEqual(expected_content, content)


if __name__ == '__main__':
    unittest.main()
