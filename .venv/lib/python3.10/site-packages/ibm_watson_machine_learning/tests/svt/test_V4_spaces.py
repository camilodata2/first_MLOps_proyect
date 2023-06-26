#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
from preparation_and_cleaning import *
import logging
from preparation_and_cleaning import *


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
        self.client = get_client()


    def test_01_service_instance_details(self):
        TestWMLClientWithSpace.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithSpace.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithSpace.logger.debug(details)

        self.assertIsInstance(details, dict)

    def test_02_save_space(self):
        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: "V4Space"
        }

        space_details = self.client.spaces.store(meta_props=metadata)

        TestWMLClientWithSpace.space_uid = self.client.spaces.get_uid(space_details)

        space_specific_details = self.client.spaces.get_details(TestWMLClientWithSpace.space_uid)
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(space_specific_details))

    def test_03_update_space(self):
        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: "my_space",
        }

        space_details = self.client.spaces.update(TestWMLClientWithSpace.space_uid, metadata)
        self.assertTrue('my_space' in str(space_details))
        TestWMLClientWithSpace.logger.info(space_details)
        self.assertTrue('V4Space' not in str(space_details))

    def test_04_get_space_details(self):
        details = self.client.spaces.get_details()
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(details))

        details2 = self.client.spaces.get_details(TestWMLClientWithSpace.space_uid)
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(details2))

    def test_05_list(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.spaces.list() # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithSpace.space_uid in captured_output.getvalue())
        self.client.spaces.list()  # Just to see values.

    def test_06_create_member(self):
        metadata = {
            self.client.spaces.MemberMetaNames.MEMBERS: [{
                "id": "17",
                "type": "user",
                "role": "viewer"
            }]
        }

        member_details = self.client.spaces.create_member(
            space_id=TestWMLClientWithSpace.space_uid,
            meta_props=metadata
        )
        TestWMLClientWithSpace.member_uid = self.client.spaces.get_member_uid(member_details)
        TestWMLClientWithSpace.member_href = self.client.spaces.get_member_href(member_details)

        member_specific_details = self.client.spaces.get_member_details(
            TestWMLClientWithSpace.space_uid,
            TestWMLClientWithSpace.member_uid
        )
        self.assertTrue(TestWMLClientWithSpace.member_uid in str(member_specific_details))

    def test_07_get_member_details(self):
        member_specific_details = self.client.spaces.get_member_details(
            TestWMLClientWithSpace.space_uid,
            TestWMLClientWithSpace.member_uid
        )
        self.assertTrue(TestWMLClientWithSpace.member_uid in str(member_specific_details))

    def test_09_delete_member(self):
        self.client.spaces.delete_member(
            TestWMLClientWithSpace.space_uid,
            TestWMLClientWithSpace.member_uid
        )

    def test_10_delete_space(self):
        self.client.spaces.delete(TestWMLClientWithSpace.space_uid)
