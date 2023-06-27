#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from ibm_watson_machine_learning.spaces import Spaces
from ibm_watson_machine_learning.tests.ICP.preparation_and_cleaning import *


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
                    self.client.repository.SpacesMetaNames.NAME: "V4Space"
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

    def test_03_update_space(self):
        metadata = {
            self.client.repository.SpacesMetaNames.NAME: "my_space_TestWMLClientWithSpace",
            self.client.repository.SpacesMetaNames.DESCRIPTION: "mnist best model",
        }

        space_details = self.client.repository.update_space(TestWMLClientWithSpace.space_uid, metadata)
        self.assertTrue('my_space_TestWMLClientWithSpace' in str(space_details))
        TestWMLClientWithSpace.logger.info(space_details)
        self.assertTrue('V4Space' not in str(space_details))

    def test_04_get_space_details(self):
        details = self.client.repository.get_space_details()
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(details))

        details2 = self.client.repository.get_space_details(TestWMLClientWithSpace.space_uid)
        self.assertTrue(TestWMLClientWithSpace.space_uid in str(details2))

    def test_05_list(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_spaces()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithSpace.space_uid in captured_output.getvalue())
        self.client.repository.list_spaces()  # Just to see values.

    def test_06_create_member(self):
        metadata = {
            self.client.spaces.MemberMetaNames.ROLE: "Viewer",
            self.client.spaces.MemberMetaNames.IDENTITY_TYPE: "service",
            self.client.spaces.MemberMetaNames.IDENTITY: "IBMid-310002RQJW"
        }

        member_details = self.client.spaces.create_member(space_uid=TestWMLClientWithSpace.space_uid,meta_props=metadata)

        TestWMLClientWithSpace.member_uid = self.client.spaces.get_member_uid(member_details)
        TestWMLClientWithSpace.member_href = self.client.spaces.get_member_href(member_details)

        member_specific_details = self.client.spaces.get_members_details(TestWMLClientWithSpace.space_uid,TestWMLClientWithSpace.member_uid)
        self.assertTrue(TestWMLClientWithSpace.member_uid in str(member_specific_details))

    def test_07_get_member_details(self):
        member_specific_details = self.client.spaces.get_members_details(TestWMLClientWithSpace.space_uid,TestWMLClientWithSpace.member_uid)
        self.assertTrue(TestWMLClientWithSpace.member_uid in str(member_specific_details))

    def test_08_delete_member(self):
        delete_member = self.client.spaces.delete_members(TestWMLClientWithSpace.space_uid,TestWMLClientWithSpace.member_uid)
        self.assertTrue("SUCCESS" in str(delete_member))
    def test_09_delete_space(self):
        delete_space = self.client.repository.delete(TestWMLClientWithSpace.space_uid)
        self.assertTrue("SUCCESS" in str(delete_space))

