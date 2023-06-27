#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import unittest
import logging
from ibm_watson_machine_learning.spaces import Spaces
from ibm_watson_machine_learning.tests.CP4D_35.preparation_and_cleaning import *
import uuid


class TestWMLClientWithPlatformSpace(unittest.TestCase):
    space_uid = None
    space_href=None
    member_uid = None
    member_href = None

    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithPlatformSpace.logger.info("Setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.space_name = str(uuid.uuid4())

        self.id1 = config.get(environment, 'id1')
        self.id2 = config.get(environment, 'id2')

        print("environment: {}".format(environment))

        if environment == 'CP4D_35':
            print("wml_credentials: {}".format(self.wml_credentials))

            print("space_name: {}".format(self.space_name))
            print("id1: {}".format(self.id1))
            print("id2: {}".format(self.id2))

    def test_01_store_space(self):
        metadata = {
                     self.client.spaces.ConfigurationMetaNames.NAME: 'cl_space' + self.space_name,
                     self.client.spaces.ConfigurationMetaNames.DESCRIPTION: self.space_name + ' description',
                   }

        space_create_details = self.client.spaces.store(meta_props=metadata)
        print(space_create_details)

        TestWMLClientWithPlatformSpace.space_id = self.client.spaces.get_id(space_create_details)

    def test_03_list(self):
        stdout_ = sys.stdout
        captured_output1 = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output1  # and redirect stdout.
        self.client.spaces.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        # If number of spaces is huge, recently created spaces may not be there
        # self.assertTrue(TestWMLClientWithPlatformSpace.space_id in captured_output1.getvalue())
        self.client.spaces.list()  # Just to see values.

        captured_output2 = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output2  # and redirect stdout.
        self.client.spaces.list(member='admin')
        self.client.spaces.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        # self.assertTrue(TestWMLClientWithPlatformSpace.space_id in captured_output2.getvalue())

    def test_04_get_space_details(self):
        space_get_details = self.client.spaces.get_details(TestWMLClientWithPlatformSpace.space_id)
        self.assertTrue(TestWMLClientWithPlatformSpace.space_id in str(space_get_details))

        space_id = self.client.spaces.get_id(space_get_details)
        print('space_details: {}'.format(space_get_details))
        print('space_id: {}'.format(space_id))

        print(self.client.spaces.get_details())

    def test_05_update_space(self):
        metadata = {
            self.client.spaces.ConfigurationMetaNames.NAME: "cl_updated_space2",
        }

        space_update_details = self.client.spaces.update(TestWMLClientWithPlatformSpace.space_id, metadata)
        self.assertTrue('updated_space2' in str(space_update_details))

        print("Updated space: ", space_update_details)

        TestWMLClientWithPlatformSpace.logger.info(space_update_details)

    # def test_06_create_member(self):
    #     metadata = {
    #                     self.client.spaces.MemberMetaNames.MEMBERS: [{
    #                                                                    "id": self.id2,
    #                                                                    "type": "user",
    #                                                                     "role": "editor"}]
    #                }
    #     members_create_details = self.client.spaces.create_member(space_id=TestWMLClientWithPlatformSpace.space_id,
    #                                                               meta_props=metadata)
    #
    #     print(members_create_details)
    #
    #     self.assertTrue(self.id2 in str(members_create_details))
    #
    # def test_07_get_member_details(self):
    #     # On CP4D, the user id varies on different clusters, its not the same.. so, disabling this check
    #     # List members shows the correct list
    #
    #     # member_get_details1 = self.client.spaces.get_member_details(TestWMLClientWithPlatformSpace.space_id,
    #     #                                                             self.id1)
    #     #
    #     # print(member_get_details1)
    #     #
    #     # self.assertTrue(self.id1 in str(member_get_details1))

        # member_get_details2 = self.client.spaces.get_member_details(TestWMLClientWithPlatformSpace.space_id,
        #                                                             self.id2)
        # self.assertTrue(self.id2 in str(member_get_details2))

    def test_08_list_members(self):
        self.client.spaces.list_members(TestWMLClientWithPlatformSpace.space_id)
        stdout_ = sys.stdout
        captured_output1 = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output1  # and redirect stdout.
        self.client.spaces.list_members(TestWMLClientWithPlatformSpace.space_id)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(self.id1 in captured_output1.getvalue())
        # self.assertTrue(self.id2 in captured_output1.getvalue())

        captured_output2 = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output2  # and redirect stdout.
        self.client.spaces.list_members(TestWMLClientWithPlatformSpace.space_id,
                                        role='editor',
                                        identity_type='user',
                                        state='active')
        sys.stdout = stdout_  # Reset redirect.
        # self.assertTrue(self.id2 in captured_output2.getvalue())
        self.assertTrue(self.id1 not in captured_output2.getvalue())

    # def test_09_update_space_member(self):
    #     metadata = {
    #         self.client.spaces.MemberMetaNames.MEMBER: {
    #             "id": self.id2,
    #             "type": "user",
    #             "role": "viewer"}
    #     }
    #
    #     # Only role is allowed to be patched
    #     members_update_details = self.client.spaces.update_member(space_id=TestWMLClientWithPlatformSpace.space_id,
    #                                                               member_id=self.id2,
    #                                                               changes=metadata)
    #
    #     print(members_update_details)
    #
    #     self.assertTrue('viewer' in str(members_update_details))
    #     TestWMLClientWithPlatformSpace.logger.info(members_update_details)
    #
    #     stdout_ = sys.stdout
    #     captured_output = io.StringIO()  # Create StringIO object
    #     sys.stdout = captured_output  # and redirect stdout.
    #     self.client.spaces.list_members(TestWMLClientWithPlatformSpace.space_id,
    #                                     role='viewer',
    #                                     identity_type='user',
    #                                     state='active')
    #     sys.stdout = stdout_  # Reset redirect.
    #     self.assertTrue(self.id2 in captured_output.getvalue())
    #     self.assertTrue(self.id1 not in captured_output.getvalue())
    #     self.assertTrue('viewer' in captured_output.getvalue())
    #     self.assertTrue('editor' not in captured_output.getvalue())

    def test_10_set_space(self):
        self.client.set.default_space(TestWMLClientWithPlatformSpace.space_id)
    #
    # def test_11_delete_member(self):
    #     self.client.spaces.delete_member(TestWMLClientWithPlatformSpace.space_id,
    #                                                      self.id2)
    def test_12_delete_space(self):
        delete_space = self.client.spaces.delete(TestWMLClientWithPlatformSpace.space_id)

        # space_ids = [ '6a4ae927-025d-4a2d-b522-2a73f8f693e5',
        #               '00cb5671-88ac-49cb-b79c-49395f83b26b',
        #               'bd4c7b9c-509d-42ef-97f5-0ab29d76b638',
        #               '2f6c6566-d5de-4aba-abfb-e47cef6859af',
        #               '6545a7f0-36de-40ad-98cc-4e8a799b5b11',
        #               '728ed6a4-6717-43cd-bab2-f58507350dcc',
        #               'dfb55132-e56b-474a-b829-9343c128eba0',
        #               '4efeb36b-b776-449e-8cc2-d8afa8d6bcd8',
        #               'e6326545-001b-42c5-99e6-0a29acd97f4f',
        #               'aea597fc-a252-4424-bd0a-e823c84d18f4',
        #               'e4c43780-24e1-481f-aa89-32479b4bc025',
        #               'acabe790-0794-4e6f-9c9a-67510e65cc4a',
        #               '1a32f203-03ce-4aa9-999d-b73bb59d6f4f',
        #               'c541a2fd-3d7b-46d1-9a62-bedfdc91bfba',
        #               '71accb0b-94fa-4246-b2e4-a25ead8dc5c5',
        #               '913d0f02-5acf-4779-a568-f3f83cf1363b',
        #               '4c2762b5-f762-4610-8c71-17625dffeba1',
        #               '2fbbde88-f422-4997-a78d-18be2132f64d',
        #               'ece4b838-1289-4e78-a27c-5244234d1adc',
        #               '2a9156f1-79fc-4bb4-b2a6-8fea74eb8076',
        #               '2b71af24-15a5-4d01-aca0-0b6d4c346236',
        #               '49c49fc0-e7b0-4eab-b59a-bb778399dcf7',
        #               '2fbbde88-f422-4997-a78d-18be2132f64d'
        #              ]
        #
        # for elem in space_ids:
        #     self.client.spaces.delete(elem)

