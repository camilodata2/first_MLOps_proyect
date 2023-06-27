#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
import json, time
from ibm_watson_machine_learning.utils import StatusLogger, print_text_header_h1, print_text_header_h2
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import  WMLClientError
from ibm_watson_machine_learning.metanames import SpacesPlatformMetaNames, SpacesPlatformMemberMetaNames
from ibm_watson_machine_learning.instance_new_plan import ServiceInstanceNewPlan
from ibm_watson_machine_learning.utils.deployment.errors import PromotionFailed

_DEFAULT_LIST_LENGTH = 50


class PlatformSpaces(WMLResource):
    """Store and manage spaces."""

    ConfigurationMetaNames = SpacesPlatformMetaNames()
    """MetaNames for spaces creation."""

    MemberMetaNames = SpacesPlatformMemberMetaNames()
    """MetaNames for space members creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._client = client

    def _get_resources(self, url, op_name, params=None):
        if params is not None and 'limit' in params.keys():
            if params[u'limit'] < 1:
                raise WMLClientError('Limit cannot be lower than 1.')
            elif params[u'limit'] > 1000:
                raise WMLClientError('Limit cannot be larger than 1000.')

        if len(params) > 0:
            response_get = requests.get(
                url,
                headers=self._client._get_headers(),
                params=params
            )

            return self._handle_response(200, op_name, response_get)
        else:

            resources = []

            while True:
                response_get = requests.get(url, headers=self._client._get_headers())

                result = self._handle_response(200, op_name, response_get)
                resources.extend(result['resources'])

                if 'next' not in result:
                    break
                else:
                    url = self._wml_credentials["url"]+result['next']['href']
                    if('start=invalid' in url):
                        break
            return {
                "resources": resources
            }

    def store(self, meta_props, background_mode=True):
        """Create a space. The instance associated with the space via COMPUTE will be used for billing purposes on
        cloud. Note that STORAGE and COMPUTE are applicable only for cloud.

        :param meta_props:  meta data of the space configuration. To see available meta names use:

            .. code-block:: python

                client.spaces.ConfigurationMetaNames.get()

        :type meta_props: dict

        :param background_mode: indicator if store() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: metadata of the stored space
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.NAME: 'my_space',
                client.spaces.ConfigurationMetaNames.DESCRIPTION: 'spaces',
                client.spaces.ConfigurationMetaNames.STORAGE: {"resource_crn": "provide crn of the COS storage"},
                client.spaces.ConfigurationMetaNames.COMPUTE: {"name": "test_instance",
                                                               "crn": "provide crn of the instance"}
            }
            spaces_details = client.spaces.store(meta_props=metadata)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        PlatformSpaces._validate_type(meta_props, u'meta_props', dict, True)

        if ('compute' in meta_props or 'storage' in meta_props) and self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("'STORAGE' and 'COMPUTE' meta props are not applicable on "
                                 "IBM Cloud Pak® for Data. If using any of these, remove and retry")

        if 'storage' not in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError("'STORAGE' is mandatory for cloud")

        if 'compute' in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            if 'name' not in meta_props[u'compute'] or 'crn' not in meta_props[u'compute']:
                raise WMLClientError("'name' and 'crn' is mandatory for 'COMPUTE'")
            temp_meta = meta_props[u'compute']
            temp_meta.update({'type': 'machine_learning'})

            meta_props[u'compute'] = temp_meta

        space_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client

        )

        if 'compute' in meta_props and self._client.CLOUD_PLATFORM_SPACES:
            payload_compute = []
            payload_compute.append(space_meta[u'compute'])
            space_meta[u'compute'] = payload_compute

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_platform_spaces_href(),
            headers=self._client._get_headers(),
            json=space_meta)

        spaces_details = self._handle_response(202, u'creating new spaces', creation_response)

        # Cloud Convergence: Set self._client.wml_credentials['instance_id'] to instance_id
        # during client.set.default_space since that's where space is associated with client
        # and also in client.set.default_project
        #
        if 'compute' in spaces_details['entity'].keys() and self._client.CLOUD_PLATFORM_SPACES:
            instance_id = spaces_details['entity']['compute'][0]['guid']
            self._client.wml_credentials[u'instance_id'] = instance_id
            self._client.service_instance = ServiceInstanceNewPlan(self._client)
            self._client.service_instance.details = self._client.service_instance.get_details()


        if background_mode:
            print("Space has been created. However some background setup activities might still be on-going. "
                  "Check for 'status' field in the response. It has to show 'active' before space can be used. "
                  "If it's not 'active', you can monitor the state with a call to spaces.get_details(space_id). "
                  "Alternatively, use background_mode=False when calling client.spaces.store().")
            return spaces_details

        else:
            # note: monitor space status
            space_id = self.get_id(spaces_details)
            print_text_header_h1(u'Synchronous space creation with id: \'{}\' started'.format(space_id))

            status = spaces_details['entity']['status'].get('state')

            with StatusLogger(status) as status_logger:
                while status not in ['failed', 'error', 'completed', 'canceled', 'active']:
                    time.sleep(10)
                    spaces_details = self.get_details(space_id)
                    status = spaces_details['entity']['status'].get('state')
                    status_logger.log_state(status)
            # --- end note

            if u'active' in status:
                print_text_header_h2(u'\nCreating space  \'{}\' finished successfully.'.format(space_id))
            else:
                raise WMLClientError(
                    f"Space {space_id} creation failed with status: {spaces_details['entity']['status']}")

            return spaces_details

    @staticmethod
    def get_id(space_details):
        """Get space_id from space details.

        :param space_details: metadata of the stored space
        :type space_details: dict

        :return: space ID
        :rtype: str

        **Example**

        .. code-block:: python

            space_details = client.spaces.store(meta_props)
            space_id = client.spaces.get_id(space_details)
        """

        PlatformSpaces._validate_type(space_details, u'space_details', object, True)

        return WMLResource._get_required_element_from_dict(space_details, u'space_details',
                                                           [u'metadata', u'id'])

    @staticmethod
    def get_uid(space_details):
        """Get Unique Id of the space.

         *Deprecated:* Use ``get_id(space_details)`` instead.

         :param space_details: metadata of the space
         :type space_details: dict

         :return: Unique Id of space
         :rtype: str

        **Example**

        .. code-block:: python

            space_details = client.spaces.store(meta_props)
            space_uid = client.spaces.get_uid(space_details)

        """
        PlatformSpaces._validate_type(space_details, u'space_details', object, True)

        return WMLResource._get_required_element_from_dict(space_details, u'space_details',
                                                           [u'metadata', u'id'])

    def delete(self, space_id):
        """Delete a stored space.

        :param space_id: space ID
        :type space_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.spaces.delete(space_id)
        """
        PlatformSpaces._validate_type(space_id, u'space_id', str, True)

        space_endpoint = self._client.service_instance._href_definitions.get_platform_space_href(space_id)

        response_delete = requests.delete(space_endpoint, headers=self._client._get_headers())

        response = self._handle_response(202, u'space deletion', response_delete, False)

        print('DELETED')

        return response

    def get_details(self, space_id=None, limit=None, asynchronous=False, get_all=False):
        """Get metadata of stored space(s).

        :param space_id: space ID
        :type space_id: str, optional
        :param limit: applicable when `space_id` is not provided, otherwise `limit` will be ignored
        :type limit: str, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all:  if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of stored space(s)
        :rtype: dict

        **Example**

        .. code-block:: python

            space_details = client.spaces.get_details(space_uid)
            space_details = client.spaces.get_details(limit=100)
            space_details = client.spaces.get_details(limit=100, get_all=True)
            space_details = []
            for entry in client.spaces.get_details(limit=100, asynchronous=True, get_all=True):
                space_details.extend(entry)
        """
        PlatformSpaces._validate_type(space_id, u'space_id', str, False)

        href = self._client.service_instance._href_definitions.get_platform_space_href(space_id)

        if space_id is not None:
            response_get = requests.get(href, headers=self._client._get_headers())

            return self._handle_response(200, 'Get space', response_get)

        else:
            return self._get_with_or_without_limit(self._client.service_instance._href_definitions.get_platform_spaces_href(),
                                                   limit,
                                                   'spaces',
                                                   summary=False,
                                                   pre_defined=False,
                                                   skip_space_project_chk=True,
                                                   _async=asynchronous,
                                                   _all=get_all)

    def list(self, limit=None, member=None, roles=None, return_as_df=True):
        """Print stored spaces in a table format. If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional
        :param member: filters the result list to only include spaces where the user with a matching user id
            is a member
        :type member: str, optional
        :param roles: limit number of fetched records
        :type roles: str, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed spaces or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.spaces.list()
        """

        PlatformSpaces._validate_type(limit, u'limit', int, False)
        href = self._client.service_instance._href_definitions.get_platform_spaces_href()

        params = {}

        if limit is not None:
            params.update({'limit': limit})

        if limit is None:
            params.update({'limit': 50})

        if member is not None:
            params.update({'member': member})

        if roles is not None:
            params.update({'roles': roles})

        space_resources = self._get_resources(href, 'spaces', params)[u'resources']

        # space_resources = self._get_no_space_artifact_details(href, None, limit, 'spaces')[u'resources']

        space_values = [(m[u'metadata'][u'id'],
                         m[u'entity'][u'name'],
                         m[u'metadata'][u'created_at']) for m in space_resources]

        if limit is None:
            print("Note: 'limit' is not provided. Only first 50 records will be displayed if the number of records "
                  "exceed 50")

        table = self._list(space_values, [u'ID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table


    def update(self, space_id, changes):
        """Updates existing space metadata. 'STORAGE' cannot be updated.
        STORAGE and COMPUTE are applicable only for cloud.

        :param space_id: ID of space which definition should be updated
        :type space_id: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated space
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.NAME:"updated_space",
                client.spaces.ConfigurationMetaNames.COMPUTE: {"name": "test_instance",
                                                               "crn": "v1:staging:public:pm-20-dev:us-south:a/09796a1b4cddfcc9f7fe17824a68a0f8:f1026e4b-77cf-4703-843d-c9984eac7272::"
                }
            }
            space_details = client.spaces.update(space_id, changes=metadata)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        if ('compute' in changes or 'storage' in changes) and self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("'STORAGE' and 'COMPUTE' meta props are not applicable on"
                                 "IBM Cloud Pak® for Data. If using any of these, remove and retry")

        if 'storage' in changes:
            raise WMLClientError("STORAGE cannot be updated")

        self._validate_type(space_id, u'space_id', str, True)
        self._validate_type(changes, u'changes', dict, True)

        details = self.get_details(space_id)

        if 'compute' in changes and self._client.CLOUD_PLATFORM_SPACES:
            changes[u'compute'][u'type'] = 'machine_learning'

            payload_compute = []
            payload_compute.append(changes[u'compute'])
            changes[u'compute'] = payload_compute

        print("changes in update: ", changes)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes)

        print("patch payload: ", patch_payload)

        href = self._client.service_instance._href_definitions.get_platform_space_href(space_id)

        response = requests.patch(href, json=patch_payload, headers=self._client._get_headers())

        updated_details = self._handle_response(200, u'spaces patch', response)

        # Cloud Convergence
        if 'compute' in updated_details['entity'].keys() and self._client.CLOUD_PLATFORM_SPACES:
            instance_id = updated_details['entity']['compute'][0]['guid']
            self._client.wml_credentials[u'instance_id'] = instance_id
            self._client.service_instance = ServiceInstanceNewPlan(self._client)
            self._client.service_instance.details = self._client.service_instance.get_details()

        return updated_details


#######SUPPORT FOR SPACE MEMBERS

    def create_member(self, space_id, meta_props):
        """Create a member within a space.

        :param space_id: ID of space which definition should be updated
        :type space_id: str
        :param meta_props:  metadata of the member configuration. To see available meta names use:

            .. code-block:: python

                client.spaces.MemberMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored member
        :rtype: dict

        .. note::
            * `role` can be any one of the following: "viewer", "editor", "admin"
            * `type` can be any one of the following: "user", "service"
            * `id` can be either service-ID or IAM-userID

        **Examples**

        .. code-block:: python

            metadata = {
                client.spaces.MemberMetaNames.MEMBERS: [{"id":"IBMid-100000DK0B",
                                                         "type": "user",
                                                         "role": "admin" }]
            }
            members_details = client.spaces.create_member(space_id=space_id, meta_props=metadata)

        .. code-block:: python

            metadata = {
                client.spaces.MemberMetaNames.MEMBERS: [{"id":"iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71",
                                                         "type": "service",
                                                         "role": "admin" }]
            }
            members_details = client.spaces.create_member(space_id=space_id, meta_props=metadata)
        """
        self._validate_type(space_id, u'space_id', str, True)

        PlatformSpaces._validate_type(meta_props, u'meta_props', dict, True)

        meta = {}

        if 'members' in meta_props:
            meta = meta_props
        elif 'member' in meta_props:
            dictionary = meta_props['member']
            payload = []
            payload.append(dictionary)
            meta['members'] = payload

        space_meta = self.MemberMetaNames._generate_resource_metadata(
            meta,
            with_validation=True,
            client=self._client
        )

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_platform_spaces_members_href(space_id),
            headers=self._client._get_headers(),
            json=space_meta)

        # TODO: Change response code one they change it to 201
        members_details = self._handle_response(200, u'creating new members', creation_response)

        return members_details

    def get_member_details(self, space_id, member_id):
        """Get metadata of member associated with a space.

        :param space_id: ID of space which definition should be updated
        :type space_id: str
        :param member_id: member ID
        :type member_id: str

        :return: metadata of member of a space
        :rtype: dict

        **Example**

        .. code-block:: python

            member_details = client.spaces.get_member_details(space_uid,member_id)
        """
        PlatformSpaces._validate_type(space_id, u'space_id', str, True)

        PlatformSpaces._validate_type(member_id, u'member_id', str, True)

        href = self._client.service_instance._href_definitions.get_platform_spaces_member_href(space_id, member_id)

        response_get = requests.get(href, headers=self._client._get_headers())

        return self._handle_response(200, 'Get space member', response_get)

    def delete_member(self, space_id, member_id):
        """Delete a member associated with a space.

        :param space_id:  space UID
        :type space_id: str
        :param member_id:  member UID
        :type member_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.spaces.delete_member(space_id,member_id)
        """
        PlatformSpaces._validate_type(space_id, u'space_id', str, True)
        PlatformSpaces._validate_type(member_id, u'member_id', str, True)

        member_endpoint = self._client.service_instance._href_definitions.get_platform_spaces_member_href(space_id, member_id)

        response_delete = requests.delete(member_endpoint, headers=self._client._get_headers())

        print('DELETED')

        return self._handle_response(204, u'space member deletion', response_delete, False)

    def update_member(self, space_id, member_id, changes):
        """Updates existing member metadata.

        :param space_id: ID of space
        :type space_id: str
        :param member_id: ID of member that needs to be updated
        :type member_id: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated member
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.spaces.MemberMetaNames.MEMBER: {"role": "editor"}
            }
            member_details = client.spaces.update_member(space_id, member_id, changes=metadata)
        """
        self._validate_type(space_id, u'space_id', str, True)
        self._validate_type(member_id, u'member_id', str, True)

        self._validate_type(changes, u'changes', dict, True)

        details = self.get_member_details(space_id, member_id)

        # The member record is a bit different than most other type of records we deal w.r.t patch
        # There is no encapsulating object for the fields. We need to be consistent with the way we
        # provide the meta in create/patch. When we give with .MEMBER, _generate_patch_payload
        # will generate with /member patch. So, separate logic for member patch inline here
        changes1 = changes['member']

        # Union of two dictionaries. The one in changes1 will override existent ones in current meta
        details.update(changes1)

        id_str = {}
        role_str = {}
        type_str = {}
        state_str = {}

        # if 'id' in details:
        #     id_str["op"] = "replace"
        #     id_str["path"] = "/id"
        #     id_str["value"] = details[u'id']
        if 'role' in details:
            role_str["op"] = "replace"
            role_str["path"] = "/role"
            role_str["value"] = details[u'role']
        # if 'type' in details:
        #     type_str["op"] = "replace"
        #     type_str["path"] = "/type"
        #     type_str["value"] = details[u'type']
        if 'state' in details:
            state_str["op"] = "replace"
            state_str["path"] = "/state"
            state_str["value"] = details[u'state']

        patch_payload = []

        # if id_str:
        #     patch_payload.append(id_str)
        if role_str:
            patch_payload.append(role_str)
        # if type_str:
        #     patch_payload.append(type_str)
        if state_str:
            patch_payload.append(state_str)

         # patch_payload = self.MemberMetaNames._generate_patch_payload(details, changes, with_validation=True)

        href = self._client.service_instance._href_definitions.get_platform_spaces_member_href(space_id,member_id)

        response = requests.patch(href, json=patch_payload, headers=self._client._get_headers())

        updated_details = self._handle_response(200, u'members patch', response)

        return updated_details

    def list_members(self, space_id, limit=None, identity_type=None, role=None, state=None, return_as_df=True):
        """Print stored members of a space in a table format.
        If limit is set to None there will be only first 50 records shown.

        :param space_id: ID of space
        :type space_id: str
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param identity_type: filter the members by type
        :type identity_type: str, optional
        :param role: filter the members by role
        :type role: str, optional
        :param state: filter the members by state
        :type state: str, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed members or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.spaces.list_members(space_id)
        """
        self._validate_type(space_id, u'space_id', str, True)

        params = {}

        if limit is not None:
            params.update({'limit': limit})

        if limit is None:
            params.update({'limit': 50})

        if identity_type is not None:
            params.update({'type': identity_type})

        if role is not None:
            params.update({'role': role})

        if state is not None:
            params.update({'state': state})

        href = self._client.service_instance._href_definitions.get_platform_spaces_members_href(space_id)

        member_resources = self._get_resources(href, 'space members', params)[u'resources']

        # space_values = [(m[u'metadata'][u'id'],
        #                  m[u'entity'][u'id'],
        #                  m[u'entity'][u'type'],
        #                  m[u'entity'][u'role'],
        #                  m[u'entity'][u'state'],
        #                  m[u'metadata'][u'created_at']) for m in member_resources]


        # self._list(space_values, [u'ID', u'IDENTITY',
        #                           u'IDENTITY_TYPE',
        #                           u'ROLE',
        #                           u'STATE',
        #                           u'CREATED'], limit, _DEFAULT_LIST_LENGTH)

        space_values = [(m[u'id'],
                         m[u'type'],
                         m[u'role'],
                         m[u'state']) if 'state' in m else
                        (m[u'id'],
                         m[u'type'],
                         m[u'role'],
                         None) for m in member_resources]

        if limit is None:
            print("Note: 'limit' is not provided. Only first 50 records will be displayed if the number of records "
                  "exceed 50")

        table = self._list(space_values, [u'ID',
                                          u'TYPE',
                                          u'ROLE',
                                          u'STATE'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def promote(self, asset_id: str, source_project_id: str, target_space_id: str, rev_id: str = None) -> str:
        """Promote asset from project to space.

        :param asset_id: stored asset
        :type asset_id: str

        :param source_project_id: source project, from which asset is promoted
        :type source_project_id: str

        :param target_space_id: target space, where asset is promoted
        :type target_space_id: str

        :param rev_id: revision ID of the promoted asset
        :type rev_id: str, optional

        :return: promoted asset id
        :rtype: str

        **Examples**

        .. code-block:: python

            promoted_asset_id = client.spaces.promote(asset_id, source_project_id=project_id, target_space_id=space_id)
            promoted_model_id = client.spaces.promote(model_id, source_project_id=project_id, target_space_id=space_id)
            promoted_function_id = client.spaces.promote(function_id, source_project_id=project_id, target_space_id=space_id)
            promoted_data_asset_id = client.spaces.promote(data_asset_id, source_project_id=project_id, target_space_id=space_id)
            promoted_connection_asset_id = client.spaces.promote(connection_id, source_project_id=project_id, target_space_id=space_id)
        """
        promote_payload = {"spaceId": target_space_id,
                           "projectId": source_project_id,
                           "assetDescription": "Asset promoted by ibm_wml client"}

        if rev_id:
            promote_payload['revisionId'] = rev_id

        promote_href = self._client.service_instance._href_definitions.promote_asset_href(asset_id)
        response = requests.post(
            promote_href,
            headers=self._client._get_headers(),
            json=promote_payload
        )
        promotion_details = self._client.repository._handle_response(200, f'promote asset', response)

        try:
            promoted_asset_id = promotion_details['promotedAsset']['asset_id']
        except KeyError as key_err:
            raise PromotionFailed(source_project_id, target_space_id, promotion_details, reason=key_err)

        return promoted_asset_id





