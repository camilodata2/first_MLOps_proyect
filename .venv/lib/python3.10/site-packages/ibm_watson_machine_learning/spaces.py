#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
import json
from ibm_watson_machine_learning.utils import SPACES_IMPORTS_DETAILS_TYPE, SPACES_EXPORTS_DETAILS_TYPE, SPACES_DETAILS_TYPE, MEMBER_DETAILS_TYPE, print_text_header_h2
from ibm_watson_machine_learning.metanames import SpacesMetaNames, MemberMetaNames, ExportMetaNames
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import  WMLClientError
from ibm_watson_machine_learning.utils.deployment.errors import PromotionFailed

_DEFAULT_LIST_LENGTH = 50


class Spaces(WMLResource):
    """Store and manage spaces. This is applicable only for IBM Cloud Pak® for Data."""

    ConfigurationMetaNames = SpacesMetaNames()
    """MetaNames for spaces creation."""
    MemberMetaNames = MemberMetaNames()
    """MetaNames for space members creation."""
    ExportMetaNames = ExportMetaNames()
    """MetaNames for spaces export."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

        self._ICP = client.ICP

    def store(self, meta_props):
        """Create a space.

        :param meta_props: meta data of the space configuration. To see available meta names use:

            .. code-block:: python

                client.spaces.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored space
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.NAME: 'my_space',
                client.spaces.ConfigurationMetaNames.DESCRIPTION: 'spaces',
            }
            spaces_details = client.spaces.store(meta_props=metadata)
        """

        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        WMLResource._chk_and_block_create_update_for_python36(self)
        Spaces._validate_type(meta_props, u'meta_props', dict, True)
        space_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client

        )

        creation_response = requests.post(
            self._wml_credentials['url'] + '/v4/spaces',
            headers=self._client._get_headers(),
            json=space_meta
        )

        spaces_details = self._handle_response(201, u'creating new spaces', creation_response)

        # Cloud Convergence: Set self._client.wml_credentials['instance_id'] to instance_id
        # during client.set.default_space since that's where space is associated with client
        # and also in client.set.default_project

        return spaces_details

    @staticmethod
    def get_href(spaces_details):
        """Get space href from space details.

        :param spaces_details: metadata of the stored space
        :type spaces_details: dict

        :return: space href
        :rtype: str

        **Example**

        .. code-block:: python

            space_details = client.spaces.get_details(space_uid)
            space_href = client.spaces.get_href(space_details)
        """

        Spaces._validate_type(spaces_details, u'spaces_details', object, True)
        Spaces._validate_type_of_details(spaces_details, SPACES_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(spaces_details, u'spaces_details',
                                                           [u'metadata', u'href'])

    @staticmethod
    def get_uid(spaces_details):
        """Get space uid from space details.

        :param spaces_details: metadata of the stored space
        :type spaces_details: dict

        :return: space UID
        :rtype: str

        **Example**

        .. code-block:: python

            space_details = client.spaces.get_details(space_uid)
            space_uid = client.spaces.get_uid(space_details)
        """

        Spaces._validate_type(spaces_details, u'spaces_details', object, True)
        Spaces._validate_type_of_details(spaces_details, SPACES_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(spaces_details, u'spaces_details',
                                                           [u'metadata', u'guid'])

    def delete(self, space_uid):
        """Delete a stored space.

        :param space_uid: space UID
        :type space_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.spaces.delete(sapce_uid)
        """
        Spaces._validate_type(space_uid, u'space_uid', str, True)

        space_endpoint = self._client.service_instance._href_definitions.get_space_href(space_uid)

        response_delete = requests.delete(space_endpoint, headers=self._client._get_headers())

        return self._handle_response(204, u'space deletion', response_delete, False)

    def get_details(self, space_uid=None, limit=None):
        """Get metadata of stored space(s). If space UID is not specified, it returns all the spaces metadata.

        :param space_uid: Space UID
        :type space_uid: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of stored space(s)
        :rtype: dict (if UID is not None) or {"resources": [dict]} (if UID is None)

        **Example**

        .. code-block:: python

            space_details = client.spaces.get_details(space_uid)
            space_details = client.spaces.get_details()
        """
        Spaces._validate_type(space_uid, u'space_uid', str, False)
        Spaces._validate_type(limit, u'limit', int, False)

        href = self._client.service_instance._href_definitions.get_spaces_href()
        if space_uid is None:
            return self._get_no_space_artifact_details(href+"?include=name,tags,custom,description", None, limit, 'spaces')
        return self._get_no_space_artifact_details(href, space_uid, limit, 'spaces')

    def list(self, limit=None, return_as_df=True):
        """Print stored spaces in a table format. If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed spaces or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.spaces.list()
        """

        space_resources = self.get_details(limit=limit)[u'resources']
        space_values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at']) for m in space_resources]

        table = self._list(space_values, [u'GUID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def update(self, space_uid, changes):
        """Updates existing space metadata.

        :param space_uid: UID of space which definition should be updated
        :type space_uid: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated space
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.NAME: "updated_space"
            }
            space_details = client.spaces.update(space_uid, changes=metadata)
        """
        self._validate_type(space_uid, u'space_uid', str, True)
        self._validate_type(changes, u'changes', dict, True)

        details = self.get_details(space_uid)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes,
                                                                            with_validation=True)

        href = self._client.service_instance._href_definitions.get_space_href(space_uid)

        response = requests.patch(href, json=patch_payload, headers=self._client._get_headers())

        updated_details = self._handle_response(200, u'spaces patch', response)

        return updated_details


#######SUPPORT FOR SPACE MEMBERS

    ###GET MEMBERS DETAILS

    def get_members_details(self, space_uid, member_id=None, limit=None):
        """Get metadata of members associated with a space.
        If member UID is not specified, it returns all the members metadata.

        :param space_uid: space UID
        :type space_uid: str
        :param member_id: member UID
        :type member_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of member(s) of a space
        :rtype: dict (if UID is not None) or {"resources": [dict]} (if UID is None)

        **Example**

        .. code-block:: python

            member_details = client.spaces.get_members_details(space_uid,member_id)
        """
        Spaces._validate_type(space_uid, u'space_uid', str, True)
        Spaces._validate_type(member_id, u'member_id', str, False)
        Spaces._validate_type(limit, u'limit', int, False)

        href = self._client.service_instance._href_definitions.get_members_href(space_uid)

        return self._get_no_space_artifact_details(href, member_id, limit, 'space members')

    ##DELETE MEMBERS

    def delete_members(self, space_uid, member_id):
        """Delete a member associated with a space.

        :param space_uid: space UID
        :type space_uid: str
        :param member_id: member UID
        :type member_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.spaces.delete_member(space_uid,member_id)
        """
        Spaces._validate_type(space_uid, u'space_uid', str, True)
        Spaces._validate_type(member_id, u'member_id', str, True)

        member_endpoint = self._client.service_instance._href_definitions.get_member_href(space_uid,member_id)

        response_delete = requests.delete(member_endpoint, headers=self._client._get_headers())

        return self._handle_response(204, u'space member deletion', response_delete, False)

#######UPDATE MEMBERS

    def update_member(self, space_uid, member_id, changes):
        """Updates existing member metadata.

        :param space_uid: UID of space
        :type space_uid: str
        :param member_id: UID of member that needs to be updated
        :type member_id: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated member
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.spaces.ConfigurationMetaNames.ROLE:"viewer"
            }
            member_details = client.spaces.update_member(space_uid, member_id, changes=metadata)
        """
        self._validate_type(space_uid, u'space_uid', str, True)
        self._validate_type(member_id, u'member_id', str, True)

        self._validate_type(changes, u'changes', dict, True)

        details = self.get_members_details(space_uid,member_id)

        patch_payload = self.MemberMetaNames._generate_patch_payload(details['entity'], changes,
                                                                            with_validation=True)

        href = self._client.service_instance._href_definitions.get_member_href(space_uid, member_id)

        response = requests.patch(href, json=patch_payload, headers=self._client._get_headers())

        updated_details = self._handle_response(200, u'members patch', response)

        return updated_details

#####CREATE MEMBER
    def create_member(self, space_uid, meta_props):
        """Create a member within a space.

        :param space_uid: UID of space
        :type space_uid: str
        :param meta_props:  metadata of the member configuration. To see available meta names use:

            .. code-block:: python

                client.spaces.MemberMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored member
        :rtype: dict

        .. note::
            * client.spaces.MemberMetaNames.ROLE can be any one of the following "viewer", "editor", "admin"
            * client.spaces.MemberMetaNames.IDENTITY_TYPE can be any one of the following "user", "service"
            * client.spaces.MemberMetaNames.IDENTITY can be either service-ID or IAM-userID

        **Example**

        .. code-block:: python

            metadata = {
                client.spaces.MemberMetaNames.ROLE:"Admin",
                client.spaces.MemberMetaNames.IDENTITY:"iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71",
                client.spaces.MemberMetaNames.IDENTITY_TYPE:"service"
            }
            members_details = client.spaces.create_member(space_uid=space_id, meta_props=metadata)
        """

        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        Spaces._validate_type(meta_props, u'meta_props', dict, True)
        space_meta = self.MemberMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client

        )

        creation_response = requests.post(
            self._wml_credentials['url'] + '/v4/spaces/'+space_uid+"/members",
            headers=self._client._get_headers(),
            json=space_meta
        )

        members_details = self._handle_response(201, u'creating new members', creation_response)

        return members_details

    def list_members(self, space_uid, limit=None, return_as_df=True):
        """Print stored members of a space in a table format.
        If limit is set to None there will be only first 50 records shown.

        :param space_uid: UID of space
        :type space_uid: str
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed members or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.spaces.list_members()
        """

        member_resources = self.get_members_details(space_uid,limit=limit)[u'resources']
        space_values = [(m[u'metadata'][u'guid'],  m[u'entity'][u'identity'], m[u'entity'][u'identity_type'], m[u'entity'][u'role'], m[u'metadata'][u'created_at']) for m in member_resources]

        table = self._list(space_values, [u'GUID', u'USERNAME', u'IDENTITY_TYPE', u'ROLE', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    @staticmethod
    def get_member_href(member_details):
        """Get member href from member details.

        :param member_details: metadata of the stored member
        :type member_details: dict

        :return: member href
        :rtype: str

        **Example**

        .. code-block:: python

            member_details = client.spaces.get_member_details(member_id)
            member_href = client.spaces.get_member_href(member_details)
        """

        Spaces._validate_type(member_details, u'member details', object, True)
        Spaces._validate_type_of_details(member_details, MEMBER_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(member_details, u'member_details',
                                                           [u'metadata', u'href'])

    @staticmethod
    def get_member_uid(member_details):
        """Get member uid from member details.

        :param member_details: metadata of the created member
        :type member_details: dict

        :return: member UID
        :rtype: str

        **Example**

        .. code-block:: python

            member_details = client.spaces.get_member_details(member_id)
            member_id = client.spaces.get_member_uid(member_details)
        """

        Spaces._validate_type(member_details, u'member_details', object, True)
        Spaces._validate_type_of_details(member_details, MEMBER_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(member_details, u'member_details',
                                                           [u'metadata', u'guid'])

    def imports(self, space_uid, file_path):
        """Imports assets in the zip file to a space. Updates existing space metadata.

        :param space_uid: UID of space which definition should be updated
        :type space_uid: str
        :param file_path: path to the content file to be imported
        :type file_path: str

        :return: metadata of import space
        :rtype: dict

        **Example**

        .. code-block:: python

            space_details = client.spaces.imports(space_uid, file_path="/tmp/spaces.zip")
        """
        self._validate_type(space_uid, u'space_uid', str, True)
        self._validate_type(file_path, u'file_path', str, True)



        with open(str(file_path), 'rb') as archive:
            data = archive.read()

        href = self._client.service_instance._href_definitions.get_space_href(space_uid) + "/imports"

        response = requests.post(href, headers=self._client._get_headers(), data=data)

        import_space_details = self._handle_response(202, u'spaces import', response)

        return import_space_details



    @staticmethod
    def get_imports_uid(imports_space_details):
        """Get imports_uid from imports space details.

        :param imports_space_details: metadata of the created space import
        :type imports_space_details: dict

        :return: imports space UID
        :rtype: str

        **Example**

        .. code-block:: python

            imports_space_details = client.spaces.get_imports_details(space_uid, imports_id)
            imports_id = client.spaces.get_imports_uid(imports_space_details)
        """

        Spaces._validate_type(imports_space_details, u'member_details', object, True)
        Spaces._validate_type_of_details(imports_space_details, SPACES_IMPORTS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(imports_space_details, u'imports_space_details',[u'metadata', u'guid'])

    @staticmethod
    def get_exports_uid(exports_space_details):
        """Get imports_uid from imports space details.

        :param exports_space_details: metadata of the created space import
        :type exports_space_details: dict

        :return: exports space UID
        :rtype: str

        **Example**

        .. code-block:: python

            exports_space_details = client.spaces.get_exports_details(space_uid, exports_id)
            exports_id = client.spaces.get_exports_uid(exports_space_details)
        """

        Spaces._validate_type(exports_space_details, u'exports_space_details', object, True)
        Spaces._validate_type_of_details(exports_space_details, SPACES_EXPORTS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(exports_space_details, u'exports_space_details',
                                                           [u'metadata', u'guid'])

    def get_imports_details(self, space_uid, imports_id=None, limit=None):
        """Get metadata of space import(s). If imports UID is not specified, it returns all the spaces imports details.

        :param space_uid: space UID
        :type space_uid: str
        :param imports_id: imports UID
        :type imports_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of space import(s)
        :rtype: dict (if UID is not None) or {"resources": [dict]} (if UID is None)

        **Examples**

        .. code-block:: python

            space_details = client.spaces.get_imports_details(space_uid)
            space_details = client.spaces.get_imports_details(space_uid, imports_id)
        """
        Spaces._validate_type(space_uid, u'space_uid', str, False)
        Spaces._validate_type(imports_id, u'imports_uid', str, False)
        Spaces._validate_type(limit, u'limit', int, False)


        if imports_id is None:
            href = self._client.service_instance._href_definitions.get_space_href(space_uid) + "/imports"
        else:
            href =  self._client.service_instance._href_definitions.get_space_href(space_uid) + "/imports/" + imports_id

        response = requests.get(href, headers=self._client._get_headers())

        import_space_details = self._handle_response(200, u'spaces import details', response)

        return import_space_details

    def exports(self, space_uid, meta_props):
        """Exports assets in the zip file from a space. Updates existing space metadata.

        :param space_uid: space UID
        :type space_uid: str
        :param meta_props: meta data of the space configuration. To see available meta names use:

            .. code-block:: python

                client.spaces.ExportMetaNames.get()

        :type meta_props: dict

        :return: metadata of exports space
        :rtype: dict

        **Example**

        .. code-block:: python

            meta_props = {
                client.spaces.ExportMetaNames.NAME: "sample",
                client.spaces.ExportMetaNames.DESCRIPTION : "test description",
                client.spaces.ExportMetaNames.ASSETS : {"data_assets": [], "wml_model":[]} }
            }
            space_details = client.spaces.exports(space_uid, meta_props=meta_props)

        """

        Spaces._validate_type(meta_props, u'meta_props', dict, True)
        space_exports_meta = self.ExportMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        space_exports_meta_json = json.dumps(space_exports_meta)

        self._validate_type(space_uid, u'space_uid', str, True)

        href = self._client.service_instance._href_definitions.get_space_href(space_uid) + "/exports"

        response = requests.post(href, headers=self._client._get_headers(), data=space_exports_meta_json)

        export_space_details = self._handle_response(202, u'spaces export', response)

        return export_space_details

    def get_exports_details(self, space_uid, exports_id=None, limit=None):
        """Get details of exports for space. If exports UID is not specified, it returns all the space exports metadata.

        :param space_uid: space UID
        :type space_uid: str
        :param exports_id: exports UID
        :type exports_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of exports of space
        :rtype: dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n

        **Example**

        .. code-block:: python

            space_details = client.spaces.get_exports_details(space_uid)
            space_details = client.spaces.get_exports_details(space_uid, exports_id)

        """
        Spaces._validate_type(space_uid, u'space_uid', str, False)
        Spaces._validate_type(exports_id, u'imports_uid', str, False)
        Spaces._validate_type(limit, u'limit', int, False)


        if exports_id is None:
            href = self._client.service_instance._href_definitions.get_space_href(space_uid) + "/exports"
        else:
            href =  self._client.service_instance._href_definitions.get_space_href(space_uid) + "/exports/" + exports_id

        response = requests.get(href, headers=self._client._get_headers())

        export_space_details = self._handle_response(200, u'spaces exports details', response)

        return export_space_details

    def download(self, space_uid, space_exports_uid, filename=None):
        """Downloads zip file deployment of specified UID.

        :param space_uid: space UID
        :type space_uid: str
        :param space_exports_uid: UID of virtual deployment
        :type space_exports_uid: str
        :param filename: filename of downloaded archive
        :type filename: str, optional

        :return: path to downloaded file
        :rtype: str

        **Example**

        .. code-block:: python

            client.spaces.download(space_uid)
        """
        Spaces._validate_type(space_exports_uid, u'space_exports_uid', str, False)

        if space_exports_uid is not None and not is_uid(space_exports_uid):
            raise WMLClientError(u'\'space_exports_uid\' is not an uid: \'{}\''.format(space_exports_uid))

        href =  self._client.service_instance._href_definitions.get_space_href(space_uid) + "/exports/" + space_exports_uid + "/content"

        response = requests.get(
            href,
            headers=self._client._get_headers()
        )

        if filename is None:
            filename = 'wmlspace.zip'

        if response.status_code == 200:
            with open(filename, "wb") as new_file:
                new_file.write(response.content)
                new_file.close()

                print_text_header_h2(
                    u'Successfully downloaded spaces export file: ' + str(filename))

                return filename
        else:
            raise WMLClientError(u'Unable to download spaces export: ' + response.text)

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

