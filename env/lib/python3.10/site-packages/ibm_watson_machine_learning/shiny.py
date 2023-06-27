#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.utils import DATA_ASSETS_DETAILS_TYPE, modify_details_for_script_and_shiny
from ibm_watson_machine_learning.metanames import ShinyMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure, ForbiddenActionForGitBasedProject
import os
import warnings

_DEFAULT_LIST_LENGTH = 50


class Shiny(WMLResource):
    """Store and manage shiny assets."""

    ConfigurationMetaNames = ShinyMetaNames()
    """MetaNames for Shiny Assets creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    def get_details(self, shiny_uid=None):
        """Get shiny asset details. If no shiny_uid is passed, details for all shiny assets will be returned.

        :param shiny_uid: Unique id of shiny asset
        :type shiny_uid: str, optional

        :return: metadata of the stored shiny asset
        :rtype:
          - **dict** - if runtime_uid is not None
          - **{"resources": [dict]}** - if runtime_uid is None

        **Example**

        .. code-block:: python

            shiny_details = client.shiny.get_details(shiny_uid)

        """
        def get_required_elements(response):
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:
                response = modify_details_for_script_and_shiny(response)
                final_response = {
                    "metadata": response[u'metadata'],
                    "entity": {}
                }

                return final_response

        return self._get_asset_based_resource(shiny_uid, 'shiny_asset', get_required_elements)

    def store(self, meta_props, file_path):
        """Create a shiny asset and uploads content to it.

        :param meta_props: metadata of shiny asset
        :type meta_props: dict
        :param file_path: path to the content file to be uploaded
        :type file_path: str

        :return: metadata of the stored shiny asset
        :rtype: dict

        **Example**

        .. code-block:: python

            meta_props = {
                client.shiny.ConfigurationMetaNames.NAME: "shiny app name"
            }

            shiny_details = client.shiny.store(meta_props, file_path="/path/to/file")
        """
        if self._client.project_type == 'local_git_storage':
            raise ForbiddenActionForGitBasedProject(reason="Storing Shiny apps is not supported for git based project.")

        WMLResource._chk_and_block_create_update_for_python36(self)

        Shiny._validate_type(file_path, u'file_path', str, True)

        shiny_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client
        )

        response = self._create_asset(shiny_meta, file_path)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            final_response = {
                "metadata": response[u'metadata'],
                "entity": {}
            }

            return final_response

    def _create_asset(self, shiny_meta, file_path):

        ##Step1: Create a shiny asset
        name = shiny_meta['metadata'][u'name']

        desc = ""
        if shiny_meta['metadata'].get('description'):
            desc = shiny_meta['metadata'][u'description']

        shiny_sw_spec_id = None
        if shiny_meta.get('software_spec_uid'):
            shiny_sw_spec_id = shiny_meta[u'software_spec_uid']

        if self._client.ICP_46 or self._client.ICP_47:
            shiny_sw_specs = ('rstudio_r4.2',)
            deprecated_shiny_sw_specs = ('shiny-r3.6',)
        else:
            shiny_sw_specs = ('shiny-r3.6',)
            deprecated_shiny_sw_specs = ()
        shiny_sw_spec_ids = [self._client.software_specifications.get_id_by_name(sw_name)
                               for sw_name in shiny_sw_specs]
        deprecated_shiny_sw_spec_ids = [self._client.software_specifications.get_id_by_name(sw_name)
                               for sw_name in deprecated_shiny_sw_specs]

        if shiny_sw_spec_id and shiny_sw_spec_id not in shiny_sw_spec_ids + deprecated_shiny_sw_spec_ids:
            raise WMLClientError(
                f"For R Shiny assets, only base software specs {','.join(shiny_sw_specs)} "
                "are supported. Specify "
                "the id you get via "
                "self._client.software_specifications.get_id_by_name(sw_name)")
        elif shiny_sw_spec_id and shiny_sw_spec_id in deprecated_shiny_sw_spec_ids:
            warnings.warn("Provided software spec is deprecated for R Shiny assets. "
                     f"Only base software specs {','.join(shiny_sw_specs)} "
                     "are supported. Specify "
                     "the id you get via "
                     "self._client.software_specifications.get_id_by_name(sw_name)")

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "shiny_asset",
                    "origin_country": "us",
                    "asset_category": "USER"
                }
            }
        else:
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "shiny_asset",
                    "origin_country": "us",
                    "asset_category": "USER"
                },
                 "entity": {
                    "shiny_asset": {
                        "ml_version": "4.0.0"
                     }
                 }
            }
            if (self._client.ICP_46 or self._client.ICP_47) and shiny_sw_spec_id:
                asset_meta['entity']['shiny_asset']['software_spec'] = \
                    {"base_id": shiny_sw_spec_id}


        base_shiny_asset = {
            "fields": [
                {
                    "key": "name",
                    "type": "string",
                    "is_array": False,
                    "is_searchable_across_types": False
                },
                {
                    "key": "entity",
                    "type": "object",
                    "is_array": False,
                    "is_searchable_across_types": False
                }
            ],
            "relationships": [],
            "name": "shiny_asset",
            "version": 1
        }

        #Step1  : Create an asset
        print("Creating Shiny asset...")

        if self._client.WSD:
            # For WSD the asset creation is done within _wsd_create_asset function using polyglot
            # Thus using the same for data_assets type


            meta_props = {
                    "name": name
            }
            details = Shiny._wsd_create_asset(self, "shiny_asset", asset_meta, meta_props, file_path, user_archive_file=True)
            return self._get_required_element_from_response(details)
        else:

            creation_response = ""
            if not self._ICP or self._client.ICP_PLATFORM_SPACES:
                creation_response = requests.post(
                        self._client.service_instance._href_definitions.get_data_assets_href(),
                        headers=self._client._get_headers(),
                        params = self._client._params(),
                        json=asset_meta
                )
            else:
                # Until global asset for shiny is created
                asset_type_response = requests.post(
                    self._client.service_instance._href_definitions.get_wsd_asset_type_href() + '?',
                    headers=self._client._get_headers(),
                    json=base_shiny_asset,
                    params=self._client._params()
                )
                if asset_type_response.status_code == 201 or asset_type_response.status_code == 409:
                    creation_response = requests.post(
                        self._client.service_instance._href_definitions.get_data_assets_href(),
                        headers=self._client._get_headers(),
                        json=asset_meta,
                        params=self._client._params()
                    )

            shiny_details = self._handle_response(201, u'creating new asset', creation_response)
            #Step2: Create attachment
            if creation_response.status_code == 201:
                asset_id = shiny_details["metadata"]["asset_id"]
                attachment_meta = {
                        "asset_type": "shiny_asset",
                        "name": "attachment_"+asset_id
                    }

                attachment_response = requests.post(
                    self._client.service_instance._href_definitions.get_attachments_href(asset_id),
                    headers=self._client._get_headers(),
                    params=self._client._params(),
                    json=attachment_meta
                )
                attachment_details = self._handle_response(201, u'creating new attachment', attachment_response)
                if attachment_response.status_code == 201:
                    attachment_id = attachment_details["attachment_id"]
                    attachment_url = attachment_details["url1"]


                    #Step3: Put content to attachment
                    try:
                        with open(file_path, 'rb') as f:
                            if not self._ICP:
                                put_response = requests.put(
                                    attachment_url,
                                    data=f.read(),
                                    # files = f
                                )
                            else:
                                put_response = requests.put(
                                    self._wml_credentials['url'] + attachment_url,
                                    files={'file': (name, f, 'application/octet-stream')}
                                )
                    except Exception as e:
                        deletion_response = requests.delete(
                            self._client.service_instance._href_definitions.get_data_asset_href(asset_id),
                            params=self._client._params(),
                            headers=self._client._get_headers()
                        )
                        print(deletion_response.status_code)
                        raise WMLClientError("Failed while reading a file", e)

                    if put_response.status_code == 201 or put_response.status_code == 200:
                        # Step4: Complete attachment
                        complete_response = requests.post(
                            self._client.service_instance._href_definitions.get_attachment_complete_href(asset_id, attachment_id),
                            headers=self._client._get_headers(),
                            params = self._client._params()

                        )

                        if complete_response.status_code == 200:
                            print("SUCCESS")
                            return self._get_required_element_from_response(shiny_details)
                        else:
                            self._delete(asset_id)
                            raise WMLClientError("Failed while creating a shiny asset. Try again.")
                    else:
                        self._delete(asset_id)
                        raise WMLClientError("Failed while creating a shiny asset. Try again.")
                else:
                    print("SUCCESS")
                    return self._get_required_element_from_response(shiny_details)
            else:
                raise WMLClientError("Failed while creating a shiny asset. Try again.")

    def list(self, limit=None, return_as_df=True):
        """Print stored shiny assets in a table format. If limit is set to None
        there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed shiny assets or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.shiny.list()

        """

        Shiny._validate_type(limit, u'limit', int, False)
        href = self._client.service_instance._href_definitions.get_search_shiny_href()

        data = {
                "query": "*:*"
        }
        if limit is not None:
            data.update({"limit": limit})

        response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(),json=data)
        self._handle_response(200, u'list assets', response)
        shiny_details = self._handle_response(200, u'list assets', response)["results"]
        space_values = [
            (m[u'metadata'][u'name'], m[u'metadata'][u'asset_type'], m["metadata"]["asset_id"]) for
            m in shiny_details]

        table = self._list(space_values, [u'NAME', u'ASSET_TYPE', u'ASSET_ID'], None, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def download(self, shiny_uid, filename, rev_uid=None):
        """Download the content of a shiny asset.

        :param shiny_uid: the Unique Id of the shiny asset to be downloaded
        :type shiny_uid: str
        :param filename: filename to be used for the downloaded file
        :type filename: str
        :param rev_uid: revision id
        :type rev_uid: str, optional

        :return: path to the downloaded shiny asset content
        :rtype: str

        **Example**

        .. code-block:: python

            client.shiny.download(shiny_uid, "shiny_asset.zip")

         """
        if rev_uid is not None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not applicable for this release')

        Shiny._validate_type(shiny_uid, u'shiny_uid', str, True)
        Shiny._validate_type(rev_uid, u'rev_uid', int, False)

        params = self._client._params()

        if rev_uid is not None:
            params.update({'revision_id': rev_uid})

        import urllib
        asset_response = requests.get(self._client.service_instance._href_definitions.get_data_asset_href(shiny_uid),
                                      params=params,
                                      headers=self._client._get_headers())
        shiny_details = self._handle_response(200, u'get shiny assets', asset_response)

        if self._WSD:
            attachment_url = shiny_details['attachments'][0]['object_key']
            artifact_content_url = self._client.service_instance._href_definitions.get_wsd_model_attachment_href() + \
                                   urllib.parse.quote('shiny_asset/' + attachment_url, safe='')

            r = requests.get(artifact_content_url, params=self._client._params(), headers=self._client._get_headers(),
                             stream=True)
            if r.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading shiny asset"), r)

            downloaded_asset = r.content
            try:
                with open(filename, 'wb') as f:
                    f.write(downloaded_asset)
                print(u'Successfully saved shiny asset content to file: \'{}\''.format(filename))
                return os.getcwd() + "/" + filename
            except IOError as e:
                raise WMLClientError(u'Saving shiny asset with artifact_url: \'{}\'  to local file failed.'.format(filename), e)
        else:
            attachment_id = shiny_details["attachments"][0]["id"]
            response = requests.get(self._client.service_instance._href_definitions.get_attachment_href(shiny_uid,attachment_id), params=params,
                                    headers=self._client._get_headers())
            if response.status_code == 200:
                attachment_signed_url = response.json()["url"]
                if 'connection_id' in shiny_details["attachments"][0]:
                    att_response = requests.get(attachment_signed_url)
                else:
                    if not self._ICP:
                        att_response = requests.get(attachment_signed_url)
                    else:
                        att_response = requests.get(self._wml_credentials["url"]+attachment_signed_url)

                if att_response.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading asset"), att_response)

                downloaded_asset = att_response.content
                try:
                    with open(filename, 'wb') as f:
                        f.write(downloaded_asset)
                    print(u'Successfully saved shiny asset content to file: \'{}\''.format(filename))
                    return os.getcwd() + "/" + filename
                except IOError as e:
                    raise WMLClientError(u'Saving shiny asset with artifact_url to local file: \'{}\' failed.'.format(filename), e)
            else:
                raise WMLClientError("Failed while downloading the shiny asset " + shiny_uid)

    @staticmethod
    def get_id(shiny_details):
        """Get Unique Id of stored shiny asset.

        :param shiny_details: metadata of the stored shiny asset
        :type shiny_details: dict

        :return: Unique Id of stored shiny asset
        :rtype: str

        **Example**

        .. code-block:: python

            shiny_uid = client.shiny.get_id(shiny_details)

        """

        return Shiny.get_uid(shiny_details)

    @staticmethod
    def get_uid(shiny_details):
        """Get Unique Id  of stored shiny asset.

        *Deprecated:* Use ``get_id(shiny_details)`` instead.

        :param shiny_details: metadata of the stored shiny asset
        :type shiny_details: dict

        :return: Unique Id of stored shiny asset
        :rtype: str

        **Example**

        .. code-block:: python

            shiny_uid = client.shiny.get_uid(shiny_details)

        """
        Shiny._validate_type(shiny_details, u'shiny_details', object, True)
        Shiny._validate_type_of_details(shiny_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(shiny_details, u'data_assets_details',
                                                           [u'metadata', u'guid'])

    @staticmethod
    def get_href(shiny_details):
        """Get url of stored shiny asset.

        :param shiny_details: stored shiny asset details
        :type shiny_details: dict

        :return: href of stored shiny asset
        :rtype: str

        **Example**

        .. code-block:: python

            shiny_details = client.shiny.get_details(shiny_uid)
            shiny_href = client.shiny.get_href(shiny_details)
        """
        Shiny._validate_type(shiny_details, u'shiny_details', object, True)
        Shiny._validate_type_of_details(shiny_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(shiny_details, u'shiny_details', [u'metadata', u'href'])

    def update(self, shiny_uid, meta_props=None, file_path=None):
        """Update shiny with either metadata or attachment or both.

        :param shiny_uid:  Shiny UID
        :type shiny_uid: str
        :param meta_props: changes to shiny metadata
        :type meta_props: dict, optional
        :param file_path: file path to new attachment
        :type file_path: str, optional

        :return: updated metadata of shiny asset
        :rtype: dict

        **Example**

        .. code-block:: python

            script_details = client.script.update(model_uid, meta, content_path)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        Shiny._validate_type(shiny_uid, 'shiny_uid', str, True)

        if meta_props is None and file_path is None:
            raise WMLClientError('Atleast either meta_props or file_path has to be provided')

        updated_details = None
        details = None

        url = self._client.service_instance._href_definitions.get_asset_href(shiny_uid)

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   CAMS has meta and entity patching. 'name' and 'description' get stored in CAMS meta section
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        # STEP 3. If file_path provided, we need to patch the attachment
        #   a. If attachment already exists for the script, delete it
        #   b. POST call to get signed url for upload
        #   c. Upload to the signed url
        #   d. Mark upload complete
        # STEP 4. Get the updated script record and return

        # STEP 1
        response = requests.get(
            url,
            params=self._client._params(),
            headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    u'Invalid input. Unable to get the details of shiny_uid provided.')
            else:
                raise ApiRequestFailure(u'Failure during {}.'.format("getting shiny asset to update"), response)

        details = self._handle_response(200, "Get shiny asset details", response)

        attachments_response = None

        # STEP 2a.
        # Patch meta if provided
        if meta_props is not None:
            self._validate_type(meta_props, u'meta_props', dict, True)

            meta_patch_payload = []

            # Since we are dealing with direct asset apis, name and description is metadata patch
            if "name" in meta_props or "description" in meta_props:
                props_for_asset_meta_patch = {}

                for key in meta_props:
                    if key == 'name' or key == 'description':
                        props_for_asset_meta_patch.update({key: meta_props[key]})

                meta_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details,
                                                                                         props_for_asset_meta_patch,
                                                                                         with_validation=True)
            if meta_patch_payload:
                meta_patch_url = self._client.service_instance._href_definitions.get_asset_href(shiny_uid)

                response_patch = requests.patch(meta_patch_url,
                                                json=meta_patch_payload,
                                                params=self._client._params(),
                                                headers=self._client._get_headers())

                updated_details = self._handle_response(200, u'shiny patch', response_patch)

        if file_path is not None:
            if "attachments" in details and details[u'attachments']:
                current_attachment_id = details[u'attachments'][0][u'id']
            else:
                current_attachment_id = None

            #STEP 3
            attachments_response = self._update_attachment_for_assets("shiny_asset",
                                                                      shiny_uid,
                                                                      file_path,
                                                                      current_attachment_id)

        if attachments_response is not None and 'success' not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._client.service_instance._href_definitions.get_asset_href(shiny_uid)

        response = requests.get(
            url,
            params=self._client._params(),
            headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    u'Invalid input. Unable to get the details of shiny_uid provided.')
            else:
                raise ApiRequestFailure(u'Failure during {}.'.format("getting shiny to update"), response)

        # response = self._handle_response(200, "Get shiny details", response)
        #
        # return self._get_required_element_from_response(response)

        response = self._get_required_element_from_response(self._handle_response(200, "Get shiny details", response))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            final_response = {
                "metadata": response[u'metadata'],
                "entity": {}
            }

            return final_response

    def _update_msg(self, updated_details):
        if updated_details is not None:
            print("Could not update the attachment because of server error."
                  " However metadata is updated. Try updating attachment again later")
        else:
            raise WMLClientError('Unable to update attachment because of server error. Try again later')

    def delete(self, shiny_uid):
        """Delete a stored shiny asset.

        :param shiny_uid: Unique Id of shiny asset
        :type shiny_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**
        
        .. code-block:: python

            client.shiny.delete(shiny_uid)

        """
        Shiny._validate_type(shiny_uid, u'shiny_uid', str, True)
        if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and \
                self._if_deployment_exist_for_asset(shiny_uid):
            raise WMLClientError(
                u'Cannot delete shiny asset that has existing deployments. Please delete all associated deployments and try again')

        response = requests.delete(self._client.service_instance._href_definitions.get_asset_href(shiny_uid), params=self._client._params(),
                                headers=self._client._get_headers())

        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, u'delete assets', response)

    def create_revision(self, shiny_uid):
        """Create revision for the given Shiny asset. Revisions are immutable once created.
        The metadata and attachment at `script_uid` is taken and a revision is created out of it.

        :param shiny_uid: shiny asset ID
        :type shiny_uid: str

        :return: stored shiny asset revisions metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            shiny_revision = client.shiny.create_revision(shiny_uid)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Shiny._validate_type(shiny_uid, u'shiny_uid', str, True)

        print("Creating shiny revision...")
        #
        # return  self._get_required_element_from_response(
        #     self._create_revision_artifact_for_assets(shiny_uid, 'Shiny'))

        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(shiny_uid, 'Shiny'))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            final_response = {
                "metadata": response[u'metadata'],
                "entity": {}
            }

            return final_response

    def list_revisions(self, shiny_uid, limit=None, return_as_df=True):
        """Print all revisions for the given shiny asset uid in a table format.

        :param shiny_uid: stored shiny asset ID
        :type shiny_uid: str
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed shiny revisions or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.shiny.list_revisions(shiny_uid)
        """

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Shiny._validate_type(shiny_uid, u'shiny_uid', str, True)

        url = self._client.service_instance._href_definitions.get_asset_href(shiny_uid) + "/revisions"
        # /v2/assets/{asset_id}/revisions returns 'results' object
        shiny_resources = self._get_with_or_without_limit(url,
                                                           limit,
                                                           'List Shiny revisions',
                                                           summary=None,
                                                           pre_defined=None)[u'resources']
        shiny_values = [
            (m[u'metadata'][u'asset_id'],
             m[u'metadata'][u'revision_id'],
             m[u'metadata'][u'name'],
             m[u'metadata'][u'commit_info'][u'committed_at']) for m in
            shiny_resources]

        table = self._list(shiny_values, [u'GUID', u'REVISION_ID', u'NAME', u'REVISION_COMMIT'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def get_revision_details(self, shiny_uid=None, rev_uid=None):
        """Get metadata of `shiny_uid` revision.

        :param shiny_uid: shiny asset ID
        :type shiny_uid: str, optional
        :param rev_uid: revision ID, if this parameter is not provided, returns latest revision if existing else error
        :type rev_uid: int, optional

        :return: stored shiny(s) metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            shiny_details = client.shiny.get_revision_details(shiny_uid, rev_uid)
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Shiny._validate_type(shiny_uid, u'shiny_uid', str, True)
        Shiny._validate_type(rev_uid, u'rev_uid', int, False)

        if rev_uid is None:
            rev_uid = 'latest'

        url = self._client.service_instance._href_definitions.get_asset_href(shiny_uid)
        # return self._get_required_element_from_response(self._get_with_or_without_limit(url,
        #                                        limit=None,
        #                                        op_name="asset_revision",
        #                                        summary=None,
        #                                        pre_defined=None,
        #                                        revision=rev_uid))
        resources = self._get_with_or_without_limit(url,
                                                    limit=None,
                                                    op_name="asset_revision",
                                                    summary=None,
                                                    pre_defined=None,
                                                    revision=rev_uid)['resources']
        responses = [self._get_required_element_from_response(resource) for resource in resources]

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return responses
        else:
            final_responses = [{
                "metadata": response[u'metadata'],
                "entity": {}
            } for response in responses]

            return final_responses

    def _delete(self, shiny_uid):
        Shiny._validate_type(shiny_uid, u'shiny_uid', str, True)

        response = requests.delete(self._client.service_instance._href_definitions.get_asset_href(shiny_uid), params=self._client._params(),
                                   headers=self._client._get_headers())

    # def get_href(self, shiny_uid):
    #     """
    #        Get metadata of stored space(s). If space UID is not specified, it returns all the spaces metadata.
    #
    #        **Parameters**
    #
    #        .. important::
    #             #. **space_uid**: Space UID (optional)\n
    #                **type**: str\n
    #             #. **limit**:  limit number of fetched records (optional)\n
    #                **type**: int\n
    #
    #        **Output**
    #
    #        .. important::
    #             **returns**: metadata of stored space(s)\n
    #             **return type**: dict
    #             dict (if UID is not None) or {"resources": [dict]} (if UID is None)\n
    #
    #        .. note::
    #             If UID is not specified, all spaces metadata is fetched\n
    #
    #        **Example**
    #
    #         >>> space_details = client.spaces.get_details(space_uid)
    #         >>> space_details = client.spaces.get_details()
    #     """
    #
    #
    #     Assets._validate_type(shiny_uid, u'shiny_uid', str, True)
    #
    #     response = requests.get(self._client.service_instance._href_definitions.get_data_asset_href(shiny_uid), params=self._client._params(),
    #                             headers=self._client._get_headers())
    #
    #     if response.status_code == 200:
    #         return response.json()["href"]
    #     else:
    #         return self._handle_response(200, u'spaces assets', response)

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data, u'shiny', dict)

        revision_id = None

        try:
            if self._client.default_space_id is not None:
                metadata = {'space_id': response_data['metadata']['space_id'],
                            'guid': response_data['metadata']['asset_id'],
                            'href': response_data['href'],
                            'name': response_data[u'metadata'][u'name'],
                            'asset_type': response_data['metadata']['asset_type'],
                            'created_at': response_data['metadata']['created_at'],
                            'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                            }
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    if "revision_id" in response_data[u'metadata']:
                        revision_id = response_data[u'metadata'][u'revision_id']
                        metadata.update({'revision_id': response_data[u'metadata'][u'revision_id']})

                    if "attachments" in response_data and response_data[u'attachments']:
                        metadata.update({'attachment_id': response_data[u'attachments'][0][u'id']})

                    if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                        metadata.update(
                            {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})

                new_el = {'metadata': metadata,
                          'entity': response_data['entity']
                          }

            elif self._client.default_project_id is not None:
                if self._client.WSD:

                    href = self._client.service_instance._href_definitions.get_base_asset_href(response_data['metadata']['asset_id']) + "?" + "project_id=" + response_data['metadata']['project_id']

                    metadata = {'project_id': response_data['metadata']['project_id'],
                                'guid': response_data['metadata']['asset_id'],
                                'href': href,
                                'name': response_data[u'metadata'][u'name'],
                                'asset_type': response_data['metadata']['asset_type'],
                                'created_at': response_data['metadata']['created_at']
                    }
                    if self._client.WSD_20 is not None:
                        if "revision_id" in response_data[u'metadata']:
                            revision_id = response_data[u'metadata'][u'revision_id']
                            metadata.update({'revision_id': response_data[u'metadata'][u'revision_id']})

                        if "attachments" in response_data and response_data[u'attachments']:
                            metadata.update({'attachment_id': response_data[u'attachments'][0][u'id']})

                        if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                            metadata.update(
                                {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})

                    new_el = {'metadata': metadata,
                              'entity': response_data['entity']
                              }
                    if 'usage' in response_data['metadata']:
                        new_el['metadata'].update(
                            {'last_updated_at': response_data['metadata']['usage']['last_updated_at']})
                    else:
                        new_el['metadata'].update(
                            {'last_updated_at': response_data['metadata']['last_updated_at']})

                else:
                    metadata = {'project_id': response_data['metadata']['project_id'],
                                'guid': response_data['metadata']['asset_id'],
                                'href': response_data['href'],
                                'name': response_data[u'metadata'][u'name'],
                                'asset_type': response_data['metadata']['asset_type'],
                                'created_at': response_data['metadata']['created_at'],
                                'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                                }

                    if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                        if "revision_id" in response_data[u'metadata']:
                            revision_id = response_data[u'metadata'][u'revision_id']
                            metadata.update({'revision_id': response_data[u'metadata'][u'revision_id']})

                        if "attachments" in response_data and response_data[u'attachments']:
                            metadata.update({'attachment_id': response_data[u'attachments'][0][u'id']})

                        if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                            metadata.update(
                                {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})

                    new_el = {'metadata': metadata,
                              'entity': response_data['entity']
                              }
            if 'description' in response_data['metadata']:
                new_el[u'metadata'].update({'description': response_data[u'metadata'][u'description']})

            if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and 'href' in response_data['metadata']:
                href_without_host = response_data['metadata']['href'].split('.com')[-1]
                new_el[u'metadata'].update({'href': href_without_host})
            return new_el
        except Exception as e:
            raise WMLClientError("Failed to read Response from down-stream service: " + response_data.text)
