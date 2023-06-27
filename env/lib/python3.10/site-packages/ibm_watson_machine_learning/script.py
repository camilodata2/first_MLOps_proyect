#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.utils import DATA_ASSETS_DETAILS_TYPE, modify_details_for_script_and_shiny
from ibm_watson_machine_learning.metanames import ScriptMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure, ForbiddenActionForGitBasedProject
import os
import warnings

_DEFAULT_LIST_LENGTH = 50


class Script(WMLResource):
    """Store and manage scripts assets."""

    ConfigurationMetaNames = ScriptMetaNames()
    """MetaNames for script assets creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    def get_details(self, script_uid=None):
        """Get script asset details. If no script_uid is passed, details for all script assets will be returned.

        :param script_uid: Unique id of script
        :type script_uid: str, optional

        :return: metadata of the stored script asset
        :rtype:
            - **dict** - if runtime_uid is not None
            - **{"resources": [dict]}** - if runtime_uid is None

        **Example**

        .. code-block:: python

            script_details = client.script.get_details(script_uid)

        """
        def get_required_elements(response):
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:
                response = modify_details_for_script_and_shiny(response)
                final_response = {
                    "metadata": response[u'metadata'],
                }

                if 'entity' in response:
                    final_response['entity'] = response[u'entity']

                    try:
                        del final_response['entity'][u'script'][u'ml_version']
                    except KeyError:
                        pass

                return final_response

        return self._get_asset_based_resource(script_uid, 'script', get_required_elements)

    def store(self, meta_props, file_path):
        """Create a script asset and upload content to it.

        :param meta_props: name to be given to the script asset
        :type meta_props: str
        :param file_path: path to the content file to be uploaded
        :type file_path: str

        :return: metadata of the stored script asset
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.script.ConfigurationMetaNames.NAME: 'my first script',
                client.script.ConfigurationMetaNames.DESCRIPTION: 'description of the script',
                client.script.ConfigurationMetaNames.SOFTWARE_SPEC_UID: '0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda'
            }

            asset_details = client.script.store(meta_props=metadata, file_path="/path/to/file")

        """
        if self._client.project_type == 'local_git_storage':
            raise ForbiddenActionForGitBasedProject(reason="Storing Scripts is not supported for git based project.")


        WMLResource._chk_and_block_create_update_for_python36(self)
        #Script._validate_type(name, u'name', str, True)
        Script._validate_type(file_path, u'file_path', str, True)
        script_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client
        )

        name, extension = os.path.splitext(file_path)

        response = self._create_asset(script_meta, file_path, extension)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:

            entity = response[u'entity']

            try:
                del entity[u'script'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

    def _create_asset(self, script_meta, file_path, extension='.py'):

        ##Step1: Create a data asset
        name = script_meta['metadata'][u'name']

        if script_meta['metadata'].get('description') is not None:
            desc = script_meta['metadata'][u'description']
        else:
            desc = ""

        if script_meta.get('software_spec_uid') is not None:
            if script_meta[u'software_spec_uid'] == "":
                raise WMLClientError("Failed while creating a script asset, SOFTWARE_SPEC_UID cannot be empty")
                return

        base_script_asset = {
            "description": "Script File",
            "fields": [
                {
                    "key": "language",
                    "type": "string",
                    "facet": False,
                    "is_array": False,
                    "search_path": "asset.type",
                    "is_searchable_across_types": True
                }
            ],
            "relationships": [
                {
                    "key": "software_spec.id",
                    "target_asset_type": "software_specification",
                    "on_delete_target": "IGNORE",
                    "on_delete": "IGNORE",
                    "on_clone_target": "IGNORE"
                }
            ],
            "name": "script",
            "version": 1
        }

        if extension == '.py':
            lang = 'python3'
        elif extension == '.R':
            lang = 'R'
        else:
            raise WMLClientError("This file type is not supported. It has to be either a python script(.py file ) or a "
                                 "R script")

        #check if the software spec specified is base or derived and
        # accordingly update the entity for asset creation
        # TODO WSD2.0
        #kaganesa: remove the below if and else. once v2/software_spec apis are available in WSD2.0,
        # everything inside if should work.
        if not self._client.WSD:
            sw_spec_details = self._client.software_specifications.get_details(script_meta[u'software_spec_uid'])

            if lang == 'R':
                if self._client.ICP_47:
                    r_sw_specs = []
                    for sw_spec_details in self._client.software_specifications.get_details()['resources']:
                        sw_configuration = sw_spec_details['entity'].get('software_specification', {}).get(
                            'software_configuration', {})
                        if sw_configuration.get('platform', {}).get('name') == 'r' and len(
                                sw_configuration.get('included_packages', [])) > 1:
                            r_sw_specs.append(sw_spec_details['metadata']['name'])
                    rscript_sw_specs = []               # names of supported r script software specifications
                    deprecated_rscript_sw_specs = []    # names of deprecated r script software specifications
                    for sw_spec in self._client.software_specifications.get_details(state_info=True)['resources']:
                        if sw_spec['name'] in r_sw_specs:
                            if sw_spec['state'] == 'supported':
                                rscript_sw_specs.append(sw_spec['name'])
                            elif sw_spec['state'] == 'deprecated':
                                deprecated_rscript_sw_specs.append(sw_spec['name'])

                elif self._client.ICP_46:
                    rscript_sw_specs = ('runtime-22.2-r4.2',)
                    deprecated_rscript_sw_specs = ('default_r3.6', 'runtime-22.1-r3.6',)
                else:
                    rscript_sw_specs = ('default_r3.6', 'runtime-22.1-r3.6',)
                    deprecated_rscript_sw_specs = ()
                rscript_sw_spec_ids = [self._client.software_specifications.get_id_by_name(sw_name)
                                       for sw_name in rscript_sw_specs]
                deprecated_sw_spec_ids = [self._client.software_specifications.get_id_by_name(sw_name)
                                       for sw_name in deprecated_rscript_sw_specs]

                if script_meta[u'software_spec_uid'] not in rscript_sw_spec_ids + deprecated_sw_spec_ids:
                    raise WMLClientError(f"For R scripts, only base software specs {','.join(rscript_sw_specs)} "
                                         "are supported. Specify "
                                         "the id you get via "
                                         "self._client.software_specifications.get_id_by_name(sw_name)")
                elif script_meta[u'software_spec_uid'] in deprecated_rscript_sw_specs:
                    warnings.warn("Provided software spec is deprecated for R scripts. "
                             f"Only base software specs {','.join(rscript_sw_specs)} "
                             "are supported. Specify "
                             "the id you get via "
                             "self._client.software_specifications.get_id_by_name(sw_name)")


            if(sw_spec_details[u'entity'][u'software_specification'][u'type'] == 'base'):
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                    asset_meta = {
                        "metadata": {
                            "name": name,
                            "description": desc,
                            "asset_type": "script",
                            "origin_country": "us",
                            "asset_category": "USER"
                        },
                        "entity": {
                            "script": {
                                "language": {
                                    # "name": "python3"
                                    "name": lang
                    },
                                "software_spec": {
                                    "base_id": script_meta[u'software_spec_uid']
                                }
                             }
                            }
                    }
                else:
                     asset_meta = {
                         "metadata": {
                             "name": name,
                             "description": desc,
                             "asset_type": "script",
                             "origin_country": "us",
                             "asset_category": "USER"
                         },
                         "entity": {
                             "script": {
                                 "ml_version": "4.0.0",
                                 "language": {
                                     # "name": "python3"
                                     "name": lang
                     },
                                 "software_spec": {
                                     "base_id": script_meta[u'software_spec_uid']
                                 }
                             }
                         }
                     }
            else:
                if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                    asset_meta = {
                        "metadata": {
                            "name": name,
                            "description": desc,
                            "asset_type": "script",
                            "origin_country": "us",
                            "asset_category": "USER"
                        },
                        "entity": {
                            "script": {
                                "language": {
                                    # "name": "python3"
                                    "name": lang
                    },
                                "software_spec": {
                                    "id": script_meta[u'software_spec_uid']
                                }
                            }
                        }
                    }
                else:
                    asset_meta = {
                        "metadata": {
                            "name": name,
                            "description": desc,
                            "asset_type": "script",
                            "origin_country": "us",
                            "asset_category": "USER"
                        },
                        "entity": {
                            "script": {
                                "ml_version": "4.0.0",
                                "language": {
                                    # "name": "python3"
                                    "name": lang
                                },
                                "software_spec": {
                                    "id": script_meta[u'software_spec_uid']
                                }
                            }
                        }
                    }
        else:
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "script",
                    "origin_country": "us",
                    "asset_category": "USER"
                },
                "entity": {
                    "script": {
                        "language": {
                            # "name": "python3"
                            "name": lang
            },
                        "software_spec": {
                            "id": script_meta[u'software_spec_uid']
                        }
                    }
                }
            }

        #Step1  : Create an asset
        print("Creating Script asset...")

        if self._client.WSD:
            # For WSD the asset creation is done within _wsd_create_asset function using polyglot
            # Thus using the same for data_assets type


            meta_props = {
                    "name": name
            }
            details = Script._wsd_create_asset(self, "script", asset_meta, meta_props, file_path, user_archive_file=True)
            return self._get_required_element_from_response(details)
        else:
            if not self._ICP:
                creation_response = requests.post(
                        self._client.service_instance._href_definitions.get_assets_href(),
                        headers=self._client._get_headers(),
                        params = self._client._params(),
                        json=asset_meta
                )
            else:
                if self._client.ICP_PLATFORM_SPACES:
                    creation_response = requests.post(
                        self._client.service_instance._href_definitions.get_data_assets_href(),
                        headers=self._client._get_headers(),
                        json=asset_meta,
                        params=self._client._params()
                    )
                else:
                    asset_type_response = requests.post(
                        self._client.service_instance._href_definitions.get_wsd_asset_type_href() + '?',
                        headers=self._client._get_headers(),
                        json=base_script_asset,
                        params=self._client._params()
                    )
                    if asset_type_response.status_code == 201 or asset_type_response.status_code == 409:
                        creation_response = requests.post(
                            self._client.service_instance._href_definitions.get_data_assets_href(),
                            headers=self._client._get_headers(),
                            json=asset_meta,
                            params=self._client._params()
                        )

            asset_details = self._handle_response(201, u'creating new asset', creation_response)
            #Step2: Create attachment
            if creation_response.status_code == 201:
                asset_id = asset_details["metadata"]["asset_id"]
                attachment_meta = {
                        "asset_type": "script",
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
                                    data=f.read()
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
                        raise WMLClientError("Failed while reading a file.", e)

                    if put_response.status_code == 201 or put_response.status_code == 200:
                        # Step4: Complete attachment
                        complete_response = requests.post(
                            self._client.service_instance._href_definitions.get_attachment_complete_href(asset_id, attachment_id),
                            headers=self._client._get_headers(),
                            params = self._client._params()

                        )

                        if complete_response.status_code == 200:
                            print("SUCCESS")
                            return self._get_required_element_from_response(asset_details)
                        else:
                            self._delete(asset_id)
                            raise WMLClientError("Failed while creating a script asset. Try again.")
                    else:
                        self._delete(asset_id)
                        raise WMLClientError("Failed while creating a script asset. Try again.")
                else:
                    print("SUCCESS")
                    return self._get_required_element_from_response(asset_details)
            else:
                raise WMLClientError("Failed while creating a script asset. Try again.")


    def list(self, limit=None, return_as_df=True):
        """Print stored scripts in a table format. If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed scripts or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.script.list()
        """

        Script._validate_type(limit, u'limit', int, False)
        href = self._client.service_instance._href_definitions.get_search_script_href()

        data = {
                "query": "*:*"
        }
        if limit is not None:
            data.update({"limit": limit})

        response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(),json=data)

        self._handle_response(200, u'list assets', response)
        asset_details = self._handle_response(200, u'list assets', response)["results"]
        space_values = [
            (m[u'metadata'][u'name'], m[u'metadata'][u'asset_type'], m["metadata"]["asset_id"]) for
            m in asset_details]

        table = self._list(space_values, [u'NAME', u'ASSET_TYPE', u'ASSET_ID'], None, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def download(self, asset_uid, filename, rev_uid=None):
        """Download the content of a script asset.

        :param asset_uid: the Unique Id of the script asset to be downloaded
        :type asset_uid: str
        :param filename: filename to be used for the downloaded file
        :type filename: str
        :param rev_uid: revision id
        :type rev_uid: str, optional

        :return: path to the downloaded asset content
        :rtype: str

        **Example**

        .. code-block:: python

            client.script.download(asset_uid, "script_file")
        """
        if rev_uid is not None and not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not applicable for this release')

        Script._validate_type(asset_uid, u'asset_uid', str, True)
        Script._validate_type(rev_uid, u'rev_uid', int, False)

        params = self._client._params()

        if rev_uid is not None:
            params.update({'revision_id': rev_uid})

        import urllib
        if not self._ICP:
            asset_response = requests.get(self._client.service_instance._href_definitions.get_asset_href(asset_uid),
                                          params=params,
                                          headers=self._client._get_headers())
        else:
            asset_response = requests.get(self._client.service_instance._href_definitions.get_data_asset_href(asset_uid),
                                          params=params,
                                          headers=self._client._get_headers())
        asset_details = self._handle_response(200, u'get assets', asset_response)

        if self._WSD:
            attachment_url = asset_details['attachments'][0]['object_key']
            artifact_content_url = self._client.service_instance._href_definitions.get_wsd_model_attachment_href() + \
                                   urllib.parse.quote('script/' + attachment_url, safe='')

            r = requests.get(artifact_content_url, params=self._client._params(), headers=self._client._get_headers(),
                             stream=True)
            if r.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading data asset"), r)

            downloaded_asset = r.content
            try:
                with open(filename, 'wb') as f:
                    f.write(downloaded_asset)
                print(u'Successfully saved data asset content to file: \'{}\''.format(filename))
                return os.path.abspath(filename)
            except IOError as e:
                raise WMLClientError(u'Saving data asset with artifact_url: \'{}\'  to local file failed.'.format(filename), e)
        else:
            attachment_id = asset_details["attachments"][0]["id"]

            response = requests.get(self._client.service_instance._href_definitions.get_attachment_href(asset_uid,attachment_id), params=params,
                                    headers=self._client._get_headers())

            if response.status_code == 200:
                attachment_signed_url = response.json()["url"]
                if 'connection_id' in asset_details["attachments"][0]:
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
                    print(u'Successfully saved data asset content to file: \'{}\''.format(filename))
                    return os.path.abspath(filename)
                except IOError as e:
                    raise WMLClientError(u'Saving asset with artifact_url to local file: \'{}\' failed.'.format(filename), e)
            else:
                raise WMLClientError("Failed while downloading the asset " + asset_uid)

    @staticmethod
    def get_id(asset_details):
        """Get Unique Id of stored script asset.

        :param asset_details: metadata of the stored script asset
        :type asset_details: dict

        :return: Unique Id of stored script asset
        :rtype: str

        **Example**

        .. code-block:: python

            asset_uid = client.script.get_id(asset_details)
        """

        return Script.get_uid(asset_details)

    @staticmethod
    def get_uid(asset_details):
        """Get Unique Id  of stored script asset.

        *Deprecated:* Use ``get_id(asset_details)`` instead.

        :param asset_details: metadata of the stored script asset
        :type asset_details: dict

        :return: Unique Id of stored script asset
        :rtype: str

        **Example**

        .. code-block:: python

            asset_uid = client.script.get_uid(asset_details)
        """
        Script._validate_type(asset_details, u'asset_details', object, True)
        Script._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(asset_details, u'data_assets_details',
                                                           [u'metadata', u'guid'])

    @staticmethod
    def get_href(asset_details):
        """Get url of stored scripts asset.

        :param asset_details: stored script details
        :type asset_details: dict

        :return: href of stored script asset
        :rtype: str

        **Example**

        .. code-block:: python

            asset_details = client.script.get_details(asset_uid)
            asset_href = client.script.get_href(asset_details)

        """
        Script._validate_type(asset_details, u'asset_details', object, True)
        Script._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(asset_details, u'asset_details', [u'metadata', u'href'])

    def update(self, script_uid, meta_props=None, file_path=None):
        """Update script with either metadata or attachment or both.

        :param script_uid: script UID
        :type script_uid: str
        :param meta_props: changes for script matadata
        :type meta_props: dict, optional
        :param file_path: file path to new attachment
        :type file_path: str, optional

        :return: updated metadata of script
        :rtype: dict

        **Example**

        .. code-block:: python

            script_details = client.script.update(model_uid, meta, content_path)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        # We need to enable this once we add functionality for WSD
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        Script._validate_type(script_uid, 'script_uid', str, True)

        if meta_props is None and file_path is None:
            raise WMLClientError('Atleast either meta_props or file_path has to be provided')

        updated_details = None
        details = None

        url = self._client.service_instance._href_definitions.get_asset_href(script_uid)

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   CAMS has meta and entity patching. 'name' and 'description' get stored in CAMS meta section
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        #   b. Construct entity patch if required and call /v2/assets/<asset_id>/attributes/script to patch entity
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
                    u'Invalid input. Unable to get the details of script_uid provided.')
            else:
                raise ApiRequestFailure(u'Failure during {}.'.format("getting script to update"), response)

        details = self._handle_response(200, "Get script details", response)

        attachments_response = None

        # STEP 2a.
        # Patch meta if provided
        if meta_props is not None:
            self._validate_type(meta_props, u'meta_props', dict, True)

            meta_patch_payload = []
            entity_patch_payload = []

            # Since we are dealing with direct asset apis, there can be metadata or entity patch or both
            if "name" in meta_props or "description" in meta_props:
                props_for_asset_meta_patch = {}

                for key in meta_props:
                    if key == 'name' or key == 'description':
                        props_for_asset_meta_patch.update({key: meta_props[key]})

                meta_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details,
                                                                                         props_for_asset_meta_patch,
                                                                                         with_validation=True)
            # STEP 2b.
            if "software_spec_uid" in meta_props:
                if details[u'entity'][u'script'][u'software_spec']:
                    entity_patch_payload = [{'op':'replace',
                                             'path': '/software_spec/base_id',
                                             'value': meta_props[u'software_spec_uid']}]
                else:
                    entity_patch_payload = [{'op': 'add',
                                             'path': '/software_spec',
                                             'value': '{base_id:' + meta_props[u'software_spec_uid'] + '}'}]

            if meta_patch_payload:
                meta_patch_url = self._client.service_instance._href_definitions.get_asset_href(script_uid)

                response_patch = requests.patch(meta_patch_url,
                                                json=meta_patch_payload,
                                                params=self._client._params(),
                                                headers=self._client._get_headers())

                updated_details = self._handle_response(200, u'script patch', response_patch)

            if entity_patch_payload:
                entity_patch_url = self._client.service_instance._href_definitions.get_asset_href(script_uid) + '/attributes/script'

                response_patch = requests.patch(entity_patch_url,
                                                json=entity_patch_payload,
                                                params=self._client._params(),
                                                headers=self._client._get_headers())

                updated_details = self._handle_response(200, u'script patch', response_patch)

        if file_path is not None:
            if "attachments" in details and details[u'attachments']:
                current_attachment_id = details[u'attachments'][0][u'id']
            else:
                current_attachment_id = None

            #STEP 3
            attachments_response = self._update_attachment_for_assets("script",
                                                                      script_uid,
                                                                      file_path,
                                                                      current_attachment_id)

        if attachments_response is not None and 'success' not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._client.service_instance._href_definitions.get_asset_href(script_uid)

        response = requests.get(
            url,
            params=self._client._params(),
            headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    u'Invalid input. Unable to get the details of script_uid provided.')
            else:
                raise ApiRequestFailure(u'Failure during {}.'.format("getting script to update"), response)

        # response = self._handle_response(200, "Get script details", response)

        # return self._get_required_element_from_response(response)

        response = self._get_required_element_from_response(self._handle_response(200, "Get script details", response))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            entity = response[u'entity']

            try:
                del entity[u'script'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

    def _update_msg(self, updated_details):
        if updated_details is not None:
            print("Could not update the attachment because of server error."
                  " However metadata is updated. Try updating attachment again later")
        else:
            raise WMLClientError('Unable to update attachment because of server error. Try again later')

    def delete(self, asset_uid):
        """Delete a stored script asset.

        :param asset_uid: UID of script asset
        :type asset_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.script.delete(asset_uid)

        """
        Script._validate_type(asset_uid, u'asset_uid', str, True)
        if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and \
                self._if_deployment_exist_for_asset(asset_uid):
            raise WMLClientError(
                u'Cannot delete script that has existing deployments. Please delete all associated deployments and try again')

        response = requests.delete(self._client.service_instance._href_definitions.get_asset_href(asset_uid), params=self._client._params(),
                                headers=self._client._get_headers())

        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, u'delete assets', response)

    def create_revision(self, script_uid):
        """Create revision for the given script. Revisions are immutable once created.
        The metadata and attachment at `script_uid` is taken and a revision is created out of it.

        :param script_uid: script ID
        :type script_uid: str

        :return: stored script revisions metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            script_revision = client.script.create_revision(script_uid)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        # We need to enable this once we add functionality for WSD
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Script._validate_type(script_uid, u'script_uid', str, True)

        print("Creating script revision...")

        # return self._get_required_element_from_response(
        #     self._create_revision_artifact_for_assets(script_uid, 'Script'))

        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(script_uid, 'Script'))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            entity = response[u'entity']

            try:
                del entity[u'script'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

    def list_revisions(self, script_uid, limit=None, return_as_df=True):
        """Print all revisions for the given script uid in a table format.

        :param script_uid: stored script ID
        :type script_uid: str
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed revisions or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.script.list_revisions(script_uid)
        """
        # We need to enable this once we add functionality for WSD
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Script._validate_type(script_uid, u'script_uid', str, True)

        url = self._client.service_instance._href_definitions.get_asset_href(script_uid) + "/revisions"
        # /v2/assets/{asset_id}/revisions returns 'results' object
        script_resources = self._get_with_or_without_limit(url,
                                                           limit,
                                                           'List Script revisions',
                                                           summary=None,
                                                           pre_defined=None)[u'resources']
        script_values = [
            (m[u'metadata'][u'asset_id'],
             m[u'metadata'][u'revision_id'],
             m[u'metadata'][u'name'],
             m[u'metadata'][u'commit_info'][u'committed_at']) for m in
            script_resources]

        table = self._list(script_values, [u'GUID', u'REVISION_ID', u'NAME', u'REVISION_COMMIT'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def get_revision_details(self, script_uid, rev_uid=None):
        """Get metadata of script revision.

        :param script_uid: script ID
        :type script_uid: str
        :param rev_uid: revision ID, if this parameter is not provided, returns latest revision if existing else error
        :type rev_uid: int, optional

        :return: stored script(s) metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            script_details = client.script.get_revision_details(script_uid, rev_uid)
        """
        # We need to enable this once we add functionality for WSD
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Script._validate_type(script_uid, u'script_uid', str, True)
        Script._validate_type(rev_uid, u'rev_uid', int, False)

        if rev_uid is None:
            rev_uid = 'latest'

        url = self._client.service_instance._href_definitions.get_asset_href(script_uid)
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
            final_responses = []
            for response in responses:
                entity = response[u'entity']

                try:
                    del entity[u'script'][u'ml_version']
                except KeyError:
                    pass

                final_responses.append({
                    "metadata": response[u'metadata'],
                    "entity": entity
                })

            return final_responses

    def _delete(self, asset_uid):
        Script._validate_type(asset_uid, u'asset_uid', str, True)

        response = requests.delete(self._client.service_instance._href_definitions.get_asset_href(asset_uid), params=self._client._params(),
                                   headers=self._client._get_headers())

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data, u'scripts', dict)

        revision_id = None

        try:
            if self._client.default_space_id is not None:
                metadata = {'space_id': response_data['metadata']['space_id'],
                            'name':response_data['metadata']['name'],
                            'guid': response_data['metadata']['asset_id'],
                            'href': response_data['href'],
                            'asset_type': response_data['metadata']['asset_type'],
                            'created_at': response_data['metadata']['created_at'],
                            'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                            }
                if 'description' in response_data[u'metadata']:
                    metadata.update(
                        {'description':response_data[u'metadata'][u'description']})

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
                                'name': response_data['metadata']['name'],
                                'href': href,
                                'asset_type': response_data['metadata']['asset_type'],
                                'created_at': response_data['metadata']['created_at']
                                }

                    if 'description' in response_data[u'metadata']:
                        metadata.update(
                            {'description': response_data[u'metadata'][u'description']})

                    new_el = {'metadata': metadata,
                              'entity': response_data['entity']
                              }
                    if self._client.WSD_20 is not None:
                        if "revision_id" in response_data[u'metadata']:
                            revision_id = response_data[u'metadata'][u'revision_id']
                            new_el['metadata'].update({'revision_id': response_data[u'metadata'][u'revision_id']})

                        if "attachments" in response_data and response_data[u'attachments']:
                            new_el['metadata'].update({'attachment_id': response_data[u'attachments'][0][u'id']})

                        if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                            new_el['metadata'].update(
                                {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})

                    if 'usage' in response_data['metadata']:
                        new_el['metadata'].update({'last_updated_at': response_data['metadata']['usage']['last_updated_at']})
                    else:
                        new_el['metadata'].update(
                         {'last_updated_at': response_data['metadata']['last_updated_at']})
                else:
                    metadata = {'project_id': response_data['metadata']['project_id'],
                                'guid': response_data['metadata']['asset_id'],
                                'href': response_data['href'],
                                'name': response_data['metadata']['name'],
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

                    if 'description' in response_data[u'metadata']:
                        metadata.update(
                            {'description': response_data[u'metadata'][u'description']})
                    new_el = {'metadata': metadata,
                              'entity': response_data['entity']
                              }
                    if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                        href_without_host = response_data['href'].split('.com')[-1]
                        new_el[u'metadata'].update({'href': href_without_host})
            return new_el
        except Exception as e:
            raise WMLClientError(f"Failed to read Response from down-stream service: {response_data}")
