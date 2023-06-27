#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.metanames import ModelDefinitionMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError
import os
import json
import uuid

_DEFAULT_LIST_LENGTH = 50


class ModelDefinition(WMLResource):
    """Store and manage model definitions."""

    ConfigurationMetaNames = ModelDefinitionMetaNames()

    """MetaNames for model definition creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP
        self.default_space_id = client.default_space_id

    def _generate_model_definition_document(self, meta_props):
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            doc = {
                "metadata":
                {
                   "name": "generated_name_"+str(uuid.uuid4()),
                   "tags": ["generated_tag_"+str(uuid.uuid4())],
                   "asset_type": "wml_model_definition",
                   "origin_country": "us",
                   "rov": {
                      "mode": 0
                   },
                   "asset_category": "USER"
                },
                "entity": {
                   "wml_model_definition": {
                       "ml_version": "4.0.0",
                       "version": "1.0",
                       "platform": {
                          "name": "python",
                          "versions": [
                              "3.5"
                             ]
                       }
                   }
                }
            }

        else:
            doc = {
                "metadata":
                {
                   "name": "My wml_model_definition assert",
                   "tags": ["string"],
                   "asset_type": "wml_model_definition",
                   "origin_country": "us",
                   "rov": {
                      "mode": 0
                   },
                   "asset_category": "USER"
                },
                "entity": {
                   "wml_model_definition": {
                       "name": "tf-model_trainings_v4_test_suite_basic",
                       "description": "Sample custom library",
                       "version": "1.0",
                       "platform": {
                          "name": "python",
                          "versions": [
                              "3.5"
                             ]
                       }
                   }
                }
            }

        if self.ConfigurationMetaNames.NAME in meta_props:
            doc["metadata"]["name"] = meta_props[self.ConfigurationMetaNames.NAME]
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                # We shouldn't have name and description in entity but this code exists in CP4D
                # So, changing only for Cloud Convergence
                doc["entity"]["wml_model_definition"]["name"] = meta_props[self.ConfigurationMetaNames.NAME]
        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            doc["metadata"]["description"] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                doc["entity"]["wml_model_definition"]["description"] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

        if self.ConfigurationMetaNames.VERSION in meta_props:
            doc["entity"]["wml_model_definition"]["version"] = meta_props[self.ConfigurationMetaNames.VERSION]

        if self.ConfigurationMetaNames.PLATFORM in meta_props:
            doc["entity"]["wml_model_definition"]["platform"]["name"] = meta_props[self.ConfigurationMetaNames.PLATFORM]['name']
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                doc["entity"]["wml_model_definition"]["platform"]["versions"] = \
                    meta_props[self.ConfigurationMetaNames.PLATFORM]['versions']
            else:
                doc["entity"]["wml_model_definition"]["platform"]["versions"][0] = \
                    meta_props[self.ConfigurationMetaNames.PLATFORM]['versions'][0]

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            if self.ConfigurationMetaNames.COMMAND in meta_props:
                doc['entity']['wml_model_definition'].update({"command": meta_props[self.ConfigurationMetaNames.COMMAND]})
            if self.ConfigurationMetaNames.CUSTOM in meta_props:
                doc['entity']['wml_model_definition'].update(
                    {"custom": meta_props[self.ConfigurationMetaNames.CUSTOM]})

        return doc

    def store(self, model_definition, meta_props):
        """Create a model definition.

        :param meta_props: meta data of the model definition configuration, to see available meta names use:

            .. code-block:: python

                client.model_definitions.ConfigurationMetaNames.get()

        :type meta_props: dict

        :param model_definition: path to the content file to be uploaded
        :type model_definition: str

        :return: metadata of the model definition created
        :rtype: dict

        **Example**

        .. code-block:: python

            client.model_definitions.store(model_definition, meta_props)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        self.ConfigurationMetaNames._validate(meta_props)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:

            # metadata doesn't seem to be used at all.. Code existing pre-convergence.
            # Should be checked sometime and removed if not needed
            metadata = {
                self.ConfigurationMetaNames.NAME: meta_props[self.ConfigurationMetaNames.NAME],
                self.ConfigurationMetaNames.VERSION: meta_props['version'],
                self.ConfigurationMetaNames.PLATFORM:
                    meta_props['platform']
                    if 'platform' in meta_props and meta_props['platform'] is not None
                    else {
                        "name": meta_props[self.ConfigurationMetaNames.PLATFORM]['name'],
                        "versions": [meta_props[self.ConfigurationMetaNames.PLATFORM]['versions']]
                    },
            }

            if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
                metadata[self.ConfigurationMetaNames.DESCRIPTION] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

            if self.ConfigurationMetaNames.SPACE_UID in meta_props:
                metadata[self.ConfigurationMetaNames.SPACE_UID] = {
                    "href": "/v4/spaces/" + meta_props[self.ConfigurationMetaNames.SPACE_UID]
                }

            # Following is not used for model_definitions since space_id/project_id
            # is passed as a query param. Code is existing pre-cloud-convergence
            if self._client.CAMS:
                if self._client.default_space_id is not None:
                    metadata['space'] = {'href': "/v4/spaces/"+self._client.default_space_id}
                elif self._client.default_project_id is not None:
                    metadata['project'] = {'href': "/v2/projects/"+self._client.default_project_id}
                else:
                    raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        document = self._generate_model_definition_document(meta_props)

        if self._client.WSD:
            return self._wsd_create_asset("wml_model_definition", document, meta_props, model_definition,user_archive_file=True)

        model_definition_attachment_def = {
          "asset_type": "wml_model_definition",
          "name": "model_definition_attachment"
        }
        paramvalue = self._client._params()

        creation_response = requests.post(
             self._client.service_instance._href_definitions.get_model_definition_assets_href(),
             params=paramvalue,
             headers=self._client._get_headers(),
             json=document)

        model_definition_details = self._handle_response(201, u'creating new model_definition', creation_response)

        self._handle_response(201, 'creating new attachment', creation_response)
        model_definition_id = model_definition_details['metadata']['asset_id']

        if self._client.CLOUD_PLATFORM_SPACES:
            model_definition_attachment_url = self._client.service_instance._href_definitions.get_attachments_href(model_definition_details['metadata']['asset_id'])
        else:
            model_definition_attachment_url = self._client.service_instance._href_definitions.get_model_definition_assets_href() + "/" + \
                                              model_definition_details['metadata']['asset_id'] + "/attachments"

        put_header = self._client._get_headers(no_content_type=True)

        attachment_response = requests.post(
            model_definition_attachment_url,
            params=paramvalue,
            headers=self._client._get_headers(),
            json=model_definition_attachment_def)

        try:
            attachment_details = self._handle_response(201, u'creating new model definition attachment', attachment_response)

            attachment_id = attachment_details['attachment_id']
            attachment_status_json = json.loads(attachment_response.content.decode("utf-8"))
            model_definition_attachment_signed_url = attachment_status_json["url1"]
         #   print("WML model_definition attachment url1: %s" % model_definition_attachement_signed_url)
            model_definition_attachment_put_url = self._client.wml_credentials['url'] + model_definition_attachment_signed_url

            with open(model_definition, 'rb') as f:
                if self._ICP:
                    put_response = requests.put(model_definition_attachment_put_url,
                                            files={'file': (model_definition, f, 'application/octet-stream')})
                else:
                    put_response = requests.put(model_definition_attachment_signed_url,
                                                data=open(model_definition, 'rb').read())

            if put_response.status_code != 201 and put_response.status_code != 200:
                self._handle_response(200, 'uploading a model_definition attachment file', put_response)

            complete_response = requests.post( self._client.service_instance._href_definitions.get_attachment_complete_href(model_definition_id, attachment_id),
                                              params=paramvalue,
                                              headers=self._client._get_headers())

            self._handle_response(200, 'updating a model_definition status', complete_response)

            response = self._get_required_element_from_response(model_definition_details)

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:
                entity = response[u'entity']

                try:
                    del entity[u'wml_model_definition'][u'ml_version']
                except KeyError:
                    pass

                final_response = {
                    "metadata": response[u'metadata'],
                    "entity": entity
                }

                return final_response
            # return self._get_required_element_from_response(model_definition_details)
        except Exception as e:
            try:
                self._delete(model_definition_id)
            finally:
                raise e

    def get_details(self, model_definition_uid=None):
        """Get metadata of stored model definition. If no `model_definition_uid` is passed,
        details for all model definitions will be returned.

        :param model_definition_uid: Unique Id of model definition
        :type model_definition_uid: str, optional

        :return: metadata of model definition
        :rtype: dict (if `model_definition_uid` is not None)

        **Example**

        .. code-block: python

            model_definition_details = client.model_definitions.get_details(model_definition_uid)

        """
        def get_required_element_from_response(response):
            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:
                final_response = {
                    "metadata": response[u'metadata'],
                }

                if 'entity' in response:
                    entity = response[u'entity']

                    try:
                        del entity[u'wml_model_definition'][u'ml_version']
                    except KeyError:
                        pass

                    final_response["entity"] = entity

                return final_response

        return self._get_asset_based_resource(model_definition_uid, 'wml_model_definition', get_required_element_from_response)

    def download(self, model_definition_uid, filename, rev_id=None):
        """Download the content of a model definition asset.

        :param model_definition_uid: the Unique Id of the model definition asset to be downloaded
        :type model_definition_uid: str
        :param filename: filename to be used for the downloaded file
        :type filename: str
        :param rev_id: revision id
        :type rev_id: str, optional

        :return: path to the downloaded asset content
        :rtype: str

        **Example**

        .. code-block:: python

            client.model_definitions.download(model_definition_uid, "model_definition_file")
        """

        self._client._check_if_either_is_set()

        ModelDefinition._validate_type(model_definition_uid, u'model_definition_uid', str, True)
        params = self._client._params()
        if rev_id is not None:
            ModelDefinition._validate_type(rev_id, u'rev_id', int, False)
            params.update({'revision_id': rev_id})

        if self._client.WSD:
            import urllib
            response = requests.get(self._client.service_instance._href_definitions.get_data_asset_href(model_definition_uid),
                                              params=self._client._params(),
                                              headers=self._client._get_headers())

            model_def_details = self._handle_response(200, u'get model', response)
            attachment_url = model_def_details['attachments'][0]['object_key']
            attachment_signed_url = self._client.service_instance._href_definitions.get_wsd_model_attachment_href() + \
                                   urllib.parse.quote('wml_model_definition/' + attachment_url, safe='')

        else:
            attachment_id = self._get_attachment_id(model_definition_uid)
            artifact_content_url = self._client.service_instance._href_definitions.get_attachment_href(model_definition_uid, attachment_id)
            if not self._ICP and not self._WSD:
                response = requests.get(self._client.service_instance._href_definitions.get_attachment_href(model_definition_uid, attachment_id), params=self._client._params(),
                                    headers=self._client._get_headers())
            else:
                response = requests.get(artifact_content_url, params=self._client._params(),
                                    headers=self._client._get_headers())
            attachment_signed_url = response.json()["url"]
        if response.status_code == 200:
            if not self._ICP and not self._client.WSD:
                if self._client.CLOUD_PLATFORM_SPACES:
                    att_response = requests.get(attachment_signed_url)
                else:
                    att_response = requests.get(self._wml_credentials["url"] + attachment_signed_url)
            else:
                if self._client.WSD:
                    att_response = requests.get(attachment_signed_url, params= self._client._params(),
                                                headers=self._client._get_headers(),
                                                stream=True)
                else:
                    att_response = requests.get(self._wml_credentials["url"]+attachment_signed_url)

            if att_response.status_code != 200:
                raise WMLClientError(u'Failure during {}.'.format("downloading model_definition asset"),
                                     att_response)

            downloaded_asset = att_response.content
            try:
                with open(filename, 'wb') as f:
                    f.write(downloaded_asset)
                print(u'Successfully saved asset content to file: \'{}\''.format(filename))
                return os.getcwd() + "/" + filename
            except IOError as e:
                raise WMLClientError(u'Saving asset with artifact_url: \'{}\' failed.'.format(filename), e)
        else:
            raise WMLClientError("Failed while downloading the asset " + model_definition_uid)

    def delete(self, model_definition_uid):
        """Delete a stored model definition.

        :param model_definition_uid: Unique Id of stored model definition
        :type model_definition_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.model_definitions.delete(model_definition_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        ModelDefinition._validate_type(model_definition_uid, u'model_definition_uid', str, True)
        paramvalue = self._client._params()

        model_definition_endpoint = self._client.service_instance._href_definitions.get_model_definition_assets_href() + "/" + model_definition_uid

        response_delete = requests.delete(model_definition_endpoint, params=paramvalue, headers=self._client._get_headers())

        return self._handle_response(204, u'Model definition deletion', response_delete, False)

    def _delete(self, model_definition_uid):
        """Delete a stored model definition.

        :param model_definition_uid: Unique Id of stored model definition
        :type model_definition_uid: str

        **Example**

        .. code-block:: python

            client.model_definitions._delete(model_definition_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        ModelDefinition._validate_type(model_definition_uid, u'model_definition_uid', str, True)
        paramvalue = self._client._params()
        model_definition_endpoint = self._client.service_instance._href_definitions.get_model_definition_assets_href() + "/" + model_definition_uid

        response_delete = requests.delete(model_definition_endpoint, params=paramvalue, headers=self._client._get_headers())

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data,  u'model_definition_response', dict)
        revision_id = None

        try:
            href = ""
            if self._client.default_space_id is not None:
                new_el = {'metadata': {'space_id': response_data['metadata']['space_id'],
                                   'guid': response_data['metadata']['asset_id'],
                                   'asset_type': response_data['metadata']['asset_type'],
                                   'created_at': response_data['metadata']['created_at'],
                                   'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                                },
                      'entity': response_data['entity']

                      }
                href = self._client.service_instance._href_definitions.get_base_asset_with_type_href(response_data['metadata']['asset_type'], response_data['metadata'][
                    'asset_id']) + "?" + "space_id=" + response_data['metadata']['space_id']

            elif self._client.default_project_id is not None:
                new_el = {'metadata': {'project_id': response_data['metadata']['project_id'],
                                       'guid': response_data['metadata']['asset_id'],
                                       'asset_type': response_data['metadata']['asset_type'],
                                       'created_at': response_data['metadata']['created_at'],
                                       'last_updated_at': response_data['metadata']['usage']['last_updated_at']
                                       },
                          'entity': response_data['entity']

                          }

                href = self._client.service_instance._href_definitions.get_base_asset_with_type_href(response_data['metadata']['asset_type'], response_data['metadata'][
                    'asset_id']) + "?" + "project_id=" + response_data['metadata']['project_id']

            if 'revision_id' in response_data['metadata']:
                new_el['metadata'].update({'revision_id': response_data['metadata']['revision_id']})
                revision_id = response_data[u'metadata'][u'revision_id']

            if 'name' in response_data['metadata']:
                new_el['metadata'].update({'name': response_data['metadata']['name']})

            if 'description' in response_data['metadata'] and response_data['metadata']['description']:
                new_el['metadata'].update({'description': response_data['metadata']['description']})

            if 'href' in response_data['metadata']:
                if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                    href_without_host = response_data['href'].split('.com')[-1]
                    new_el[u'metadata'].update({'href': href_without_host})
                else:
                    new_el['metadata'].update({'href': response_data['href']})
            else:
                new_el['metadata'].update({'href': href})

            if "attachments" in response_data and response_data[u'attachments']:
                new_el['metadata'].update({'attachment_id': response_data[u'attachments'][0][u'id']})
            else:
                new_el['metadata'].update({'href': href})

            if "commit_info" in response_data[u'metadata'] and revision_id is not None:
                new_el['metadata'].update(
                    {'revision_commit_date': response_data[u'metadata'][u'commit_info']['committed_at']})
            return new_el
        except Exception as e:
            raise WMLClientError("Failed to read Response from down-stream service: " + response_data)

    def _get_attachment_id(self, model_definition_uid, rev_id=None):
        op_name = 'getting attachment id '
        url = self._client.service_instance._href_definitions.get_model_definition_assets_href() + u'/' + model_definition_uid
        paramvalue = self._client._params()

        if rev_id is not None:
            paramvalue.update({'revision_id': rev_id})

        response_get = requests.get(
            url,
            params=paramvalue,
            headers=self._client._get_headers()
        )
        details = self._handle_response(200, op_name, response_get)
        try:
            attachment_id = details["attachments"][0]["id"]
        except KeyError:
            raise WMLClientError(f'No attachment exists for model definition (id={model_definition_uid}).')

        return attachment_id

    def list(self, limit=None, return_as_df=True):
        """Print stored model definition assets in a table format.
        If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed model definitions or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.model_definitions.list()

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        href = self._client.service_instance._href_definitions.get_model_definition_search_asset_href()
        if limit is None:
            data = {
                "query": "*:*"
            }
        else:
            ModelDefinition._validate_type(limit, u'limit', int, False)
            data = {
                "query": "*:*",
                "limit": limit
            }

        response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(), json=data)

        self._handle_response(200, u'model_definition assets', response)
        asset_details = self._handle_response(200, u'model_definition assets', response)["results"]
        model_def_values = [
            (m[u'metadata'][u'name'], m[u'metadata'][u'asset_type'], m[u'metadata'][u'asset_id']) for
            m in asset_details]

        table = self._list(model_def_values, [u'NAME', u'ASSET_TYPE', u'GUID'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table


    def get_id(self, model_definition_details):
        """Get Unique Id of stored model definition asset.

        :param model_definition_details: metadata of the stored model definition asset
        :type model_definition_details: dict

        :return: Unique Id of stored model definition asset
        :rtype: str

        **Example**

        .. code-block:: python

            asset_uid = client.model_definition.get_id(asset_details)

        """

        return ModelDefinition.get_uid(self, model_definition_details)

    def get_uid(self, model_definition_details):
        """Get uid of stored model.

        *Deprecated:* Use ``get_id(model_definition_details)`` instead.

        :param model_definition_details: stored model definition details
        :type model_definition_details: dict

        :return: uid of stored model definition
        :rtype: str

        **Example**

        .. code-block:: python

            model_definition_uid = client.model_definitions.get_uid(model_definition_details)
        """
        if 'asset_id' in model_definition_details['metadata']:
            return WMLResource._get_required_element_from_dict(model_definition_details, u'model_definition_details',
                                                               [u'metadata', u'asset_id'])
        else:
            ModelDefinition._validate_type(model_definition_details, u'model__definition_details', object, True)
            #ModelDefinition._validate_type_of_details(model_definition_details, MODEL_DEFINITION_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(model_definition_details, u'model_definition_details', [u'metadata', u'guid'])

    def get_href(self, model_definition_details):
        """Get href of stored model definition.

        :param model_definition_details: stored model definition details
        :type model_definition_details: dict

        :return: href of stored model definition
        :rtype: str

        **Example**

        .. code-block:: python

            model_definition_uid = client.model_definitions.get_href(model_definition_details)
        """
        if 'asset_id' in model_definition_details['metadata']:
            return WMLResource._get_required_element_from_dict(model_definition_details, u'model_definition_details', [u'metadata', u'asset_id'])
        else:
            ModelDefinition._validate_type(model_definition_details, u'model__definition_details', object, True)
            # ModelDefinition._validate_type_of_details(model_definition_details, MODEL_DEFINITION_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(model_definition_details, u'model_definition_details',
                                                               [u'metadata', u'href'])

    def update(self, model_definition_id, meta_props=None, file_path=None):
        """Update model definition with either metadata or attachment or both.

        :param model_definition_id: model definition ID
        :type model_definition_id: str
        :param meta_props: meta data of the model definition configuration to be updated
        :type meta_props: dict
        :param file_path: path to the content file to be uploaded
        :type file_path: str, optional

        :return: updated metadata of model definition
        :rtype: dict

        **Example**

        .. code-block:: python

            model_definition_details = client.model_definition.update(model_definition_id, meta_props, file_path)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        # We need to enable this once we add functionality for WSD
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(u'Not supported in this release')

        ModelDefinition._validate_type(model_definition_id, 'model_definition_id', str, True)

        if meta_props is None and file_path is None:
            raise WMLClientError('At least either meta_props or file_path has to be provided')

        updated_details = None

        url = self._client.service_instance._href_definitions.get_asset_href(model_definition_id)

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   CAMS has meta and entity patching. 'name' and 'description' get stored in CAMS meta section
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        #   b. Construct entity patch if required and call /v2/assets/<asset_id>/attributes/script to patch entity
        # STEP 3. If file_path provided, we need to patch the attachment
        #   a. If attachment already exists for the model_definition, delete it
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
                    u'Invalid input. Unable to get the details of model_definition_id provided.')
            else:
                raise WMLClientError(u'Failure during {}.'.format("getting script to update"), response)

        details = self._handle_response(200, "Get script details", response)

        attachments_response = None

        # STEP 2a.
        # Patch meta if provided
        if meta_props is not None:
            self._validate_type(meta_props, u'meta_props', dict, True)

            props_for_asset_meta_patch = {}

            # Since we are dealing with direct asset apis, there can be metadata or entity patch or both
            if "name" in meta_props or "description" in meta_props:

                for key in meta_props:
                    if key == 'name' or key == 'description':
                        props_for_asset_meta_patch.update({key: meta_props[key]})

            meta_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details,
                                                                                     props_for_asset_meta_patch,
                                                                                     with_validation=True)

            props_for_asset_entity_patch = {}
            for key in meta_props:
                if key != 'name' and key != 'description':
                    props_for_asset_entity_patch.update({key: meta_props[key]})

            entity_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity']['wml_model_definition'],
                                                                                       props_for_asset_entity_patch,
                                                                                       with_validation=True)

            if meta_patch_payload:
                meta_patch_url = self._client.service_instance._href_definitions.get_asset_href(model_definition_id)

                response_patch = requests.patch(meta_patch_url,
                                                json=meta_patch_payload,
                                                params=self._client._params(),
                                                headers=self._client._get_headers())

                updated_details = self._handle_response(200, u'script patch', response_patch)

            if entity_patch_payload:
                entity_patch_url = self._client.service_instance._href_definitions.get_asset_href(model_definition_id) + \
                                   '/attributes/wml_model_definition'

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
            attachments_response = self._update_attachment_for_assets("wml_model_definition",
                                                                      model_definition_id,
                                                                      file_path,
                                                                      current_attachment_id)

        if attachments_response is not None and 'success' not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._client.service_instance._href_definitions.get_asset_href(model_definition_id)

        response = requests.get(
            url,
            params=self._client._params(),
            headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    u'Invalid input. Unable to get the details of model_definition_id provided.')
            else:
                raise WMLClientError(u'Failure during {}.'.format("getting script to update"), response)

        response = self._get_required_element_from_response(self._handle_response(200, "Get script details", response))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            entity = response[u'entity']

            try:
                del entity[u'wml_model_definition'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

        # return self._get_required_element_from_response(response)

    def _update_msg(self, updated_details):
        if updated_details is not None:
            print("Could not update the attachment because of server error."
                  " However metadata is updated. Try updating attachment again later")
        else:
            raise WMLClientError('Unable to update attachment because of server error. Try again later')

    def create_revision(self, model_definition_uid):
        """Create revision for the given model definition. Revisions are immutable once created.
        The metadata and attachment at model definition is taken and a revision is created out of it.

        :param model_definition_uid: model definition ID
        :type model_definition_uid: str

        :return: stored model definition revisions metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            model_definition_revision = client.model_definitions.create_revision(model_definition_id)
        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revisions APIs are not supported in this release.')

        self._client._check_if_either_is_set()

        ModelDefinition._validate_type(model_definition_uid, u'model_definition_uid', str, True)

        print("Creating model_definition revision...")

        # return self._get_required_element_from_response(
        #     self._create_revision_artifact_for_assets(model_defn_id, 'Model definition'))

        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(model_definition_uid, 'Model definition'))

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            return response
        else:
            entity = response[u'entity']

            try:
                del entity[u'wml_model_definition'][u'ml_version']
            except KeyError:
                pass

            final_response = {
                "metadata": response[u'metadata'],
                "entity": entity
            }

            return final_response

    def get_revision_details(self, model_definition_uid, rev_uid=None):
        """Get metadata of model definition.

        :param model_definition_uid: model definition ID
        :type model_definition_uid: str

        :param rev_uid: revision ID, if this parameter is not provided, returns latest revision if existing else error
        :type rev_uid: int, optional

        :return: stored model definitions metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            script_details = client.model_definitions.get_revision_details(model_definition_uid, rev_uid)
        """

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revisions APIs are not supported in this release.')
        op_name = 'getting model_definition revision details'
        ModelDefinition._validate_type(model_definition_uid, u'model_definition_uid', str, True)

        url = self._client.service_instance._href_definitions.get_model_definition_assets_href() + u'/' + model_definition_uid
        paramvalue = self._client._params()

        if rev_uid is None:
            rev_uid = 'latest'

        paramvalue.update({'revision_id': rev_uid})

        response_get = requests.get(
            url,
            params=paramvalue,
            headers=self._client._get_headers()
        )
        if response_get.status_code == 200:
            # get_model_definition_details = self._handle_response(200, op_name, response_get)
            # return self._get_required_element_from_response(get_model_definition_details)

            response = self._get_required_element_from_response(self._handle_response(200, op_name, response_get))

            if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
                return response
            else:
                entity = response[u'entity']

                try:
                    del entity[u'wml_model_definition'][u'ml_version']
                except KeyError:
                    pass

                final_response = {
                    "metadata": response[u'metadata'],
                    "entity": entity
                }

                return final_response
        else:
            return self._handle_response(200, op_name, response_get)

    def list_revisions(self, model_definition_uid, limit=None):
        """Print stored model definition assets in a table format.
        If limit is set to None there will be only first 50 records shown.

        :param model_definition_uid: Unique id of model definition
        :type model_definition_uid: str
        :param limit: limit number of fetched records
        :type limit: int, optional

        **Example**

        .. code-block:: python

            client.model_definitions.list_revisions()

        """
        ##For CP4D, check if either spce or project ID is set
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                u'Revisions APIs are not supported in this release.')
        self._client._check_if_either_is_set()
        href = self._client.service_instance._href_definitions.get_model_definition_assets_href() + "/" + model_definition_uid +\
               u'/revisions'
        params = self._client._params()
        #params = None
        if limit is not None:
            ModelDefinition._validate_type(limit, u'limit', int, False)
            params.update( {
                "limit": limit
            })
        response = requests.get(href, params=params, headers=self._client._get_headers())
        self._handle_response(200, u'model_definition revision assets', response)
        asset_details = self._handle_response(200, u'model_definition revision assets', response)["results"]
        model_def_values = [
            (m[u'metadata'][u'asset_id'],
             m[u'metadata'][u'revision_id'],
             m[u'metadata'][u'name'],
             m[u'metadata'][u'asset_type'],
             m[u'metadata'][u'commit_info'][u'committed_at']) for
            m in asset_details]

        self._list(model_def_values, [u'GUID', u'REV_ID', u'NAME', u'ASSET_TYPE', u'REVISION_COMMIT'],
                   limit,
                   _DEFAULT_LIST_LENGTH)



