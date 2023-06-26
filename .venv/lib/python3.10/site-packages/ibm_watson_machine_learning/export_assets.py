#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.metanames import ExportMetaNames
import os
import json
from ibm_watson_machine_learning.wml_client_error import WMLClientError, UnexpectedType, ApiRequestFailure
from ibm_watson_machine_learning.wml_resource import WMLResource
_DEFAULT_LIST_LENGTH = 50

class Export(WMLResource):
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

        self._client = client
        self.ConfigurationMetaNames = ExportMetaNames()

    def start(self, meta_props, space_id=None, project_id=None):
        """Start the export. Either space_id or project_id has to be provided and is mandatory.
        ALL_ASSETS is by default False. No need to provide explicitly unless it has to be set to True.
        Either ALL_ASSETS or ASSET_TYPES or ASSET_IDS has to be given in the meta_props. Only one of these can be
        provided.

        In the `meta_props`:

        ALL_ASSETS is a boolean. When set to True, it exports all assets in the given space.
        ASSET_IDS is an array containing the list of assets ids to be exported.
        ASSET_TYPES is for providing the asset types to be exported. All assets of that asset type will be exported.

                Eg: wml_model, wml_model_definition, wml_pipeline, wml_function, wml_experiment,
                software_specification, hardware_specification, package_extension, script

        :param meta_props: meta data,
            to see available meta names use ``client.export_assets.ConfigurationMetaNames.get()``
        :type meta_props: dict
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project: str, optional

        :return: Response json
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.export_assets.ConfigurationMetaNames.NAME: "export_model",
                client.export_assets.ConfigurationMetaNames.ASSET_IDS: ["13a53931-a8c0-4c2f-8319-c793155e7517",
                                                                        "13a53931-a8c0-4c2f-8319-c793155e7518"]}

            details = client.export_assets.start(meta_props=metadata, space_id="98a53931-a8c0-4c2f-8319-c793155e4598")

        .. code-block:: python

            metadata = {
                client.export_assets.ConfigurationMetaNames.NAME: "export_model",
                client.export_assets.ConfigurationMetaNames.ASSET_TYPES: ["wml_model"]}

            details = client.export_assets.start(meta_props=metadata, space_id="98a53931-a8c0-4c2f-8319-c793155e4598")

        .. code-block:: python

            metadata = {
                client.export_assets.ConfigurationMetaNames.NAME: "export_model",
                client.export_assets.ConfigurationMetaNames.ALL_ASSETS: True}

            details = client.export_assets.start(meta_props=metadata, space_id="98a53931-a8c0-4c2f-8319-c793155e4598")

            """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Either 'space_id' or 'project_id' has to be provided")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        Export._validate_type(meta_props, u'meta_props', dict, True)
        self._validate_input_meta(meta_props)

        meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        start_meta = {}
        assets = {}

        start_meta[u'name'] = meta[u'name']
        if 'description' in meta:
            start_meta[u'description'] = meta[u'description']

        if "all_assets" not in meta:
            assets.update({"all_assets": False})
        else:
            assets.update({"all_assets": meta[u'all_assets']})

        if "asset_types" in meta:
            assets.update({"asset_types": meta[u'asset_types']})

        if "asset_ids" in meta:
            assets.update(({"asset_ids": meta[u'asset_ids']}))

        start_meta[u'assets'] = assets

        href = self._client.service_instance._href_definitions.exports_href()

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        creation_response = requests.post(href,
                                          params=params,
                                          headers=self._client._get_headers(),
                                          json=start_meta)

        details = self._handle_response(expected_status_code=202,
                                        operationName=u'export start',
                                        response=creation_response)

        export_id = details[u'metadata']['id']

        print("export job with id {} has started. Monitor status using client.export_assets.get_details api. "
              "Check 'help(client.export_assets.get_details)' for details on the api usage".format(export_id))

        return details

    def _validate_input_meta(self, meta_props):
        if 'name' not in meta_props:
            raise WMLClientError("Its mandatory to provide 'NAME' in meta_props. Example: "
                                 "client.export_assets.ConfigurationMetaNames.NAME: 'name'")

        if 'all_assets' not in meta_props and 'asset_ids' not in meta_props and 'asset_types' not in meta_props:
            raise WMLClientError("Its mandatory to provide either 'ALL_ASSETS' or 'ASSET_IDS' or 'ASSET_TYPES' " 
                                 "in meta_props. Example: client.export_assets.ConfigurationMetaNames.ALL_ASSETS: True")

        count = 0

        if 'all_assets' in meta_props:
            count = count + 1
        if 'asset_ids' in meta_props:
            count = count + 1
        if 'asset_types' in meta_props:
            count = count + 1

        if count > 1:
            raise WMLClientError("Only one of 'ALL_ASSETS' or 'ASSET_IDS' or 'ASSET_TYPES' can be provided")

    def cancel(self, export_id, space_id=None, project_id=None):
        """Cancel an export job. `space_id` or `project_id` has to be provided.

        .. note::
            To delete a `export_id` job, use ``delete()`` api.

        :param export_id: export job identifier
        :type export_id: str
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional

        :returns: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.export_assets.cancel(export_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                                        space_id='3421cf1-252f-424b-b52d-5cdd981495fe')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        Export._validate_type(export_id, u'export_id', str, True)

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        href = self._client.service_instance._href_definitions.export_href(export_id)

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        cancel_response = requests.delete(href,
                                          params=params,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'cancel export',
                                        response=cancel_response)

        if "SUCCESS" == details:
            print("Export job cancelled")

    def delete(self, export_id, space_id=None, project_id=None):
        """Deletes the given `export_id` job. `space_id` or `project_id` has to be provided.

        :param export_id: export job identifier
        :type export_id: str
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.export_assets.delete(export_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                                        space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        Export._validate_type(export_id, u'export_id', str, True)

        href = self._client.service_instance._href_definitions.export_href(export_id)

        params = {"hard_delete": True}

        if space_id is not None:
            params.update({'space_id': space_id})
        else:
            params.update({'project_id': project_id})

        delete_response = requests.delete(href,
                                          params=params,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'delete export job',
                                        response=delete_response)

        if "SUCCESS" == details:
            print("Export job deleted")

    def get_details(self, export_id=None, space_id=None, project_id=None, limit=None, asynchronous=False,
                    get_all=False):
        """Get metadata of the given export job. if no `export_id` is specified all exports metadata is returned.

        :param export_id: export job identifier
        :type export_id: str, optional
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: export(s) metadata
        :rtype: dict (if export_id is not None) or {"resources": [dict]} (if export_id is None)

        **Example**

        .. code-block:: python

            details = client.export_assets.get_details(export_id, space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')
            details = client.export_assets.get_details()
            details = client.export_assets.get_details(limit=100)
            details = client.export_assets.get_details(limit=100, get_all=True)
            details = []
            for entry in client.export_assets.get_details(limit=100, asynchronous=True, get_all=True):
                details.extend(entry)
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")
        
        Export._validate_type(export_id, u'export_id', str, False)
        Export._validate_type(limit, u'limit', int, False)

        href = self._client.service_instance._href_definitions.exports_href()

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        if export_id is None:
            return self._get_artifact_details(href, export_id, limit, 'export job', query_params=params,
                                              _async=asynchronous, _all=get_all)

        else:
            return self._get_artifact_details(href, export_id, limit, 'export job', query_params=params)
    
    def list(self, space_id=None, project_id=None, limit=None, return_as_df=True):
        """Print export jobs in a table format. If limit is set to None there will be only first 50 records shown.

        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed connections or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.export_assets.list()
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        if space_id is not None:
            resources = self.get_details(space_id=space_id)[u'resources']
        else:
            resources = self.get_details(project_id=project_id)[u'resources']


        values = [(m[u'metadata'][u'id'],
                   m[u'metadata'][u'name'],
                   m[u'metadata'][u'created_at'],
                   m[u'entity'][u'status'][u'state']) for m in resources]

        table = self._list(values, [u'ID', u'NAME', u'CREATED', u'STATUS'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    @staticmethod
    def get_id(export_details):
        """Get ID of export job from export details.

        :param export_details: metadata of the export job
        :type export_details: dict

        :return: ID of the export job
        :rtype: str

        **Example**

        .. code-block:: python

            id = client.export_assets.get_id(export_details)
        """
        Export._validate_type(export_details, u'export_details', object, True)

        return WMLResource._get_required_element_from_dict(export_details,
                                                           u'export_details',
                                                           [u'metadata', u'id'])

    def get_exported_content(self, export_id, space_id=None, project_id=None, file_path=None):
        """Get the exported content as a zip file.

        :param export_id: export job identifier
        :type export_id: str
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional
        :param file_path: name of local file to create, this should be absolute path of the file
            and the file shouldn't exist
        :type file_path: str, optional


        :return: path to the downloaded function content
        :rtype: str

        **Example**

        .. code-block:: python

            client.exports.get_exported_content(export_id,
                                                space_id='98a53931-a8c0-4c2f-8319-c793155e4598',
                                                file_path='/home/user/my_exported_content.zip')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        if os.path.isfile(file_path):
            raise WMLClientError(u'File with name: \'{}\' already exists.'.format(file_path))

        Export._validate_type(file_path, u'file_path', str, True)

        href = self._client.service_instance._href_definitions.export_content_href(export_id)

        try:
            response = requests.get(href,
                                    params=params,
                                    headers=self._client._get_headers(),
                                    stream=True)

            if response.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading export content"), response)

            downloaded_exported_content = response.content
            self._logger.info(u'Successfully downloaded artifact with artifact_url: {}'.format(href))
        except WMLClientError as e:
            raise e
        except Exception as e:
            raise WMLClientError(u'Downloading export content with artifact_url: \'{}\' failed.'.format(href), e)

        try:
            with open(file_path, 'wb') as f:
                f.write(downloaded_exported_content)
            print(u'Successfully saved export content to file: \'{}\''.format(file_path))
            return file_path
        except IOError as e:
            raise WMLClientError(u'Downloading export content with artifact_url: \'{}\' failed.'.format(href), e)

