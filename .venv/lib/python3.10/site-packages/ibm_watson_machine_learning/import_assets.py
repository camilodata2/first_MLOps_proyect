#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
import os
import json
from ibm_watson_machine_learning.wml_client_error import WMLClientError, UnexpectedType, ApiRequestFailure
from ibm_watson_machine_learning.wml_resource import WMLResource
_DEFAULT_LIST_LENGTH = 50

class Import(WMLResource):
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

        self._client = client

    def start(self, file_path=None, space_id=None, project_id=None):
        """Start the import. Either `space_id` or `project_id` has to be provided. Note that
        on IBM Cloud Pak® for Data 3.5, import into non-empty space/project is not supported.

        :param file_path: file path to zip file with exported assets
        :type file_path: dict
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional

        :return: response json
        :rtype: dict

        **Example**

        .. code-block:: python

            details = client.import_assets.start(space_id="98a53931-a8c0-4c2f-8319-c793155e4598",
                                                 file_path="/home/user/data_to_be_imported.zip")
        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Either 'space_id' or 'project_id' has to be provided")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        if file_path is None:
            raise WMLClientError("Its mandatory to provide 'file_path'")

        if not os.path.isfile(file_path):
            raise WMLClientError(u'File with name: \'{}\' does not exist'.format(file_path))

        # with open(file_path, "rb") as a_file:
        #     file_dict = {file_path: a_file}

        # file_handler = {'file.zip': open(file_path,'rb')}

        with open(file_path, 'rb') as file:

            # from datetime import datetime
            # start_time = datetime.now()
            #
            data = file.read()
            #
            # end_time = datetime.now()
            #
            # dt = end_time - start_time
            # ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0

            # print("Time taken to read the zip file in ms: {}".format(ms))

        href = self._client.service_instance._href_definitions.imports_href()

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        creation_response = requests.post(href,
                                          params=params,
                                          headers=self._client._get_headers(content_type='application/zip'),
                                          data=data)

        details = self._handle_response(expected_status_code=202,
                                        operationName=u'import start',
                                        response=creation_response)

        import_id = details[u'metadata']['id']

        print("import job with id {} has started. Monitor status using client.import_assets.get_details api. "
              "Check 'help(client.import_assets.get_details)' for details on the api usage".format(import_id))

        return details

    def _validate_input(self, meta_props):
        if 'name' not in meta_props:
            raise WMLClientError("Its mandatory to provide 'NAME' in meta_props. Example: "
                                 "client.import_assets.ConfigurationMetaNames.NAME: 'name'")

        if 'all_assets' not in meta_props and 'asset_ids' not in meta_props:
            raise WMLClientError("Its mandatory to provide either 'ALL_ASSETS' or 'ASSET_IDS' in meta_props. Example: "
                                 "client.import_assets.ConfigurationMetaNames.ALL_ASSETS: True")

    def cancel(self, import_id, space_id=None, project_id=None):
        """Cancel an import job. Either `space_id` or `project_id` has to be provided.

        .. note::

            To delete an import_id job, use delete() api

        :param import_id: import job identifier
        :type import_id: str
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.import_assets.cancel(import_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                                        space_id='3421cf1-252f-424b-b52d-5cdd981495fe')

        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        Import._validate_type(import_id, u'import_id', str, True)

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        href = self._client.service_instance._href_definitions.import_href(import_id)

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        cancel_response = requests.delete(href,
                                          params=params,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'cancel import',
                                        response=cancel_response)

        if "SUCCESS" == details:
            print("Import job cancelled")

    def delete(self, import_id, space_id=None, project_id=None):
        """Deletes the given `import_id` job. `space_id` or `project_id` has to be provided.

        :param import_id: import job identifier
        :type import_id: str
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.import_assets.delete(import_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                                        space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")

        Import._validate_type(import_id, u'import_id', str, True)

        href = self._client.service_instance._href_definitions.import_href(import_id)

        params = {"hard_delete": True}

        if space_id is not None:
            params.update({'space_id': space_id})
        else:
            params.update({'project_id': project_id})

        delete_response = requests.delete(href,
                                          params=params,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'delete import job',
                                        response=delete_response)

        if "SUCCESS" == details:
            print("Import job deleted")

    def get_details(self, import_id=None, space_id=None, project_id=None, limit=None, asynchronous=False, get_all=False):
        """Get metadata of the given import job. if no `import_id` is specified, all imports metadata is returned.

        :param import_id: import job identifier
        :type import_id: str, optional
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

        :return: import(s) metadata
        :rtype: dict (if import_id is not None) or {"resources": [dict]} (if import_id is None)

        **Example**

        .. code-block:: python

            details = client.import_assets.get_details(import_id)
            details = client.import_assets.get_details()
            details = client.import_assets.get_details(limit=100)
            details = client.import_assets.get_details(limit=100, get_all=True)
            details = []
            for entry in client.import_assets.get_details(limit=100, asynchronous=True, get_all=True):
                details.extend(entry)
        """
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported in this release")

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None and project_id is not None:
            raise WMLClientError("Either 'space_id' or 'project_id' can be provided, not both")
        
        Import._validate_type(import_id, u'import_id', str, False)
        Import._validate_type(limit, u'limit', int, False)

        href = self._client.service_instance._href_definitions.imports_href()

        params = {}

        if space_id is not None:
            params.update({"space_id": space_id})
        else:
            params.update({"project_id": project_id})

        if import_id is None:
            return self._get_artifact_details(href, import_id, limit, 'import job', query_params=params,
                                              _async=asynchronous, _all=get_all)

        else:
            return self._get_artifact_details(href, import_id, limit, 'import job', query_params=params)
    
    def list(self, space_id=None, project_id=None, limit=None, return_as_df=True):
        """Print import jobs in a table format. If limit is set to None there will be only first 50 records shown.

        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed assets or None if return_as_df is False
        :rtype: pandas.DataFrame or None


        **Example**

        .. code-block:: python

            client.import_assets.list()
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
                   m[u'metadata'][u'created_at'],
                   m[u'entity'][u'status'][u'state']) for m in resources]

        table = self._list(values, [u'ID',  u'CREATED', u'STATUS'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    @staticmethod
    def get_id(import_details):
        """Get ID of import job from import details.

        :param import_details: metadata of the import job
        :type import_details: dict

        :return: ID of the import job
        :rtype: str

        **Example**

        .. code-block:: python

            id = client.import_assets.get_id(import_details)
        """
        Import._validate_type(import_details, u'import_details', object, True)

        return WMLResource._get_required_element_from_dict(import_details,
                                                           u'import_details',
                                                           [u'metadata', u'id'])

