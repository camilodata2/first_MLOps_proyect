#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests

from ibm_watson_machine_learning.messages.messages import Messages
from ibm_watson_machine_learning.metanames import ConnectionMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError
import os
import json
from urllib.parse import unquote, quote

_DEFAULT_LIST_LENGTH = 0


class Connections(WMLResource):
    """Store and manage Connections."""

    ConfigurationMetaNames = ConnectionMetaNames()
    """MetaNames for Connection creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data, u'connection_response', dict)
        try:
            new_el = {'metadata': {'id': response_data['metadata']['asset_id'],
                                   'asset_type': response_data['metadata']['asset_type'],
                                   'create_time': response_data['metadata']['create_time']
                                   if 'create_time' in response_data['metadata']
                                   else response_data['metadata']['created_at'],
                                   'last_access_time': response_data['metadata']['usage'].get('last_access_time')
                                   },
                      'entity': {
                          'datasource_type': response_data['entity']['datasource_type']
                          if 'datasource_type' in response_data['entity']
                          else response_data['entity']['connection']['datasource_type'],
                          'name': response_data['entity']['name']
                          if 'name' in response_data['entity']
                          else response_data['metadata']['name']
                      }
            }

            for el in ['description', 'origin_country', 'owner_id', 'properties']:
                if el in response_data['entity']:
                    new_el['entity'][el] = response_data['entity'].get(el)

            if self._client.default_space_id is not None:
                new_el['metadata']['space_id'] = response_data['metadata']['space_id']

            elif self._client.default_project_id is not None:
                new_el['metadata']['project_id'] = response_data['metadata']['project_id']

                if not self._client.WSD:
                    if 'href' in response_data['metadata']:
                        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
                            href_without_host = response_data['href'].split('.com')[-1]
                            new_el[u'metadata'].update({'href': href_without_host})
                        else:
                            new_el['metadata'].update({'href': response_data['href']})

            return new_el
        except Exception as e:
            raise e
            #raise WMLClientError(Messages.get_message(response_data, message_id="failed_to_read_response_from_down_stream_service"))

    def get_details(self, connection_id=None):
        """Get connection details for the given unique Connection id.
        If no connection_id is passed, details for all connections will be returned.

        :param connection_id: Unique id of Connection
        :type connection_id: str

        :return: metadata of the stored Connection
        :rtype: dict

        **Example**

        .. code-block:: python

            connection_details = client.connections.get_details(connection_id)
            connection_details = client.connections.get_details()

        """
        self._client._check_if_either_is_set()
        Connections._validate_type(connection_id, u'connection_id', str, False)

        if self._client.WSD:
            header_param = self._client._get_headers(wsdconnection_api_req=True)
        else:
            header_param = self._client._get_headers()

        if self._client._iam_id:
            header_param['IBM-WDP-Impersonate'] = str({'iam_id': str(self._client._iam_id)})

        if connection_id:
            with self.requests_retry_session() as sess:
                response = sess.get(
                    self._client.service_instance._href_definitions.get_connection_by_id_href(connection_id),
                    params=self._client._params(),
                    headers=header_param)

                return self._get_required_element_from_response(
                    self._handle_response(200, u'get connection details', response))
        else:
            with self.requests_retry_session() as sess:
                response = sess.post(
                    self._client.service_instance._href_definitions.get_asset_search_href('connection'),
                    json={"query": "*:*",
                          "include": "entity"
                          },
                    params=self._client._params(),
                    headers=header_param)

                return {'resources': [self._get_required_element_from_response(r) for r
                                      in self._handle_response(200, u'get connection details', response)['results']]}

    def create(self, meta_props):
        """Create a connection. Input to PROPERTIES field examples:

        1. MySQL

            .. code-block:: python

                client.connections.ConfigurationMetaNames.PROPERTIES: {
                    "database": "database",
                    "password": "password",
                    "port": "3306",
                    "host": "host url",
                    "ssl": "false",
                    "username": "username"
                }

        2. Google Big query

            a. Method1: Use service account json. The service account json generated can be provided as
                input as-is. Provide actual values in json. Example is only indicative to show
                the fields. Refer to Google big query documents how to generate the service account json.

                .. code-block:: python

                    client.connections.ConfigurationMetaNames.PROPERTIES: {
                        "type": "service_account",
                        "project_id": "project_id",
                        "private_key_id": "private_key_id",
                        "private_key": "private key contents",
                        "client_email": "client_email",
                        "client_id": "client_id",
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": "client_x509_cert_url"
                    }

            b. Method2: Using OAuth Method. Refer to Google big query documents how to generate OAuth token.

                .. code-block:: python

                    client.connections.ConfigurationMetaNames.PROPERTIES: {
                        "access_token": "access token generated for big query",
                        "refresh_token": "refresh token",
                        "project_id": "project_id",
                        "client_secret": "This is your gmail account password",
                        "client_id": "client_id"
                    }

        3. MS SQL

            .. code-block:: python

                client.connections.ConfigurationMetaNames.PROPERTIES: {
                    "database": "database",
                    "password": "password",
                    "port": "1433",
                    "host": "host",
                    "username": "username"
                }

        4. Tera data

            .. code-block:: python

                client.connections.ConfigurationMetaNames.PROPERTIES: {
                    "database": "database",
                    "password": "password",
                    "port": "1433",
                    "host": "host",
                    "username": "username"
                }

        :param meta_props:  metadata of the connection configuration. To see available meta names use:

            .. code-block:: python

                client.connections.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored connection
        :rtype: dict

        **Example**

        .. code-block:: python

            sqlserver_data_source_type_id = client.connections.get_datasource_type_uid_by_name('sqlserver')
            connections_details = client.connections.create({
                client.connections.ConfigurationMetaNames.NAME: "sqlserver connection",
                client.connections.ConfigurationMetaNames.DESCRIPTION: "connection description",
                client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: sqlserver_data_source_type_id,
                client.connections.ConfigurationMetaNames.PROPERTIES: { "database": "database",
                                                                        "password": "password",
                                                                        "port": "1433",
                                                                        "host": "host",
                                                                        "username": "username"}
            })

        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        connection_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client
        )

        big_query_data_source_type_id = self.get_datasource_type_uid_by_name('bigquery')

        # Either service acct json credentials can be given or oauth json can be given
        # If service acct json, then we need to create a newline json with "credentials" key
        if connection_meta[u'datasource_type'] == big_query_data_source_type_id:
            if 'private_key' in connection_meta[u'properties']:
                result = json.dumps(connection_meta[u'properties'], separators=(',\n', ':'))
                newmap = {"credentials": result}
                connection_meta[u'properties'] = newmap

        connection_meta.update({'origin_country': 'US'})
        #Step1  : Create an asset
        print(Messages.get_message(message_id="creating_connections"))

        if self._client.WSD:
            header_param = self._client._get_headers(wsdconnection_api_req=True)
        else:
            header_param = self._client._get_headers()

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_connections_href(),
            headers=header_param,
            json=connection_meta,
            params=self._client._params()
        )
        connection_details = self._handle_response(201, u'creating new connection', creation_response)
        if creation_response.status_code == 201:
            connection_id = connection_details["metadata"]["asset_id"]
            print(Messages.get_message(message_id="success"))
            return self._get_required_element_from_response(connection_details)
        else:
            raise WMLClientError(Messages.get_message(message_id="failed_while_creating_connections"))

    def delete(self, connection_id):
        """Delete a stored Connection.

        :param connection_id: Unique id of the connection to be deleted.
        :type connection_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.connections.delete(connection_id)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Connections._validate_type(connection_id, u'connection_id', str, True)
        if self._client.WSD:
            header_param = self._client._get_headers(wsdconnection_api_req=True)
        else:
            header_param = self._client._get_headers()

        connection_endpoint = self._client.service_instance._href_definitions.get_connection_by_id_href(connection_id)
        response_delete = requests.delete(connection_endpoint, params=self._client._params(),
                                          headers=header_param)

        return self._handle_response(204, u'connection deletion', response_delete, False)

    @staticmethod
    def get_uid(connection_details):
        """Get Unique Id of stored connection.

        :param connection_details: metadata of the stored connection
        :type connection_details: dict

        :return: Unique Id of stored connection
        :rtype: str

        **Example**

        .. code-block:: python

            connection_uid = client.connection.get_uid(connection_details)

        """
        Connections._validate_type(connection_details, u'connection_details', object, True)

        return WMLResource._get_required_element_from_dict(connection_details, u'connection_details',
                                                           [u'metadata', u'id'])

    def list_datasource_types(self, return_as_df=True):
        """Print stored datasource types assets in a table format.
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed datasource types or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.connections.list_datasource_types()

        """

        if self._client.WSD:
            header_param = self._client._get_headers(wsdconnection_api_req=True)
        else:
            header_param = self._client._get_headers()

        response = requests.get(self._client.service_instance._href_definitions.get_connection_data_types_href(),
                                headers=header_param)

        datasource_details = self._handle_response(200, u'list datasource types', response)['resources']
        space_values = [
            (m[u'entity'][u'name'], m[u'metadata'][u'asset_id'], m[u'entity'][u'type'], m['entity']['status']) for
            m in datasource_details]

        table = self._list(space_values, [u'NAME', u'DATASOURCE_ID', u'TYPE', u'STATUS'], None, None)
        if return_as_df:
            return table

    def list(self, return_as_df=True):
        """Print all stored connections in a table format.

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed connections or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.connections.list()

        """
        datasource_details = self.get_details()
        space_values = [
            (m[u'entity'][u'name'], m[u'metadata'][u'id'], m['metadata']['create_time'],
             m['entity']['datasource_type']) for
            m in datasource_details['resources']]

        list_table = self._list(space_values, [u'NAME', u'ID', u'CREATED', 'DATASOURCE_TYPE_ID'], None, None)
        if return_as_df:
            return list_table

    def list_uploaded_db_drivers(self, return_as_df=True):
        """Print uploaded db driver jars in table a format. Supported for IBM Cloud Pak for Data only.

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed uploaded db drivers or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.connections.list_uploaded_db_drivers()

        """
        if not self._client.ICP:
            raise WMLClientError('Not supported on this environment.')

        try:
            if not self.get_uploaded_db_drivers():
                raise Exception('List empty for new api')
            table = self._list_uploaded_db_drivers_new_api()
        except:
            response = requests.get(
                self._client.service_instance._href_definitions.get_wsd_model_attachment_href() + 'dbdrivers',
                headers=self._client._get_headers(no_content_type=True),
                params=self._client._params()
            )
            jars = [[el['path'].split('/')[-1]] for el in response.json()['resources']]

            table = self._list(jars, [u'NAME'], None, None)
        if return_as_df:
            return table

    def get_uploaded_db_drivers(self): # TODO not yet documented
        # """
        #    Get uploaded db driver jar names and paths.
        #    Supported for IBM Cloud Pak for Data, version 4.6.1 and above.
        #
        #    **Output**
        #
        #    .. important::
        #         Returns dictionary containing name and path for connection files.\n
        #         **return type**: Dict[Str, Str]\n
        #
        #    **Example**
        #
        #     >>> result = client.connections.get_uploaded_db_drivers()
        #
        # """
        if not self._client.ICP:
            raise WMLClientError('Not supported on this environment.')

        response = requests.get(
            self._client.service_instance._href_definitions.get_connections_files_href(),
            headers=self._client._get_headers(no_content_type=True)
        )
        result = self._handle_response(200, u'get uploaded db drivers', response)['resources']
        return dict([(el['fileName'], el['url']) for el in result])

    def _list_uploaded_db_drivers_new_api(self):
        """List uploaded db driver jars. Supported for IBM Cloud Pak for Data only.

        .. important::
            This method only prints the uploaded db driver jar names.

        **Example**

        .. code-clock:: python

            client.connections._list_uploaded_db_drivers_new_api()

        """
        jars = [[name] for name in self.get_uploaded_db_drivers()]
        return self._list(jars, [u'NAME'], None, None)

    def get_datasource_type_uid_by_name(self, name):
        """Get stored datasource types id for the given datasource type name.

        :param name:  name of datasource type
        :type name: int

        :return: datasource Unique Id
        :rtype: str

        **Example**

        .. code-block:: python

            client.connections.get_datasource_type_uid_by_name('cloudobjectstorage')

        """
        Connections._validate_type(name, u'name', str, True)

        if self._client.WSD:
            header_param = self._client._get_headers(wsdconnection_api_req=True)
        else:
            header_param = self._client._get_headers()

        with self.requests_retry_session() as sess:
            response = sess.get(self._client.service_instance._href_definitions.get_connection_data_types_href(),
                                headers=header_param)

            datasource_id = 'None'
            datasource_details = self._handle_response(200, u'list datasource types', response)['resources']

        for i, ds_resource in enumerate(datasource_details):
            if ds_resource['entity']['name'] == name:
                datasource_id = ds_resource['metadata']['asset_id']
        return datasource_id

    def upload_db_driver(self, path: str):
        """Upload db driver jar. Supported for IBM Cloud Pak for Data only, version 4.0.4 and above.

        :param path:  path to db driver jar
        :type path: str

        **Example**

        .. code-block:: python

            client.connections.upload_db_driver('example/path/db2jcc4.jar')

        """
        if not self._client.ICP:
            raise WMLClientError('Not supported on this environment.')

        try:
            self._upload_db_driver_new_api(path)
        except:
            driver_file_name = path.split('/')[-1]

            with open(path, 'rb') as fdata:
                content_upload_url = self._client.service_instance._href_definitions.get_wsd_model_attachment_href() + \
                                     "dbdrivers/" + quote(driver_file_name, safe='')
                response = requests.put(
                    content_upload_url,
                    files={'file': ('native', fdata, 'application/octet-stream', {'Expires': '0'})},
                    headers=self._client._get_headers(no_content_type=True),
                    params=self._client._params()
                )

                self._client.repository._handle_response(201, 'uploading db driver jar', response)

    def _upload_db_driver_new_api(self, path: str):
        """Upload db driver jar. Supported for IBM Cloud Pak for Data only, version 4.6.1 and above.

        :param path:  path to db driver jar
        :type path: str

        **Example**

        .. code-block:: python

            client.connections._upload_db_driver_new_api('example/path/db2jcc4.jar')

        """
        if not self._client.ICP:
            raise WMLClientError('Not supported on this environment.')

        driver_file_name = path.split('/')[-1]

        with open(path, 'rb') as fdata:
            content_upload_url = self._client.service_instance._href_definitions.get_connections_file_href(quote(driver_file_name, safe=''))
            response = requests.post(
                content_upload_url,
                data=fdata,
                headers=self._client._get_headers(content_type='application/octet-stream')
            )

            if response.status_code == 403:
                raise WMLClientError("User is missing [configure_platform] permission to upload new jar file.")

            self._client.repository._handle_response(200, 'uploading db driver jar', response, json_response=False)

    def get_db_driver_url(self, name: str): # TODO not document yet
        # """
        # Get signed db driver jar url to be used during creating of JDBC generic connection. The jar name passed as argument needs to be uploaded into system first.
        # Supported for IBM Cloud Pak for Data only, version 4.6.1 and above.
        #
        # :param name:  db driver jar name
        # :type name: str
        #
        # **Example**
        #
        # .. code-block:: python
        #
        #     client.connections.get_db_driver_url('db2jcc4.jar')
        #
        # """
        if not self._client.ICP:
            raise WMLClientError('Not supported on this environment.')

        try:
            return self.get_uploaded_db_drivers()[name]
        except WMLClientError as e:
            raise e
        except:
            raise WMLClientError(f"Driver jar with name {name} not found.")

    def sign_db_driver_url(self, jar_name: str):
        """Get signed db driver jar url to be used during creating of JDBC generic connection.
        The jar name passed as argument needs to be uploaded into system first.
        Supported for IBM Cloud Pak for Data only, version 4.0.4 and above.

        :param jar_name:  db driver jar name
        :type jar_name: str

        :return: signed db driver url
        :rtype: str

        **Example**

        .. code-block:: python

            jar_uri = client.connections.sign_db_driver_url('db2jcc4.jar')

        """
        try:
            #if self._client.ICP_46:
            #    print("Warning: This function is deprecated. Use `get_db_driver_url(name)` instead.")
            res = self.get_db_driver_url(jar_name)
            return res
        except:
            if not self._client.ICP:
                raise WMLClientError('Not supported on this environment.')

            signed_url = self._client.service_instance._href_definitions.get_wsd_model_attachment_href() + \
                         quote("dbdrivers/" + jar_name, safe='') + "/signed"
            params = self._client._params()

            params['expires_in'] = 5000

            response = requests.post(
                signed_url,
                headers=self._client._get_headers(no_content_type=True),
                params=params
            )

            self._client.repository._handle_response(201, 'signing db driver url', response, json_response=False)

            return unquote(response.headers['Location'])

    @staticmethod
    def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504, 521, 524), session=None):
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry

        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
