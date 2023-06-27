__all__ = [
    "BaseDataConnection"
]

#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import io
import json
import os

from abc import ABC, abstractmethod
from typing import Union, Tuple, TYPE_CHECKING
from warnings import warn

import pandas as pd

import ibm_watson_machine_learning._wrappers.requests as requests
from pandas import concat, DataFrame, read_csv

from ibm_watson_machine_learning.utils.autoai.utils import try_load_dataset, try_load_tar_gz, try_import_autoai_libs
from ibm_watson_machine_learning.utils.autoai.enums import DataConnectionTypes
from ibm_watson_machine_learning.utils.autoai.errors import CannotReadSavedRemoteDataBeforeFit, WrongAssetType
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure, WMLClientError, DataStreamError
from ibm_watson_machine_learning.utils.utils import is_lib_installed
from ibm_watson_machine_learning.data_loaders.datasets.experiment import DEFAULT_SAMPLING_TYPE, DEFAULT_SAMPLE_SIZE_LIMIT

if TYPE_CHECKING:
    from ibm_boto3 import resource


class BaseDataConnection(ABC):
    """Base class for DataConnection."""

    def __init__(self):

        self.type = None
        self.connection = None
        self.location = None
        self.auto_pipeline_params = None
        self._wml_client = None
        self._run_id = None
        self._obm = None
        self._obm_cos_path = None
        self.id = None
        self._datasource_type = None

    @abstractmethod
    def _to_dict(self) -> dict:
        """Convert DataConnection object to dictionary representation."""
        pass

    @classmethod
    @abstractmethod
    def _from_dict(cls, _dict: dict) -> 'BaseDataConnection':
        """Create a DataConnection object from dictionary."""
        pass

    @abstractmethod
    def read(self, with_holdout_split: bool = False) -> Union['DataFrame', Tuple['DataFrame', 'DataFrame']]:
        """Download dataset stored in remote data storage."""
        pass

    @abstractmethod
    def write(self, data: Union[str, 'DataFrame'], remote_name: str) -> None:
        """Upload file to a remote data storage."""
        pass

    def _fill_experiment_parameters(self, prediction_type: str, prediction_column: str, holdout_size: float,
                                    csv_separator: str = ',', excel_sheet: Union[str, int] = None,
                                    encoding: str = 'utf-8') -> None:
        """To be able to recreate a holdout split, this method need to be called."""
        self.auto_pipeline_params = {
            'prediction_type': prediction_type,
            'prediction_column': prediction_column,
            'holdout_size': holdout_size,
            'csv_separator': csv_separator,
            'excel_sheet': excel_sheet,
            'encoding': encoding
        }

    def _download_obm_data_from_cos(self, cos_client: 'resource') -> 'DataFrame':
        """Download preprocessed OBM data. COS version."""

        if '//' in self._obm_cos_path:  # sometimes there is an additional slash, need to replace it
            self._obm_cos_path = self._obm_cos_path.replace('//', '/')

        # note: fetch all OBM file part names
        cos_summary = cos_client.Bucket(self.location.bucket).objects.filter(Prefix=self._obm_cos_path)
        file_names = [file_name.key for file_name in cos_summary]

        # note: if path does not exist, try to find in different one
        if not file_names:
            cos_summary = cos_client.Bucket(self.location.bucket).objects.filter(
                Prefix=self._obm_cos_path.split('./')[-1])
            file_names = [file_name.key for file_name in cos_summary]
            # --- end note
        # --- end note

        # TODO: this can be done simultaneously (multithreading / multiprocessing)
        # note: download all data parts and concatenate them into one output
        parts = []
        for file_name in file_names:
            file = cos_client.Object(self.location.bucket, file_name).get()
            buffer = io.BytesIO(file['Body'].read())
            parts.append(try_load_dataset(buffer=buffer))

        data = concat(parts)
        # --- end note
        return data

    def _download_obm_json(self) -> dict:
        """Download obm.json log."""
        if self._obm:
            if self._obm_cos_path:
                path = self._obm_cos_path.rsplit('features', 1)[0] + 'obm.json'
            else:
                path = f"{self.location.path}/{self._run_id}/data/obm/obm.json"
            data = self._download_json_file(path)
            return data
        else:
            raise Exception("OBM function called when not OBM scenario.")

    def _download_csv_file(self, path) -> dict:
        """Download csv file."""
        df = pd.DataFrame()

        if '//' in path:  # sometimes there is an additional slash, need to replace it
            path = path.replace('//', '/')

        if self.type == DataConnectionTypes.FS:
            csv_url = self._wml_client.service_instance._href_definitions.get_wsd_model_attachment_href() + f"/auto_ml/{path.split('/auto_ml/')[-1]}"

            with requests.get(csv_url, params=self._wml_client._params(), headers=self._wml_client._get_headers(),
                              stream=True) as file_response:
                if file_response.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading json"), file_response)

                downloaded_asset = file_response.content
                df = pd.read_csv(io.BytesIO(downloaded_asset))
                #json_content = json.loads(buffer.getvalue().decode('utf-8'))

        elif self.type == DataConnectionTypes.CA or self.type == DataConnectionTypes.CN:
            if self._check_if_connection_asset_is_s3():
                cos_client = self._init_cos_client()

                try:
                    file = cos_client.Object(self.location.bucket, path).get()
                    content = file["Body"].read()
                    df = pd.read_csv(io.BytesIO(content))

                except Exception as cos_access_exception:
                    raise ConnectionError(
                        f"Unable to access data object in cloud object storage with credentials supplied. "
                        f"Error: {cos_access_exception}")

            else:
                raise NotImplementedError(f"Unsupported connection type: {self.type}. "
                                          f"Datasource type is not supported. "
                                          f"Supported type is: bluemixcloudobjectstorage")

        else:
            raise NotImplementedError(f"Unsupported connection type: {self.type}")

        return df

    def _download_json_file(self, path) -> dict:
        """Download json file."""
        json_content = {}

        if '//' in path:  # sometimes there is an additional slash, need to replace it
            path = path.replace('//', '/')

        # TODO: remove S3 implementation
        if self.type == DataConnectionTypes.S3:
            warn(message="S3 DataConnection is deprecated! Please use data_asset_id instead.")

            cos_client = self._init_cos_client()

            try:
                file = cos_client.Object(self.location.bucket, path).get()
                content = file["Body"].read()
                json_content = json.loads(content.decode('utf-8'))
            except Exception as cos_access_exception:
                raise ConnectionError(
                    f"Unable to access data object in cloud object storage with credentials supplied. "
                    f"Error: {cos_access_exception}")
        elif self.type == DataConnectionTypes.FS:
            json_url = self._wml_client.service_instance._href_definitions.get_wsd_model_attachment_href() + f"/auto_ml/{path.split('/auto_ml/')[-1]}"

            with requests.get(json_url, params=self._wml_client._params(), headers=self._wml_client._get_headers(),
                              stream=True) as file_response:
                if file_response.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading json"), file_response)

                downloaded_asset = file_response.content
                buffer = io.BytesIO(downloaded_asset)
                json_content = json.loads(buffer.getvalue().decode('utf-8'))

        elif self.type == DataConnectionTypes.CA or self.type == DataConnectionTypes.CN:
            if self._check_if_connection_asset_is_s3():
                cos_client = self._init_cos_client()

                try:
                    file = cos_client.Object(self.location.bucket, path).get()
                    content = file["Body"].read()
                    json_content = json.loads(content.decode('utf-8'))

                except Exception as cos_access_exception:
                    raise ConnectionError(
                        f"Unable to access data object in cloud object storage with credentials supplied. "
                        f"Error: {cos_access_exception}")

            else:
                raise NotImplementedError(f"Unsupported connection type: {self.type}. "
                                          f"Datasource type is not supported. "
                                          f"Supported type is: bluemixcloudobjectstorage")

        else:
            raise NotImplementedError(f"Unsupported connection type: {self.type}")

        return json_content

    def _get_attachemnt_details(self, data_asset_id: str):
        response = requests.get(self._wml_client.service_instance._href_definitions.get_data_asset_href(data_asset_id),
                                params=self._wml_client._params(),
                                headers=self._wml_client._get_headers())
        data_asset_details = self._wml_client.data_assets._handle_response(200, u'Cannot get data asset details', response)

        attachments_data_asset_details = data_asset_details.get('attachments', [{}])[0]
        # note: Return attachment details from data asset details if it is not a connected data asset:
        if attachments_data_asset_details and attachments_data_asset_details.get('connection_id') is None:
            return attachments_data_asset_details
        else:
            attachment_id = attachments_data_asset_details.get('id')
            if not self._wml_client.ICP and not self._wml_client.WSD:
                response = requests.get(
                    self._wml_client.service_instance._href_definitions.get_attachment_href(data_asset_id, attachment_id),
                    params=self._wml_client._params(),
                    headers=self._wml_client._get_headers())
            else:
                response = requests.get(
                    self._wml_client.service_instance._href_definitions.get_attachment_href(data_asset_id, attachment_id),
                    params=self._wml_client._params(),
                    headers=self._wml_client._get_headers(),
                    verify=False)

            self._wml_client.data_assets._handle_response(200, u'Cannot get attachment details', response)

            return response.json()

    def _check_if_connection_asset_is_s3(self) -> bool:
        try:
            # fast return true if the connection is an instance of S3Connection and non s3 type, or its attribute
            # "cos_type" takes True
            from .connections import S3Connection
            if (isinstance(self.connection, S3Connection) and self.type != 's3' and self.connection.to_dict()) or (
                    self.connection is not None and getattr(self.connection, 'cos_type', False)):
                return True

            if self.type == 'data_asset':
                if self.connection is not None and hasattr(self.connection, 'href'):
                    items = self.connection.href.split('/')

                else:
                    items = self.location.href.split('/')
                _id = items[-1].split('?')[0]

                if self._wml_client is not None:
                    attachment_details = self._get_attachemnt_details(_id)
                    if 'connection_id' not in attachment_details:
                        if 'handle' in attachment_details and 'bucket' in attachment_details['handle']:
                            connection_details = self._create_conn_details_for_container()
                            attachment_details['connection_path'] = '/' + attachment_details['handle']['bucket'] + '/' + attachment_details['handle']['key']
                        elif 'url' in attachment_details and 'name' in attachment_details and 'bucket' in attachment_details:
                            connection_details = self._create_conn_details_for_container()
                            attachment_details['connection_path'] = '/' + attachment_details['bucket']['bucket_name'] + '/' + attachment_details['name']
                        else:
                            return False

                    else:
                        connection_details = self._wml_client.connections.get_details(
                            attachment_details['connection_id'])

                else:
                    try:
                        from project_lib import Project
                        project = Project.access()

                        # note: Check if asset is located directly in the project files
                        #       If yes it is not a connected data.
                        #       [Prevents unnecessary logging].
                        if any(file['asset_id'] == _id for file in project.get_files()):
                            return False
                        # --- end note

                        try:
                            connection_details = project.get_connected_data(_id)

                        except RuntimeError:  # when data asset is normal not s3 or database
                            return False

                        attachment_details = {'connection_path': connection_details.get('datapath')}

                    except ModuleNotFoundError:
                        raise NotImplementedError(f"This functionality can be run only on Watson Studio.")

            elif self.type == 'connection_asset':
                if self._wml_client is not None:
                    connection_details = self._wml_client.connections.get_details(self.connection.id)

                else:
                    try:
                        from project_lib import Project
                        project = Project.access()
                        connection_details = project.get_connection(self.connection.id)

                    except ModuleNotFoundError:
                        raise NotImplementedError(f"This functionality can be run only on Watson Studio.")

            elif self.type == 'container':
                connection_details = self._create_conn_details_for_container()

            elif self.type == 's3' or self.type == 'fs':
                return False

            else:
                raise WrongAssetType(asset_type=self.type)

            # Note: Check with project libs if connection points to S3
            if self._wml_client is not None:
                datasource_type = connection_details['entity']['datasource_type']
                self._datasource_type = datasource_type
                datasource_type_id_ibm_cos = self._wml_client.connections.get_datasource_type_uid_by_name(
                    'bluemixcloudobjectstorage')
                datasource_type_id_aws_cos = self._wml_client.connections.get_datasource_type_uid_by_name(
                    'cloudobjectstorage')

                if (datasource_type == datasource_type_id_ibm_cos or datasource_type == datasource_type_id_aws_cos or
                        datasource_type == 'bluemixcloudobjectstorage' or datasource_type == 'cloudobjectstorage'):
                    cos_type = True

                else:
                    cos_type = False

            elif self.type == 'container':
                cos_type = True

            elif 'url' in connection_details:
                cos_type = True
                connection_details['entity'] = {'properties': connection_details}

            else:
                cos_type = False
            # --- end note

            if cos_type:
                if 'cos_hmac_keys' in str(connection_details['entity']['properties']):
                    creds = json.loads(connection_details['entity']['properties']['credentials'])
                    connection_details['entity']['properties']['access_key'] = creds['cos_hmac_keys']['access_key_id']
                    connection_details['entity']['properties']['secret_key'] = creds['cos_hmac_keys'][
                        'secret_access_key']

                if self.connection is None:
                    from .connections import S3Connection
                    if ('api_key' in str(connection_details['entity']['properties']) and
                            'iam_url' in str(connection_details['entity']['properties'])):
                        self.connection = S3Connection(
                            api_key=connection_details['entity']['properties'].get('api_key'),
                            auth_endpoint=connection_details['entity']['properties'].get('iam_url'),
                            endpoint_url=connection_details['entity']['properties'].get('url'),
                            resource_instance_id=connection_details['entity']['properties'].get('resource_instance_id'),
                            _internal_use=True
                        )

                    else:
                        self.connection = S3Connection(
                            endpoint_url=connection_details['entity']['properties'].get('url'),
                            access_key_id=connection_details['entity']['properties'].get('access_key'),
                            secret_access_key=connection_details['entity']['properties'].get('secret_key'),
                            _internal_use=True
                        )

                else:
                    if ('api_key' in str(connection_details['entity']['properties']) and
                            'iam_url' in str(connection_details['entity']['properties'])):
                        self.connection.api_key = connection_details['entity']['properties'].get('api_key')
                        self.connection.auth_endpoint = connection_details['entity']['properties'].get('iam_url')
                        self.connection.endpoint_url = connection_details['entity']['properties'].get('url')
                        self.connection.resource_instance_id = connection_details['entity']['properties'].get('resource_instance_id')

                    else:
                        self.connection.secret_access_key = connection_details['entity']['properties'].get('secret_key')
                        self.connection.access_key_id = connection_details['entity']['properties'].get('access_key')
                        self.connection.endpoint_url = connection_details['entity']['properties'].get('url')

                if self.type == 'container':
                    self.location.bucket = connection_details['entity']['properties']['bucket_name']

                if self.type == 'data_asset':
                    try:
                        # Workaround for connected data assets promoted from catalog
                        if not attachment_details['connection_path'].startswith('/'):
                            attachment_details['connection_path'] = '/' + attachment_details['connection_path']
                        self.location.bucket = attachment_details['connection_path'].split('/')[1]

                        # Remove excel sheet name from connection_path if data asset is excel file:
                        if '.xlsx/' in attachment_details['connection_path'] or '.xls/' in attachment_details['connection_path']:
                            excel_sheet = attachment_details['connection_path'].split('/')[-1]
                            # note: set recognized excel sheet in params
                            if not self.auto_pipeline_params.get('excel_sheet'):
                                self.auto_pipeline_params['excel_sheet'] = excel_sheet
                            # end note
                            self.location.path = '/'.join(attachment_details['connection_path'].split('/')[2:-1])
                        else:
                            self.location.path = '/'.join(attachment_details['connection_path'].split('/')[2:])

                    except IndexError:
                        self.location.bucket = connection_details['entity']['properties']['bucket']
                        self.location.path = attachment_details['connection_path']

                    self.location.id = None
                self.connection.cos_type = True
                return True

            else:
                return False

        except Exception as e:
            if os.environ.get('USER_ACCESS_TOKEN') is None and os.environ.get('RUNTIME_ENV_ACCESS_TOKEN_FILE') is None:
                raise e

            else:
                return False  # if we are in WS, ignore this check even if there was some error

    def _is_data_asset_normal(self) -> bool:
        """Returns `True` if data asset is normal data asset - not connected data asset."""
        try:
            if self.type == 'data_asset':
                if self.connection is not None and hasattr(self.connection, 'href'):
                    items = self.connection.href.split('/')
                else:
                    items = self.location.href.split('/')
                _id = items[-1].split('?')[0]

                if self._wml_client is not None:
                    attachment_details = self._get_attachemnt_details(_id)
                    return bool('connection_id' not in attachment_details)

                else:
                    try:
                        from project_lib import Project
                        project = Project.access()

                        # note: Check if asset is located directly in the project files
                        #       If yes it is not a connected data.
                        #       [Prevents unnecessary logging].
                        return any(file['asset_id'] == _id for file in project.get_files())
                        # --- end note

                    except ModuleNotFoundError:
                        raise NotImplementedError(f"This functionality can be run only on Watson Studio.")

        except Exception as e:
            if os.environ.get('USER_ACCESS_TOKEN') is None and os.environ.get('RUNTIME_ENV_ACCESS_TOKEN_FILE') is None:
                raise e
            else:
                return False  # if we are in WS, ignore this check even if there was some error

    def _is_data_asset_nfs(self):
        if self.type == 'data_asset':
            if self.connection is not None and hasattr(self.connection, 'href'):
                items = self.connection.href.split('/')
            else:
                items = self.location.href.split('/')
            _id = items[-1].split('?')[0]

            if self._wml_client is not None:
                attachment_details = self._get_attachemnt_details(_id)

                return ('connection_id' in attachment_details and attachment_details.get(
                    'datasource_type') == self._wml_client.connections.get_datasource_type_uid_by_name('volumes'))
        return False


    def _download_indices_from_cos(self, cos_client: 'resource', location_path) -> 'DataFrame':
        """Download indices for this connection. COS version"""

        try:
            file = cos_client.Object(self.location.bucket,
                                     location_path).get()
        except:
            file = list(cos_client.Bucket(self.location.bucket).objects.filter(
                Prefix=location_path))[0].get()

        buffer = io.BytesIO(file['Body'].read())

        if '.csv' in location_path:
            file_name = 'indices.csv'
            with open(file_name, 'wb') as out:
                out.write(buffer.read())

            data = read_csv(file_name, sep=self.auto_pipeline_params.get('csv_separator', ','),
                            encoding=self.auto_pipeline_params.get('encoding', 'utf-8'))

        else:
            data = try_load_tar_gz(buffer=buffer,
                                   separator=self.auto_pipeline_params.get('csv_separator', ','),
                                   encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                   )

        return data

    def _download_training_data_from_data_asset_storage(self) -> 'DataFrame':
        """Download training data for this connection. Data Storage."""

        if self._wml_client is not None:
            # note: as we need to load a data into the memory,
            # we are using pure requests and helpers from the WML client
            asset_id = self.location.href.split('?')[0].split('/')[-1]

            # note: download data asset details
            asset_response = requests.get(
                self._wml_client.service_instance._href_definitions.get_data_asset_href(asset_id),
                params=self._wml_client._params(),
                headers=self._wml_client._get_headers())

            asset_details = self._wml_client.data_assets._handle_response(200, u'get assets', asset_response)

            attachment_id = asset_details['attachments'][0]['id']
            response = requests.get(
                self._wml_client.service_instance._href_definitions.get_attachment_href(asset_id, attachment_id),
                params=self._wml_client.data_assets._client._params(),
                headers=self._wml_client.data_assets._client._get_headers())

            if response.status_code == 200:
                try:
                    attachment_details = response.json()
                    if "url" in attachment_details:
                        file_asset_url = attachment_details['url']
                        if not file_asset_url.startswith("http"):
                            file_asset_url = self._wml_client.wml_credentials['url'] + file_asset_url
                        csv_response = requests.get(file_asset_url)

                        if csv_response.status_code != 200:
                            raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

                        downloaded_asset = csv_response.content

                        # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
                        buffer = io.BytesIO(downloaded_asset)
                        data = try_load_dataset(buffer=buffer,
                                                sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                                separator=self.auto_pipeline_params.get('csv_separator', ','),
                                                encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                                )

                        return data

                except requests.exceptions.MissingSchema:
                    pass  # go to 'handle' part and check if we are able to download data asset from WS

            # note: read the csv url
            if 'handle' in asset_details['attachments'][0]:
                attachment_url = asset_details['attachments'][0]['handle']['key']

                # note: make the whole url pointing out the csv
                artifact_content_url = (
                    f"{self._wml_client.service_instance._href_definitions.get_wsd_model_attachment_href()}"
                    f"{attachment_url}")

                # note: stream the whole CSV file
                csv_response = requests.get(artifact_content_url,
                                            params=self._wml_client._params(),
                                            headers=self._wml_client._get_headers(),
                                            stream=True)

                if csv_response.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

                downloaded_asset = csv_response.content

                # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
                buffer = io.BytesIO(downloaded_asset)
                data = try_load_dataset(buffer=buffer,
                                        sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                        separator=self.auto_pipeline_params.get('csv_separator', ','),
                                        encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                        )

                return data

            else:
                # NFS scenario
                connection_id = asset_details['attachments'][0]['connection_id']
                connection_path = asset_details['attachments'][0]['connection_path']

                return self._download_data_from_nfs_connection_using_id_and_path(connection_id, connection_path)

        else:
            try:
                from project_lib import Project
                project = Project.access()

            except ModuleNotFoundError:
                raise NotImplementedError(f"This functionality can be run only on Watson Studio.")

            asset_id = self.location.href.split('?')[0].split('/')[-1]
            assets_list = project.get_assets()

            data_asset_name = None
            for asset in assets_list:
                if asset['asset_id'] == asset_id:
                    data_asset_name = asset['name']

            if data_asset_name is None:
                raise FileNotFoundError(f"Cannot find data asset with id: {asset_id}")

            buffer = project.get_file(data_asset_name)
            data = try_load_dataset(buffer=buffer,
                                    sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                    separator=self.auto_pipeline_params.get('csv_separator', ','),
                                    encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                    )

            return data

    def _download_training_data_from_file_system(self, binary: bool = False) -> Tuple['DataFrame', bytes]:
        """Download training data for this connection. File system version."""

        try:
            url = self._wml_client.service_instance._href_definitions.get_wsd_model_attachment_href() + f"/{self.location.path.split('/assets/')[-1]}"
            # note: stream the whole CSV file
            csv_response = requests.get(url,
                                        params=self._wml_client._params(),
                                        headers=self._wml_client._get_headers(),
                                        stream=True)

            if csv_response.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

            downloaded_asset = csv_response.content

            if binary:
                return downloaded_asset

            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)
            data = try_load_dataset(buffer=buffer,
                                    sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                    separator=self.auto_pipeline_params.get('csv_separator', ','),
                                    encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                    )
        except (ApiRequestFailure, AttributeError):
            with open(self.location.path, 'rb') as data_buffer:
                if binary:
                    return data_buffer.read()

                data = try_load_dataset(buffer=data_buffer,
                                        sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                        separator=self.auto_pipeline_params.get('csv_separator', ','),
                                        encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                        )

        return data

    def _download_indices_from_file_system(self, location_path: str) -> 'DataFrame':
        """Download indices for this connection. File system version."""

        try:
            url = self._wml_client.service_instance._href_definitions.get_wsd_model_attachment_href() + f"/{location_path.split('/assets/')[-1]}"
            # note: stream the whole CSV file
            csv_response = requests.get(url,
                                        params=self._wml_client._params(),
                                        headers=self._wml_client._get_headers(),
                                        stream=True)

            if csv_response.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

            downloaded_asset = csv_response.content
            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)
            data = try_load_dataset(buffer=buffer,
                                    sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                    separator=self.auto_pipeline_params.get('csv_separator', ','),
                                    encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                    )
        except (ApiRequestFailure, AttributeError):
            with open(location_path, 'rb') as data_buffer:
                data = try_load_tar_gz(buffer=data_buffer,
                                       separator=self.auto_pipeline_params.get('csv_separator', ','),
                                       encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                       )

        return data

    def _download_obm_data_from_file_system(self) -> 'DataFrame':
        """Download preprocessed OBM data. FS version."""

        # note: fetch all OBM file part names
        url = self._wml_client.service_instance._href_definitions.get_wsd_model_attachment_href() + f"/auto_ml/{self.location.path.split('/auto_ml/')[-1]}/{self._run_id}/data/obm/features"
        params = self._wml_client._params()
        params['flat'] = "true"

        response = requests.get(url,
                                params=params,
                                headers=self._wml_client._get_headers())

        if response.status_code != 200:
            raise ApiRequestFailure(u'Failure during {}.'.format("getting files information"), response)

        file_names = [e['path'].split('/')[-1] for e in response.json()['resources'] if
                      e['type'] == 'file' and e['path'].split('/')[-1].startswith('part')]

        # TODO: this can be done simultaneously (multithreading / multiprocessing)
        # note: download all data parts and concatenate them into one output
        parts = []
        for file_name in file_names:
            csv_response = requests.get(url + '/' + file_name,
                                        params=self._wml_client._params(),
                                        headers=self._wml_client._get_headers(),
                                        stream=True)

            if csv_response.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

            downloaded_asset = csv_response.content
            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)
            parts.append(try_load_dataset(buffer=buffer))

        data = concat(parts)
        # --- end note
        return data

    def _download_data_from_nfs_connection(self) -> 'DataFrame':
        """Download training data for this connection. NFS."""

        # note: as we need to load a data into the memory,
        # we are using pure requests and helpers from the WML client
        data_path = self.location.path
        connection_id = self.connection.asset_id

        return self._download_data_from_nfs_connection_using_id_and_path(connection_id, data_path)

    def _download_data_from_nfs_connection_using_id_and_path(self, connection_id, connection_path) -> 'DataFrame':
        """Download training data for this connection. NFS."""

        # it means that it is on ICP env and it is before fit, so let's throw error
        if not self._wml_client:
            raise CannotReadSavedRemoteDataBeforeFit()

        buffer = None
        # Note: workaround with volumes API as connections API changes data format
        try:
            connection_details = self._wml_client.connections.get_details(connection_id)
        except ApiRequestFailure as conn_error:
            if os.environ.get('TRAINING_NFS_PATH'):
                # Note: Only viable on AutoAI runtime
                base_path = os.environ.get('TRAINING_NFS_PATH') + "/0"
                if connection_path.startswith('/'):
                    data_path = f"{base_path}{connection_path}"
                else:
                    data_path = f"{base_path}/{connection_path}"
                with open(data_path, 'rb') as f:
                    buffer = io.BytesIO(f.read())
            else:
                raise conn_error

        if buffer is None:
            href = self._wml_client.volumes._client.service_instance._href_definitions.volume_upload_href(
                connection_details['entity']['properties']['volume'])
            full_href = f"{href[:-1]}{connection_path}"

            csv_response = requests.get(full_href,
                                        headers=self._wml_client._get_headers(),
                                        stream=True)

            # Note: if file is written in directory we need to create different href for download
            if csv_response.status_code != 200:
                path_parts = connection_path.split('/')
                full_href = f"{href[:-1]}{'/'.join(path_parts[:-1])}%2F{path_parts[-1]}"

                csv_response = requests.get(full_href,
                                            headers=self._wml_client._get_headers(),
                                            stream=True)

                if csv_response.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading data"), csv_response)

            downloaded_asset = csv_response.content
            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)

        data = try_load_dataset(buffer=buffer,
                                sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                separator=self.auto_pipeline_params.get('csv_separator', ','),
                                encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                )

        return data

    def _download_data_from_cos(self, cos_client: 'resource') -> 'DataFrame':
        """Download training data for this connection. COS version"""
        location = self.location.file_name if hasattr(self.location, "file_name") else self.location.path
        try:
            file = cos_client.Object(self.location.bucket,
                                     location).get()
        except Exception as e:
            cos_objects_list = list(cos_client.Bucket(self.location.bucket).objects.filter(
                Prefix=location))
            if len(cos_objects_list) > 0:
                file = list(cos_client.Bucket(self.location.bucket).objects.filter(
                    Prefix=location))[0].get()
            else:
                raise e

        buffer = io.BytesIO(file['Body'].read())
        data = try_load_dataset(buffer=buffer,
                                sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                separator=self.auto_pipeline_params.get('csv_separator', ','),
                                encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                )

        return data

    def _create_conn_details_for_container(self) -> dict:
        if self._wml_client is not None:
            if self._wml_client.default_space_id is not None:
                details = self._wml_client.spaces.get_details(self._wml_client.default_space_id)

            else:
                project_url = self._wml_client.service_instance._href_definitions.get_project_href(
                    self._wml_client.default_project_id)

                response = requests.get(
                    project_url,
                    headers=self._wml_client._get_headers(projects_token=True)
                )

                details = response.json()

            properties = details['entity']['storage']['properties']
            creds = details['entity']['storage']['properties']['credentials'].get('admin')
            creds = creds if creds else details['entity']['storage']['properties']['credentials'].get('editor')
            properties.update(creds)
            properties['url'] = properties['endpoint_url']
            properties['access_key'] = properties.get('access_key_id')
            properties['secret_key'] = properties.get('secret_access_key')

            connection_details = {
                'entity': {
                    'datasource_type': 'bluemixcloudobjectstorage',
                    'properties': properties
                }
            }

        else:
            try:
                from project_lib import Project
                token = self._get_token_from_environment()

                if token is None:
                    raise NotImplementedError(
                        f"""To succesfully read the training data used in AutoAI experiment, you need to provide the project token.
                    **To insert the project token to your notebook:**
                        Click the More icon on your notebook toolbar and then click Insert project token.
                        Run the inserted code cell.
                    Note:
                    If you are told in a message that no project token exists, click the link in the message to be redirected to the project's Settings page where you can create a project token.
                    **To create a project token:**
                        Click New token in the Access tokens section on the Settings page of your project.
                        Enter a name, select Editor role for the project, and create a token.
                        Go back to your notebook, click the More icon on the notebook toolbar and then click Insert project token.
                        Run the inserted code cell."""
                    )

                token = token.split('Bearer ')[-1]

                _project = Project(project_id=os.environ.get('PROJECT_ID'),
                                   project_access_token=token)

                details = _project.get_storage_metadata()

                properties = details['properties']
                properties.update(properties['credentials']['editor'])
                properties['url'] = properties['endpoint_url']
                properties['access_key'] = properties.get('access_key_id')
                properties['secret_key'] = properties.get('secret_access_key')

                connection_details = {
                    'entity': {
                        'datasource_type': 'bluemixcloudobjectstorage',
                        'properties': properties
                    }
                }

            except ModuleNotFoundError:
                raise NotImplementedError(f"This functionality can be run only on Watson Studio.")

        return connection_details

    def _download_data_from_flight_service(self,
                                           data_location: 'DataConnection',
                                           binary: bool = False,
                                           read_to_file: str = None,
                                           flight_parameters: dict = None,
                                           headers: dict = None,
                                           number_of_batch_rows: int = None,
                                           sampling_type: str = DEFAULT_SAMPLING_TYPE,
                                           return_data_as_iterator: bool = False,
                                           enable_sampling: bool = True,
                                           _return_subsampling_stats: bool = False,
                                           total_size_limit=DEFAULT_SAMPLE_SIZE_LIMIT,
                                           total_nrows_limit=None,
                                           total_percentage_limit=1.0):

        is_lib_installed(lib_name='pyarrow', minimum_version='3.0.0', install=True)

        # import the class only if flight scenario is enabled - do not import it in main import section
        from ibm_watson_machine_learning.data_loaders.experiment import ExperimentDataLoader
        from ibm_watson_machine_learning.data_loaders.datasets.experiment import ExperimentIterableDataset

        if flight_parameters is None:
            flight_parameters = {"num_partitions": 4}

        dict_connection = data_location._to_dict()

        experiment_metadata = {
            "n_parallel_data_connections": flight_parameters.get("num_partitions", 4),
            "prediction_column": self.auto_pipeline_params.get('prediction_column'),
            "prediction_type": self.auto_pipeline_params.get('prediction_type'),
            "project_id": self._wml_client.default_project_id,
            "space_id": self._wml_client.default_space_id,
            'headers': self._wml_client._get_headers() if headers is None else headers
        }

        experiment_metadata.update(self.auto_pipeline_params)

        experiment_iterable_dataset_setup_parameters = dict(
            connection=dict_connection,
            enable_sampling=enable_sampling if not binary else False,  # don't use sampling to read binary
            experiment_metadata=experiment_metadata,
            binary_data=binary,
            read_to_file=read_to_file,
            flight_parameters=flight_parameters if flight_parameters is not None else {},
            fallback_to_one_connection=False,
            _return_subsampling_stats=_return_subsampling_stats,
            number_of_batch_rows=number_of_batch_rows,
            sampling_type=sampling_type,
            _wml_client=self._wml_client,
            total_size_limit=total_size_limit,
            total_nrows_limit=total_nrows_limit,
            total_percentage_limit=total_percentage_limit
        )

        iterable_dataset = ExperimentIterableDataset(**experiment_iterable_dataset_setup_parameters)
        data_loader = ExperimentDataLoader(dataset=iterable_dataset)

        try:

            if not return_data_as_iterator:
                for data in data_loader:
                    return data
            else:
                return data_loader
        except TypeError as e1:
            # note: retry if there is problem with types:
            # try download data with  infer_as_varchar set to 'true' if some error occurs
            # final data downloaded and converted to proper types might has smaller size than sample_size_limit,
            # because data limit is calculated on data downloaded as varchar, before the conversion to more optimal type.
            try:
                iterable_dataset = ExperimentIterableDataset(**experiment_iterable_dataset_setup_parameters)

                iterable_dataset.connection.infer_as_varchar = 'true'
                data_loader = ExperimentDataLoader(dataset=iterable_dataset)

                if not return_data_as_iterator:
                    for data in data_loader:
                        return data
                else:
                    return data_loader
            except Exception as e2:
                raise DataStreamError(f"First attempt of downloading data failed with error: {e1}. \n"
                                      f"Retry and use infer as varchar also failed with error: {e2}.")

    def _upload_data_via_flight_service(self,
                                        data_location: 'DataConnection',
                                        data: DataFrame = None,
                                        file_path: str = None,
                                        remote_name: str = None,
                                        flight_parameters: dict = None,
                                        headers: dict = None):
        is_lib_installed(lib_name='pyarrow', minimum_version='3.0.0', install=True)

        # import the class only if flight scenario is enabled - do not import it in main import section
        from ibm_watson_machine_learning.data_loaders.datasets.experiment import ExperimentIterableDataset

        if flight_parameters is None:
            flight_parameters = {"num_partitions": 1}

        dict_connection = data_location._to_dict()

        if remote_name:
            dict_connection['location']['path'] = self._get_path_with_remote_name(dict_connection, remote_name)
        elif dict_connection.get('location', {}).get('file_name'):
            dict_connection['location']['path'] = dict_connection['location']['file_name']

        experiment_metadata = {
            "n_parallel_data_connections": flight_parameters.get("num_partitions", 1),
            "project_id": self._wml_client.default_project_id,
            "space_id": self._wml_client.default_space_id,
            'headers': self._wml_client._get_headers() if headers is None else headers
        }

        experiment_metadata.update(self.auto_pipeline_params)

        iterable_dataset = ExperimentIterableDataset(
            connection=dict_connection,
            experiment_metadata=experiment_metadata,
            binary_data=True if file_path is not None else False,
            flight_parameters=flight_parameters if flight_parameters is not None else {},
            _wml_client=self._wml_client
        )

        if data is not None:
            try:
                iterable_dataset.write(data=data)

            except Exception as e:
                if 'gRPC message exceeds maximum size' in str(e):
                    raise ValueError(f"Exceeds maximum data size. Please provide data file path "
                                     f"instead of the pandas DataFrame to upload data in binary mode. Error: {e}")

                else:
                    raise e

        else:
            iterable_dataset.write(file_path=file_path)

    @staticmethod
    def _get_path_with_remote_name(dict_connection: dict, remote_name: str) -> str:
        if dict_connection.get('location', {}).get('path'):
            updated_path = dict_connection['location']['path'] + '/' + remote_name
            updated_path = updated_path.replace('//', '/')
        elif dict_connection.get('location', {}).get('file_name'):
            actual_path = dict_connection['location']['file_name']
            last_slash_index = actual_path.rfind('/') if '/' in actual_path else 0

            if '.' in actual_path[last_slash_index:]:
                actual_path = actual_path[:-last_slash_index]

            if actual_path:
                updated_path = actual_path + '/' + remote_name
                updated_path = updated_path.replace('//', '/')
            else:
                updated_path = remote_name
        else:
            updated_path = remote_name

        return updated_path

    @staticmethod
    def _get_token_from_environment():
        if os.environ.get('RUNTIME_ENV_ACCESS_TOKEN_FILE'):
            with open(os.environ.get('RUNTIME_ENV_ACCESS_TOKEN_FILE'), 'r') as f:
                token = f.read()
        else:
            token = os.environ.get('USER_ACCESS_TOKEN')

        if token:
            token = token.replace('Bearer ', '')

        return token

    def _is_size_acceptable(self):
        """
        Checks if data asset size is acceptable to download based on available memory in pod (MEM env variable)/ T-shirt size.
        Returns: True when data asset size is equal or lower than T-shirt limitation or when limitation is not set (outside autoai pod).
                False when data asset size is known and is above supported limit.
                None when data asset size is unknown or data_connection is not data asset type.
        """
        from ibm_watson_machine_learning.utils.autoai.connection import get_max_sample_size_limit

        if self._wml_client and self.type == 'data_asset':
            if self.connection is not None and hasattr(self.connection, 'href'):
                items = self.connection.href.split('/')
            else:
                items = self.location.href.split('/')
            data_asset_id = items[-1].split('?')[0]
            asset_size = self._wml_client.data_assets.get_details(data_asset_id).get('metadata', {}).get("size", 0)

            if asset_size == 0:                                 # data size unknown
                return None
            elif asset_size <= get_max_sample_size_limit():     # data size is within acceptable range
                return True
            elif not os.environ.get('MEM', False):
                return True                                     # no limitation were set
            else:
                return False                                    # data size is above supported limit
        else:
            return None                                         # not a data asset



