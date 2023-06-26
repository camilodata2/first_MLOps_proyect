#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.metanames import VolumeMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
import os
import time
import shlex
import subprocess
_DEFAULT_LIST_LENGTH = 50


class Volume(WMLResource):
    """Store and manage volume assets."""

    ConfigurationMetaNames = VolumeMetaNames()
    """MetaNames for volume assets creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    def get_details(self, volume_id):
        """Get volume details.

        :param volume_id: Unique ID of the volume
        :type volume_id: str

        :return: metadata of the volume details
        :rtype: dict

        **Example**

        .. code-block:: python

            volume_details = client.volumes.get_details(volume_id)

        """
        Volume._validate_type(volume_id, u'volume_id', str, True)

        params = {'addon_type': 'volumes',
                  'include_service_status': True
                  }

        response = requests.get(self._client.service_instance._href_definitions.volume_href(volume_id),
                                headers=self._client._get_headers(zen=True))

        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code, response.text)
            raise WMLClientError("Failed to Get the volume details. Try again.")

    def create(self, meta_props):
        """Create a volume asset.

        :param meta_props: metadata of the volume asset
        :type meta_props: dict

        :return: metadata of the created volume details
        :rtype: dict

        **Examples**

        Provision new PVC volume:

        .. code-block:: python

            metadata = {
                client.volumes.ConfigurationMetaNames.NAME: 'volume-for-wml-test',
                client.volumes.ConfigurationMetaNames.NAMESPACE: 'wmldev2',
                client.volumes.ConfigurationMetaNames.STORAGE_CLASS: 'nfs-client'
                client.volumes.ConfigurationMetaNames.STORAGE_SIZE: "2G"
            }

            asset_details = client.volumes.store(meta_props=metadata)

        Provision an existing PVC volume:

        .. code-block:: python

            metadata = {
                client.volumes.ConfigurationMetaNames.NAME: 'volume-for-wml-test',
                client.volumes.ConfigurationMetaNames.NAMESPACE: 'wmldev2',
                client.volumes.ConfigurationMetaNames.EXISTING_PVC_NAME: 'volume-for-wml-test'
            }

            asset_details = client.volumes.store(meta_props=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Failed to create volume. It is supported only for CP4D 3.5")

        volume_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client
        )

        create_meta = {}
        if self.ConfigurationMetaNames.EXISTING_PVC_NAME in meta_props and \
            meta_props[self.ConfigurationMetaNames.EXISTING_PVC_NAME] is not None:
            if self.ConfigurationMetaNames.STORAGE_CLASS in meta_props and \
                    meta_props[self.ConfigurationMetaNames.STORAGE_CLASS] is not None:
                raise WMLClientError("Failed while creating volume. Either provide EXISTING_PVC_NAME to create a volume using existing volume or"
                                     "provide STORAGE_CLASS and STORAGE_SIZE for new volume creation")
            else:
                create_meta.update({ "existing_pvc_name": meta_props[self.ConfigurationMetaNames.EXISTING_PVC_NAME]})
        else:
            if self.ConfigurationMetaNames.STORAGE_CLASS in meta_props and \
               meta_props[self.ConfigurationMetaNames.STORAGE_CLASS] is not None:
               if self.ConfigurationMetaNames.STORAGE_SIZE in meta_props and \
                       meta_props[self.ConfigurationMetaNames.STORAGE_SIZE] is not None:
                   create_meta.update({"storageClass": meta_props[self.ConfigurationMetaNames.STORAGE_CLASS]})
                   create_meta.update({"storageSize": meta_props[self.ConfigurationMetaNames.STORAGE_SIZE]})
               else:
                   raise WMLClientError("Failed to create volume. Missing input STORAGE_SIZE" )

        if self.ConfigurationMetaNames.EXISTING_PVC_NAME in meta_props and meta_props[self.ConfigurationMetaNames.EXISTING_PVC_NAME] is not None:
            input_meta = {
                "addon_type":"volumes",
                "addon_version":"-",
                "create_arguments":{
                    "metadata":create_meta
                },
                "namespace":meta_props[self.ConfigurationMetaNames.NAMESPACE],
                "display_name":meta_props[self.ConfigurationMetaNames.NAME]
            }
        else:
            input_meta = {
                "addon_type": "volumes",
                "addon_version": "-",
                "create_arguments": {
                    "metadata": create_meta
                },
                "namespace": meta_props[self.ConfigurationMetaNames.NAMESPACE],
                "display_name": meta_props[self.ConfigurationMetaNames.NAME]
            }
        creation_response = {}
        try:
            if self._client.CLOUD_PLATFORM_SPACES:
                creation_response = requests.post(
                    self._client.service_instance._href_definitions.volumes_href(),
                    headers=self._client._get_headers(zen=True),
                    json=input_meta
                )

            else:
                creation_response = requests.post(self._client.service_instance._href_definitions.volumes_href(),
                        headers=self._client._get_headers(zen=True),
                        json=input_meta
                    )
            if creation_response.status_code == 200:
                volume_id_details = creation_response.json()  # messy details returned for backward compability
                import copy
                volume_details = copy.deepcopy(input_meta)
                volume_details.update(volume_id_details)
                actual_details = self.get_details(self.get_id(volume_id_details))
                volume_details.update(actual_details)
                return volume_details
            else:
                print(creation_response.status_code, creation_response.text)
                raise WMLClientError("Failed to create a volume. Try again.")
        except Exception as e:
            print("Exception: ", {e})
            raise WMLClientError("Failed to create a volume. Try again.")

    def start(self, name, wait_for_available=False):
        """Start the volume service.

        :param name: unique name of the volume to be started
        :type name: str

        :param wait_for_available: flag indicating if method should wait until volume service is available
        :type wait_for_available: bool

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.volumes.start(volume_name)

        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.ICP_PLATFORM_SPACES and not self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError("Volume APIs are not supported. It is supported only for CP4D 3.5")

        if (self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47) and '::' not in name:
            raise WMLClientError('Invalid name to start volume. Correct volume name format: `<namespace>::<name>`. Retrieve the correct name using `client.volumes.get_name(client.volumes.get_details(volume_id))` command.')

        start_url = self._client.service_instance._href_definitions.volume_service_href(name)
        # Start the volume  service
        start_data = {}
        try:
            start_data = {}
            creation_response = requests.post(
                start_url,
                headers=self._client._get_headers(zen=True),
                json=start_data
            )
            if creation_response.status_code == 200:
                print("Volume Service started")
                if wait_for_available:
                    retries = 0
                    volume_status = False
                    while True and retries < 60 and not volume_status:
                        volume_status = self.get_volume_status(name)
                        time.sleep(5)
                        retries += 1
                    if not volume_status:
                        print("Volume Service has been started, but it is not available yet."
                              "Check volume availability using get_volume_status method.")

            elif creation_response.status_code == 500:
                print("Failed to start the volume. Make sure volume is in running with status RUNNING or UNKNOW and then re-try")
            else:
                print(creation_response.status_code, creation_response.text)
                raise WMLClientError("Failed to start the file to  volume. Try again.")
        except Exception as e:
            print("Exception:", {e})
            raise WMLClientError("Failed to start the file to  volume. Try again.")

    def get_volume_status(self, name):
        """Monitor a volume's file server status.

        :param name: name of the volume to retrieve status for
        :type name: str

        :return: status of the volume (True if volume is available, otherwise False)
        :rtype: bool

        **Example**

        .. code-block:: python

            client.volumes.get_volume_status(volume_name)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if not self._client.ICP_PLATFORM_SPACES and not self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError("Volume APIs are not supported. It is supported only for CP4D 3.5")

        if (self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47) and '::' not in name:
            raise WMLClientError('Invalid name to start volume. Correct volume name format: `<namespace>::<name>`. Retrieve the correct name using `client.volumes.get_name(client.volumes.get_details(volume_id))` command.')

        monitor_url = self._client.service_instance._href_definitions.volume_monitor_href(name)
        try:
            monitor_response = requests.get(
                monitor_url,
                headers=self._client._get_headers(zen=True)
            )
            if monitor_response.status_code == 200:
                return True
            elif monitor_response.status_code == 502:
                return False
            else:
                print(monitor_response.status_code, monitor_response.text)
                raise WMLClientError("Cannot retrieve status of the volume.")
        except Exception as e:
            print("Exception:", {e})
            raise WMLClientError("Cannot retrieve status of the volume.")

    def upload_file(self, name,  file_path):
        """Upload the data file into stored volume.

        :param name: unique name of the stored volume
        :type name: str
        :param file_path: file to be uploaded into the volume
        :type file_path: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.volumes.upload_file('testA', 'DRUG.csv')

        """
        WMLResource._chk_and_block_create_update_for_python36(self)

        if not self._client.ICP_PLATFORM_SPACES and not self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError("Volume APIs are not supported. It is supported only for CP4D 3.5 and above.")

        header_input = self._client._get_headers(zen=True)
        zen_token = header_input.get('Authorization')

        filename_to_upload = file_path.split('/')[-1]
        upload_url_file = self._client.service_instance._href_definitions.volume_upload_href(name) + filename_to_upload
        cmd_str = 'curl -k  -X PUT "' + upload_url_file + '"' + "  -H 'Content-Type: multipart/form-data' -H 'Authorization: " + zen_token + \
                  "' -F upFile='@" + file_path + "'"
        args = shlex.split(cmd_str)
        upload_response = subprocess.run(args, capture_output=True, text=True)
        if upload_response.returncode == 0:
            import json
            try:
                cmd_output = json.loads(upload_response.stdout)
                print(cmd_output.get('message'))
                return "SUCCESS"
            except Exception as e:
                print(upload_response.returncode, upload_response.stdout)
                print("Failed to upload the file to  volume. Try again.")
                return "FAILED"
        else:
            print(upload_response.returncode, upload_response.stdout, upload_response.stderr)
            print("Failed to upload the file to  volume. Try again.")
            return "FAILED"

    def list(self, return_as_df=True):
        """Print stored volumes in a table format. If limit is set to None there will be only first 50 records shown.

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed volumes or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.volumes.list()
        """

        href = self._client.service_instance._href_definitions.volumes_href()
        params = {}
        params.update({'addon_type': 'volumes'})

        response = requests.get(href, params=params, headers=self._client._get_headers(zen=True))

        asset_details = self._handle_response(200, u'list volumes', response)
        asset_list = asset_details.get('service_instances')
        volume_values = [
            (m[u'display_name'],
             m[u'id'],
             m['provision_status']) for m in asset_list]

        table = self._list(volume_values, [u'NAME', u'ID', u'PROVISION_STATUS'], None, _DEFAULT_LIST_LENGTH)

        if return_as_df:
            return  table


    @staticmethod
    def get_id(volume_details):
        """Get unique Id of stored volume details.

        :param volume_details: metadata of the stored volume details
        :type volume_details: dict

        :return: unique Id of stored volume asset
        :rtype: str

        **Example**

        .. code-block:: python

            volume_uid = client.volumes.get_id(volume_details)

        """

        Volume._validate_type(volume_details, u'asset_details', object, True)
        if 'service_instance' in volume_details and  volume_details.get('service_instance') is not None:
            vol_details = volume_details.get('service_instance')
            return WMLResource._get_required_element_from_dict(vol_details, u'volume_assets_details',
                                                               [u'id'])
        else:
            return WMLResource._get_required_element_from_dict(volume_details, u'volume_assets_details',
                                                           [u'id'])

    @staticmethod
    def get_name(volume_details):
        """Get unique name of stored volume asset.

        :params volume_details: metadata of the stored volume asset
        :type volume_details: dict

        :return: unique name of stored volume asset
        :rtype: str

        **Example**

        .. code-block:: python

            volume_name = client.volumes.get_name(asset_details)

        """
        Volume._validate_type(volume_details, u'asset_details', object, True)
        if 'service_instance' in volume_details and  volume_details.get('service_instance') is not None:
            vol_details = volume_details.get('service_instance')
            return WMLResource._get_required_element_from_dict(vol_details, u'volume_assets_details',
                                                               [u'display_name'])
        else:
            return WMLResource._get_required_element_from_dict(volume_details, u'volume_assets_details',
                                                           [u'display_name'])

    def delete(self, volume_id):
        """Delete a volume.

        :param volume_id: unique ID of the volume
        :type volume_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.volumes.delete(volume_id)

        """
        Volume._validate_type(volume_id, u'asset_uid', str, True)
        if (not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES):
            raise WMLClientError(u'Volume API is not supported.')

        response = requests.delete(self._client.service_instance._href_definitions.volume_href(volume_id),
                                   headers=self._client._get_headers(zen=True))

        if response.status_code == 200 or response.status_code == 204:
            print("Successfully deleted volume service.")
            return "SUCCESS"
        else:
            print("Failed to delete volume.")
            print(response.status_code, response.text)
            return "FAILED"

    def stop(self, volume_name):
        """Stop the volume service.

        :param volume_name: unique name of the volume
        :type volume_name: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.volumes.stop(volume_name)

        """
        Volume._validate_type(volume_name, u'asset_uid', str, True)
        if (not self._client.CLOUD_PLATFORM_SPACES and not self._client.ICP_PLATFORM_SPACES):
            raise WMLClientError(u'Volume API is not supported.')

        response = requests.delete(self._client.service_instance._href_definitions.volume_service_href(volume_name),
                                   headers=self._client._get_headers(zen=True))

        if response.status_code == 200:
            print("Successfully stopped volume service.")
            return "SUCCESS"
        else:
            print(response.status_code, response.text)
            return "FAILED"

