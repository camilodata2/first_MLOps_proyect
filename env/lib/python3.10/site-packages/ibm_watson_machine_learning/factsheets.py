#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
import json
from ibm_watson_machine_learning.utils import INSTANCE_DETAILS_TYPE, print_text_header_h2
from ibm_watson_machine_learning.wml_client_error import WMLClientError, WrongMetaProps
from ibm_watson_machine_learning.wml_resource import WMLResource
from .metanames import FactsheetsMetaNames


class Factsheets(WMLResource):
    """Link WML Model to Model Entry."""

    cloud_platform_spaces = False
    icp_platform_spaces = False

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not client.CLOUD_PLATFORM_SPACES and not client.ICP_PLATFORM_SPACES:
            Factsheets._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Factsheets._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self._ICP = client.ICP
        self.ConfigurationMetaNames = FactsheetsMetaNames()

        if client.CLOUD_PLATFORM_SPACES:
            Factsheets.cloud_platform_spaces = True

        if client.ICP_PLATFORM_SPACES:
            Factsheets.icp_platform_spaces = True

    def register_model_entry(self, model_id, meta_props, catalog_id=None):
        """Link WML Model to Model Entry

        :param model_id: published model/asset ID
        :type model_id: str

        :param meta_props: metaprops, to see the available list of metanames use:

            .. code-block:: python

                client.factsheets.ConfigurationMetaNames.get()

        :type meta_props: dict

        :param catalog_id: catalog ID where you want to register model
        :type catalog_id: str, optional

        :return: metadata of the registration
        :rtype: dict

        **Example**

        .. code-block:: python

            meta_props = {
                wml_client.factsheets.ConfigurationMetaNames.ASSET_ID: '83a53931-a8c0-4c2f-8319-c793155e7517'}

            registration_details = client.factsheets.register_model_entry(model_id, catalog_id, meta_props)

        or

        .. code-block:: python

            meta_props = {
                wml_client.factsheets.ConfigurationMetaNames.NAME: "New model entry",
                wml_client.factsheets.ConfigurationMetaNames.DESCRIPTION: "New model entry"}

            registration_details = client.factsheets.register_model_entry(model_id, meta_props)

        """
        Factsheets._validate_type(model_id, u'model_id', str, True)
        Factsheets._validate_type(catalog_id, u'catalog_id', str, False)
        metaProps = self.ConfigurationMetaNames._generate_resource_metadata(meta_props)

        params = self._client._params()

        if catalog_id is not None:
            params['catalog_id'] = catalog_id
            if 'project_id' in params:
                del params['project_id']
            elif 'space_id' in params:
                del params['space_id']

        name_in = self.ConfigurationMetaNames.NAME in metaProps
        description_in = self.ConfigurationMetaNames.DESCRIPTION in metaProps
        asset_id_in = self.ConfigurationMetaNames.ASSET_ID in metaProps

        # check for metaprops correctness
        reason = "Please provide either NAME and DESCRIPTION or ASSET_ID"
        if name_in and description_in:
            if asset_id_in:
                raise WrongMetaProps(reason=reason)

        elif asset_id_in:
            if name_in or description_in:
                raise WrongMetaProps(reason=reason)

        else:
            raise WrongMetaProps(reason=reason)

        url = self._client.service_instance._href_definitions.get_wkc_model_register_href(model_id)

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            response = requests.post(
                url,
                json=metaProps,
                params=params,  # version is mandatory
                headers=self._client._get_headers())
        else:
            response = requests.post(
                url,
                json=metaProps,
                headers=self._client._get_headers())

        if response.status_code == 200:
            print_text_header_h2(f"Successfully finished linking WML Model '{model_id}' to Model Entry.")

        else:
            error_msg = u'WML Model registration failed'
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(error_msg + '. Error: ' + str(response.status_code) + '. ' + reason)

        return response.json()

    def list_model_entries(self, catalog_id=None):
        """Returns all WKC Model Entry assets for a catalog.

        :param catalog_id: catalog ID where you want to register model, if None list from all catalogs
        :type catalog_id: str, optional

        :return: all WKC Model Entry assets for a catalog
        :rtype: dict

        **Example**

        .. code-block:: python

            model_entries = client.factsheets.list_model_entries(catalog_id)

        """
        if catalog_id is not None:
            Factsheets._validate_type(catalog_id, u'catalog_id', str, True)
            url = self._client.service_instance._href_definitions.get_wkc_model_list_from_catalog_href(catalog_id)

        else:
            url = self._client.service_instance._href_definitions.get_wkc_model_list_all_href()

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            response = requests.get(
                url,
                params=self._client._params(),  # version is mandatory
                headers=self._client._get_headers())
        else:
            response = requests.get(
                url,
                headers=self._client._get_headers())

        if response.status_code == 200:
            return response.json()

        else:
            error_msg = u'WKC Models listing failed'
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(error_msg + '. Error: ' + str(response.status_code) + '. ' + reason)

    def unregister_model_entry(self, asset_id, catalog_id: str = None):
        """Unregister WKC Model Entry

        :param asset_id:  WKC model entry id
        :type asset_id: str

        :param catalog_id: catalog ID where asset is stored, when not provided,
            default client space or project will be taken
        :type catalog_id: str, optional

        **Example**

        .. code-block:: python

            model_entries = client.factsheets.unregister_model_entry(asset_id='83a53931-a8c0-4c2f-8319-c793155e7517',
                                                                     catalog_id='34553931-a8c0-4c2f-8319-c793155e7517')

        or

        .. code-block:: python

            client.set.default_space('98f53931-a8c0-4c2f-8319-c793155e7517')
            model_entries = client.factsheets.unregister_model_entry(asset_id='83a53931-a8c0-4c2f-8319-c793155e7517')

        """
        Factsheets._validate_type(asset_id, u'asset_id', str, True)
        Factsheets._validate_type(catalog_id, u'catalog_id', str, False)
        url = self._client.service_instance._href_definitions.get_wkc_model_delete_href(asset_id)

        params = self._client._params()
        if catalog_id is not None:
            params['catalog_id'] = catalog_id

            if 'space_id' in str(params):
                del params['space_id']

            elif 'project_id' in str(params):
                del params['project_id']

        response = requests.delete(
            url,
            params=params,  # version is mandatory
            headers=self._client._get_headers())

        if response.status_code == 204:
            print_text_header_h2(f"Successfully finished unregistering WKC Model '{asset_id}' Entry.")

        else:
            error_msg = u'WKC Model Entry unregistering failed'
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(error_msg + '. Error: ' + str(response.status_code) + '. ' + reason)
