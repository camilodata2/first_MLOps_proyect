#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.utils import SW_SPEC_DETAILS_TYPE
from ibm_watson_machine_learning.metanames import SwSpecMetaNames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
import os,json

_DEFAULT_LIST_LENGTH = 50


class SwSpec(WMLResource):
    """Store and manage software specs."""

    ConfigurationMetaNames = SwSpecMetaNames()
    """MetaNames for Software Specification creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP
        self.software_spec_list = None
        if self._client.WSD_20:
            self.software_spec_list = {
                                "default_py3.6": "0062b8c9-8b7d-44a0-a9b9-46c416adcbd9",
                                "scikit-learn_0.20-py3.6": "09c5a1d0-9c1e-4473-a344-eb7b665ff687",
                                "ai-function_0.1-py3.6": "0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda",
                                "shiny-r3.6": "0e6e79df-875e-4f24-8ae9-62dcc2148306",
                                "pytorch_1.1-py3.6": "10ac12d6-6b30-4ccd-8392-3e922c096a92" ,
                                "scikit-learn_0.22-py3.6": "154010fa-5b3b-4ac1-82af-4d5ee5abbc85",
                                "default_r3.6":  "1b70aec3-ab34-4b87-8aa0-a4a3c8296a36",
                                "tensorflow_1.15-py3.6":  "2b73a275-7cbf-420b-a912-eae7f436e0bc",
                                "pytorch_1.2-py3.6":  "2c8ef57d-2687-4b7d-acce-01f94976dac1",
                                "spark-mllib_2.3":  "2e51f700-bca0-4b0d-88dc-5c6791338875",
                                "pytorch-onnx_1.1-py3.6-edt": "32983cea-3f32-4400-8965-dde874a8d67e",
                                "spark-mllib_2.4": "390d21f8-e58b-4fac-9c55-d7ceda621326",
                                "xgboost_0.82-py3.6":  "39e31acd-5f30-41dc-ae44-60233c80306e",
                                "pytorch-onnx_1.2-py3.6-edt": "40589d0e-7019-4e28-8daa-fb03b6f4fe12",
                                "spark-mllib_2.4-r_3.6": "49403dff-92e9-4c87-a3d7-a42d0021c095",
                                "xgboost_0.90-py3.6": "4ff8d6c2-1343-4c18-85e1-689c965304d3",
                                "pytorch-onnx_1.1-py3.6":  "50f95b2a-bc16-43bb-bc94-b0bed208c60b",
                                "spark-mllib_2.4-scala_2.11": "55a70f99-7320-4be5-9fb9-9edb5a443af5",
                                "spss-modeler_18.1":  "5c3cad7e-507f-4b2a-a9a3-ab53a21dee8b",
                                "spark-mllib_2.3-r_3.6":  "6586b9e3-ccd6-4f92-900f-0f8cb2bd6f0c",
                                "spss-modeler_18.2":  "687eddc9-028a-4117-b9dd-e57b36f1efa5",
                                "pytorch-onnx_1.2-py3.6":  "692a6a4d-2c4d-45ff-a1ed-b167ee55469a",
                                "do_12.9":  "75a3a4b0-6aa0-41b3-a618-48b1f56332a6",
                                "spark-mllib_2.3-scala_2.11": "7963efe5-bbec-417e-92cf-0574e21b4e8d",
                                "caffe_1.0-py3.6":  "7bb3dbe2-da6e-4145-918d-b6d84aa93b6b",
                                "cuda-py3.6":  "82c79ece-4d12-40e6-8787-a7b9e0f62770",
                                "hybrid_0.1":  "8c1a58c6-62b5-4dc4-987a-df751c2756b6",
                                "caffe-ibm_1.0-py3.6":  "8d863266-7927-4d1e-97d7-56a7f4c0a19b",
                                "spss-modeler_17.1":  "902d0051-84bd-4af6-ab6b-8f6aa6fdeabb",
                                "do_12.10":  "9100fd72-8159-4eb9-8a0b-a87e12eefa36",
                                "hybrid_0.2":  "9b3f9040-9cee-4ead-8d7a-780600f542f7"}

    def get_details(self, sw_spec_uid=None, state_info=False):
        """Get software specification details. If no sw_spec_id is passed, details for all software specifications
        will be returned.

        :param sw_spec_uid: UID of software specification
        :type sw_spec_uid: str, optional

        :param state_info: works only when `sw_spec_uid` is None, instead of returning details of software specs returns
            state of software specs information (supported, unsupported, deprecated), containing suggested replacement
            in case of unsupported or deprecated software specs

        :return: metadata of the stored software specification(s)
        :rtype:
          - **dict** - if `sw_spec_uid` is not None
          - **{"resources": [dict]}** - if `sw_spec_uid` is None

        **Examples**

        .. code-block:: python

            sw_spec_details = client.software_specifications.get_details(sw_spec_uid)
            sw_spec_details = client.software_specifications.get_details()
            sw_spec_state_details = client.software_specifications.get_details(state_info=True)
        """
        if self._client.WSD_20:
            raise WMLClientError(u'get_details API is not supported in Watson Studio Desktop.')

        SwSpec._validate_type(sw_spec_uid, u'sw_spec_uid', str, False)

        if sw_spec_uid:
            if not self._ICP or self._client.ICP_PLATFORM_SPACES:
                response = requests.get(self._client.service_instance._href_definitions.get_sw_spec_href(sw_spec_uid),
                                        params=self._client._params(skip_space_project_chk=True),
                                        headers=self._client._get_headers())
            else:
                response = requests.get(
                    self._client.service_instance._href_definitions.get_sw_spec_href(sw_spec_uid),
                    params=self._client._params(),
                    headers=self._client._get_headers())

            if response.status_code == 200:
                return self._get_required_element_from_response(
                    self._handle_response(200, u'get sw spec details', response))
            else:
                return self._handle_response(200, u'get sw spec details', response)
        else:
            if state_info:
                response = requests.get(
                    self._client.service_instance._href_definitions.get_software_specifications_list_href(),
                    params=self._client._params(),
                    headers=self._client._get_headers())

                return {'resources': self._handle_response(200, u'get sw specs details', response)['results']}
            else:
                response = requests.get(self._client.service_instance._href_definitions.get_sw_specs_href(),
                                        params=self._client._params(
                                            skip_space_project_chk=not self._ICP or self._client.ICP_PLATFORM_SPACES),
                                        headers=self._client._get_headers())

                return {'resources': [
                    self._get_required_element_from_response(x)
                    for x in self._handle_response(200, u'get sw specs details', response)['resources']]}

    def store(self, meta_props):
        """Create a software specification.

        :param meta_props: metadata of the space configuration. To see available meta names use:

            .. code-block:: python

                client.software_specifications.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored space
        :rtype: dict

        **Example**

        .. code-block:: python

            meta_props = {
                client.software_specifications.ConfigurationMetaNames.NAME: "skl_pipeline_heart_problem_prediction",
                client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "description scikit-learn_0.20",
                client.software_specifications.ConfigurationMetaNames.PACKAGE_EXTENSIONS_UID: [],
                client.software_specifications.ConfigurationMetaNames.SOFTWARE_CONFIGURATIONS: {},
                client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION_ID: "guid"
            }

            sw_spec_details = client.software_specifications.store(meta_props)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.WSD_20:
            raise WMLClientError(u'store() API is not supported in Watson Studio Desktop.')

        SwSpec._validate_type(meta_props, u'meta_props', dict, True)
        sw_spec_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        sw_spec_meta_json = json.dumps(sw_spec_meta)
        href = self._client.service_instance._href_definitions.get_sw_specs_href()

        creation_response = requests.post(href, params=self._client._params(), headers=self._client._get_headers(), data=sw_spec_meta_json)

        sw_spec_details = self._handle_response(201, u'creating sofware specifications', creation_response)

        return sw_spec_details

    def list(self, limit=None, return_as_df=True):
        """Print software specifications in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional
        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed software specifications or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.software_specifications.list()
        """

        if not self._client.WSD_20:
            href = self._client.service_instance._href_definitions.get_sw_specs_href()

            response = requests.get(href, params=self._client._params(), headers=self._client._get_headers())

            asset_details = self._handle_response(200, u'list assets', response)['resources']

            href = self._client.service_instance._href_definitions.get_software_specifications_list_href()

            response = requests.get(href, params=self._client._params(), headers=self._client._get_headers())

            spec_state_dict = {el['id']: el for el
                               in self._handle_response(200,
                                                        'list sw_specs with spec_state info', response)['results']}

            sw_spec_values = [
                (m['metadata']['name'], m['metadata']['asset_id'],
                 m['entity']['software_specification'].get('type', 'derived'),
                 spec_state_dict.get(m['metadata']['asset_id'], {}).get('state', 'not_provided'),
                 spec_state_dict.get(m['metadata']['asset_id'], {}).get('replacement', '')) for
                m in asset_details]

            table = self._list(sw_spec_values, ['NAME', 'ID', 'TYPE', 'STATE', 'REPLACEMENT'], limit, _DEFAULT_LIST_LENGTH)
            if return_as_df:
                return table
        else:
            from tabulate import tabulate
            header = ['NAME', 'ID', 'TYPE']
            print(tabulate(self.software_spec_list.items(), headers=header))

    @staticmethod
    def get_id(sw_spec_details):
        """Get Unique Id of software specification.

        :param sw_spec_details: metadata of the software specification
        :type sw_spec_details: dict

        :return: Unique Id of software specification
        :rtype: str

        **Example**

        .. code-block:: python

            asset_uid = client.software_specifications.get_id(sw_spec_details)

        """

        return SwSpec.get_uid(sw_spec_details)

    @staticmethod
    def get_uid(sw_spec_details):
        """Get Unique Id of software specification.

        *Deprecated:* Use ``get_id(sw_spec_details)`` instead.

        :param sw_spec_details: metadata of the software specification
        :type sw_spec_details: dict

        :return: Unique Id of software specification
        :rtype: str

        **Example**

        .. code-block:: python

            asset_uid = client.software_specifications.get_uid(sw_spec_details)

        """
        SwSpec._validate_type(sw_spec_details, u'sw_spec_details', object, True)
        SwSpec._validate_type_of_details(sw_spec_details, SW_SPEC_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(sw_spec_details, u'sw_spec_details',
                                                           [u'metadata', u'asset_id'])

    def get_id_by_name(self, sw_spec_name):
        """Get Unique Id of software specification.

        :param sw_spec_name: name of the software specification
        :type sw_spec_name: str

        :return: Unique Id of software specification
        :rtype: str

        **Example**

        .. code-block:: python

            asset_uid = client.software_specifications.get_id_by_name(sw_spec_name)

        """

        return SwSpec.get_uid_by_name(self, sw_spec_name)

    def get_uid_by_name(self, sw_spec_name):
        """Get Unique Id of software specification.

        *Deprecated:* Use ``get_id_by_name(self, sw_spec_name)`` instead.

        :param sw_spec_name: name of the software specification
        :type sw_spec_name: str

        :return: Unique Id of software specification
        :rtype: str

        **Example**

        .. code-block:: python

            asset_uid = client.software_specifications.get_uid_by_name(sw_spec_name)

        """

        SwSpec._validate_type(sw_spec_name, u'sw_spec_uid', str, True)
        if not self._client.WSD_20:
            parameters = self._client._params(skip_space_project_chk=True)
            parameters.update(name=sw_spec_name)

            response = requests.get(self._client.service_instance._href_definitions.get_sw_specs_href(),
                                    params=parameters,
                                    headers=self._client._get_headers())

            total_values = self._handle_response(200, u'list assets', response)["total_results"]
            if total_values != 0:
                sw_spec_details = self._handle_response(200, u'list assets', response)["resources"]
                return sw_spec_details[0][u'metadata'][u'asset_id']
            else:
                return "Not Found"
        else:
            return self.software_spec_list.get(sw_spec_name)

    @staticmethod
    def get_href(sw_spec_details):
        """Get url of software specification.

        :param sw_spec_details: software specification details
        :type sw_spec_details: dict

        :return: href of software specification
        :rtype: str

        **Example**

        .. code-block:: python

            sw_spec_details = client.software_specifications.get_details(sw_spec_uid)
            sw_spec_href = client.software_specifications.get_href(sw_spec_details)

        """
        SwSpec._validate_type(sw_spec_details, u'sw_spec_details', object, True)
        SwSpec._validate_type_of_details(sw_spec_details, SW_SPEC_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(sw_spec_details, u'sw_spec_details', [u'metadata', u'href'])

    def delete(self, sw_spec_uid):
        """Delete a software specification.

        :param sw_spec_uid: Unique Id of software specification
        :type sw_spec_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.software_specifications.delete(sw_spec_uid)

        """
        if self._client.WSD_20:
            raise WMLClientError(u'delete API is not supported in Watson Studio Desktop.')

        SwSpec._validate_type(sw_spec_uid, u'sw_spec_uid', str, True)

        response = requests.delete(self._client.service_instance._href_definitions.get_sw_spec_href(sw_spec_uid), params=self._client._params(),
                                headers=self._client._get_headers())

        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, u'delete software specification', response)

    def add_package_extension(self, sw_spec_uid, pkg_extn_id):
        """Add a package extension to software specifications existing metadata.

        :param sw_spec_uid: Unique Id of software specification which should be updated
        :type sw_spec_uid: str
        :param pkg_extn_id: Unique Id of package extension which should needs to added to software specification
        :type pkg_extn_id: str

        **Example**

        .. code-block:: python

            client.software_specifications.add_package_extension(sw_spec_uid, pkg_extn_id)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        if self._client.WSD_20:
            raise WMLClientError(u'package extension APIs are not supported in Watson Studio Desktop.')

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        self._validate_type(sw_spec_uid, u'sw_spec_uid', str, True)
        self._validate_type(pkg_extn_id, u'pkg_extn_id', str, True)

        url = self._client.service_instance._href_definitions.get_sw_spec_href(sw_spec_uid)

        url = url + "/package_extensions/" + pkg_extn_id

        response = requests.put(url, params=self._client._params(), headers=self._client._get_headers())

        if response.status_code == 204:
            print("SUCCESS")
            return "SUCCESS"
        else:
            return self._handle_response(204, u'pkg spec add', response, False)

    def delete_package_extension(self, sw_spec_uid, pkg_extn_id):
        """Delete a package extension from software specifications existing metadata.

        :param sw_spec_uid: Unique Id of software specification which should be updated
        :type sw_spec_uid: str
        :param pkg_extn_id: Unique Id of package extension which should needs to deleted from software specification
        :type pkg_extn_id: str

        **Example**

        .. code-block:: python

            client.software_specifications.delete_package_extension(sw_spec_uid, pkg_extn_id)

        """
        if self._client.WSD_20:
            raise WMLClientError(u'package extension APIs are not supported in Watson Studio Desktop.')

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        self._validate_type(sw_spec_uid, u'sw_spec_uid', str, True)
        self._validate_type(pkg_extn_id, u'pkg_extn_id', str, True)

        url = self._client.service_instance._href_definitions.get_sw_spec_href(sw_spec_uid)

        url = url + "/package_extensions/" + pkg_extn_id

        response = requests.delete(url,
                                   params=self._client._params(),
                                   headers=self._client._get_headers())

        return self._handle_response(204, u'pkg spec delete', response, False)

    def _get_required_element_from_response(self, response_data):

        WMLResource._validate_type(response_data, u'sw_spec_response', dict)
        try:
            if self._client.default_space_id is not None:
                new_el = {'metadata': {
                                       'name': response_data['metadata']['name'],
                                       'asset_id': response_data['metadata']['asset_id'],
                                       'href': response_data['metadata']['href'],
                                       'asset_type': response_data['metadata']['asset_type'],
                                       'created_at': response_data['metadata']['created_at']
                                       #'updated_at': response_data['metadata']['updated_at']
                                       },
                          'entity': response_data['entity']

                          }
            elif self._client.default_project_id is not None:
                if self._client.WSD:

                    href = self._client.service_instance._href_definitions.get_base_asset_href(response_data['metadata']['asset_id']) + "?" + "project_id=" + response_data['metadata']['project_id']

                    new_el = {'metadata': {
                                           'name': response_data['metadata']['name'],
                                           'asset_id': response_data['metadata']['asset_id'],
                                           'href': response_data['metadata']['href'],
                                           'asset_type': response_data['metadata']['asset_type'],
                                           'created_at': response_data['metadata']['created_at']
                                           },
                              'entity': response_data['entity']

                              }
                else:
                    new_el = {'metadata': {
                                           'name': response_data['metadata']['name'],
                                           'asset_id': response_data['metadata']['asset_id'],
                                           'href': response_data['metadata']['href'],
                                           'asset_type': response_data['metadata']['asset_type'],
                                           'created_at': response_data['metadata']['created_at']
                                       },
                             'entity': response_data['entity']

                            }

            else:
                # For system software spec
                new_el = {'metadata': {
                                       'name': response_data['metadata']['name'],
                                       'asset_id': response_data['metadata']['asset_id'],
                                       'href': response_data['metadata']['href'],
                                       'asset_type': response_data['metadata']['asset_type'],
                                       'created_at': response_data['metadata']['created_at']
                                       #'updated_at': response_data['metadata']['updated_at']
                                       },
                          'entity': response_data['entity']

                          }
            if (self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES) and 'href' in response_data['metadata']:
                href_without_host = response_data['metadata']['href'].split('.com')[-1]
                new_el[u'metadata'].update({'href': href_without_host})
            return new_el
        except Exception as e:
            raise WMLClientError("Failed to read Response from down-stream service: " + response_data.text)
