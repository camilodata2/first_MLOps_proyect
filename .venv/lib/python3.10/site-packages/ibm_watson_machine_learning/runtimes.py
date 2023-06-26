#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
requests.packages.urllib3.disable_warnings()
import json
import os
from ibm_watson_machine_learning.utils import INSTANCE_DETAILS_TYPE, RUNTIME_SPEC_DETAILS_TYPE, MODEL_DETAILS_TYPE, LIBRARY_DETAILS_TYPE, FUNCTION_DETAILS_TYPE, get_type_of_details
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from ibm_watson_machine_learning.href_definitions import is_uid
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.metanames import RuntimeMetaNames, LibraryMetaNames
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact import MLRepositoryArtifact
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps, MetaNames
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES,PIPELINES, LIBRARIES, EXPERIMENTS, RUNTIMES, DEPLOYMENTS


def LibraryDefinition(name, version, filepath, description=None, platform=None, model_definition=None, custom=None,
                      command=None, tags=None, space_uid=None):
    WMLResource._validate_type(name, 'name', str, True)
    WMLResource._validate_type(version, 'version', str, True)
    WMLResource._validate_type(platform, 'platform', dict, False)
    WMLResource._validate_type(description, 'description', str, False)
    WMLResource._validate_type(filepath, 'filepath', str, True)
    WMLResource._validate_type(model_definition, 'model_definition', bool, False)
    WMLResource._validate_type(custom, 'custom', dict, False)
    WMLResource._validate_type(command, 'command', str, False)
    WMLResource._validate_type(tags, 'tags', dict, False)
    WMLResource._validate_type(space_uid, 'space_uid', str, False)

    definition = {
        'name': name,
        'version': version,
        'filepath': filepath
    }

    if description is not None:
        definition['description'] = description

    if platform is not None:
        definition['platform'] = platform
    if model_definition is not None:
        definition['model_definition'] = model_definition
    if custom is not None:
        definition['custom'] = custom
    if command is not None:
        definition['command'] = command
    if tags is not None:
        definition['tags'] = tags
    if space_uid is not None:
        definition['space_uid'] = space_uid
    return definition


class Runtimes(WMLResource):
    """Create Runtime Specs and associated Custom Libraries.

    .. note::
        There are a list of pre-defined runtimes available. To see the list of pre-defined runtimes, use:

        .. code-block:: python

            client.runtimes.list(pre_defined=True)
    """
    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        if not client.ICP and not client.WSD and not client.CLOUD_PLATFORM_SPACES:
            Runtimes._validate_type(client.service_instance.details, u'instance_details', dict, True)
            Runtimes._validate_type_of_details(client.service_instance.details, INSTANCE_DETAILS_TYPE)
        self.ConfigurationMetaNames = RuntimeMetaNames()
        self.LibraryMetaNames = LibraryMetaNames()
        self._ICP = client.ICP

    def _create_library_from_definition(self, definition, runtime_definition):
        self._validate_meta_prop(definition, 'name', str, True)
        self._validate_meta_prop(definition, 'version', str, True)
        self._validate_meta_prop(definition, 'platform', dict, False)
        self._validate_meta_prop(definition, 'description', str, False)
        self._validate_meta_prop(definition, 'filepath', str, True)
        self._validate_meta_prop(definition, 'model_defintion', bool, False)
        self._validate_meta_prop(definition, 'custom', dict, False)
        self._validate_meta_prop(definition, 'command', str, False)
        self._validate_meta_prop(definition, 'tags', dict, False)
        self._validate_meta_prop(definition, 'space_uid', str, False)

        lib_metadata = {
            self.LibraryMetaNames.NAME: definition['name'],
            self.LibraryMetaNames.VERSION: definition['version'],
            self.LibraryMetaNames.PLATFORM:
                definition['platform']
                if 'platform' in definition and definition['platform'] is not None
                else {
                    "name": runtime_definition[self.ConfigurationMetaNames.PLATFORM]['name'],
                    "versions": [runtime_definition[self.ConfigurationMetaNames.PLATFORM]['version']]
                },
            self.LibraryMetaNames.FILEPATH: definition['filepath'],
            self.LibraryMetaNames.MODEL_DEFINITION: definition['model_definiton'],
            self.LibraryMetaNames.COMMAND: definition['command'],
            self.LibraryMetaNames.CUSTOM: definition['custom'],
            self.LibraryMetaNames.TAGS: definition['tags'],
            self.LibraryMetaNames.SPACE_UID: definition['space_uid'],
        }

        if 'description' in definition:
            lib_metadata[self.LibraryMetaNames.DESCRIPTION] = definition['description']
        if 'tags' in definition:
            lib_metadata[self.LibraryMetaNames.TAGS] = definition['tags']
        if 'model_definition' in definition:
            lib_metadata[self.LibraryMetaNames.MODEL_DEFINITION] = definition['model_definition']
        if 'custom' in definition:
            lib_metadata[self.LibraryMetaNames.CUSTOM] = definition['custom']
        if 'command' in definition:
            lib_metadata[self.LibraryMetaNames.COMMAND] = definition['command']
        if 'space_uid' in definition:
            lib_metadata[self.LibraryMetaNames.SPACE_UID] = definition['space_uid']

        details = self.store_library(lib_metadata)
        return self.get_library_uid(details)

    def store_library(self, meta_props):
        """Create a library.

        :param meta_props:  metadata of the library configuration. To see available meta names use:

            .. code-block:: python

                client.runtimes.LibraryMetaNames.get()

        :type meta_props: dict
        :return: metadata of the library created
        :rtype: dict

        **Example**

        .. code-block:: python

            library_details = client.runtimes.store_library({
                client.runtimes.LibraryMetaNames.NAME: "libraries_custom",
                client.runtimes.LibraryMetaNames.DESCRIPTION: "custom libraries for scoring",
                client.runtimes.LibraryMetaNames.FILEPATH: custom_library_path,
                client.runtimes.LibraryMetaNames.VERSION: "1.0",
                client.runtimes.LibraryMetaNames.PLATFORM: {"name": "python", "versions": ["3.5"]}
            })
        """
        self.LibraryMetaNames._validate(meta_props)

        lib_metadata = self.LibraryMetaNames._generate_resource_metadata(meta_props, with_validation=True)
        if self._client.CAMS:
            if self._client.default_space_id is not None:
                lib_metadata['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
            else:
                raise WMLClientError(
                    "It is mandatory is set the space. Use client.set.default_space(<SPACE_GUID>) to set the space.")

        try:
            response_post = requests.post(self._client.service_instance._href_definitions.get_custom_libraries_href(), json=lib_metadata,
                                          headers=self._client._get_headers())

            details = self._handle_response(201, u'saving libraries', response_post)

            if self.LibraryMetaNames.FILEPATH in meta_props:
                try:
                    base_url = self._wml_credentials[u'url']
                    libraries_content_url = base_url + details[u'metadata'][u'href'] + '/content'

                    put_header = self._client._get_headers(no_content_type=True)
                    with open(meta_props[self.LibraryMetaNames.FILEPATH], 'rb') as data:
                        response_definition_put = requests.put(libraries_content_url, data=data, headers=put_header)

                except Exception as e:
                    raise e
                self._handle_response(200, u'saving libraries content', response_definition_put, False)
        except Exception as e:
            raise WMLClientError('Failure during creation of libraries.', e)
        return details

    def _create_runtime_spec(self, custom_libs_list, meta_props):

        metadata = {
            self.ConfigurationMetaNames.NAME : meta_props[self.ConfigurationMetaNames.NAME],
            self.ConfigurationMetaNames.PLATFORM: meta_props[self.ConfigurationMetaNames.PLATFORM],
        }

        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            metadata[self.ConfigurationMetaNames.DESCRIPTION] = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

        if self.ConfigurationMetaNames.CUSTOM in meta_props:
            metadata[self.ConfigurationMetaNames.CUSTOM] = {meta_props[self.ConfigurationMetaNames.CUSTOM]}

        if self.ConfigurationMetaNames.COMPUTE in meta_props:
            metadata[self.ConfigurationMetaNames.COMPUTE] = meta_props[self.ConfigurationMetaNames.COMPUTE]

        if self.ConfigurationMetaNames.SPACE_UID in meta_props:
            metadata[self.ConfigurationMetaNames.SPACE_UID] = {
                "href": "/v4/spaces/"+meta_props[self.ConfigurationMetaNames.SPACE_UID]
            }
        if self._client.CAMS:
            if self._client.default_space_id is not None:
                metadata['space'] = {'href': "/v4/spaces/" + self._client.default_space_id}
            else:
                raise WMLClientError(
                    "It is mandatory to set the space. Use client.set.default_space(<SPACE_UID>) to proceed.")

        if custom_libs_list is not None:
            custom_list = []
            for uid in custom_libs_list:
                each_href = {"href": "/v4/libraries/" + uid}
                custom_list.append(each_href)
            metadata["custom_libraries"] = custom_list

        if self.ConfigurationMetaNames.CONFIGURATION_FILEPATH in meta_props:
            metadata[MetaNames.CONTENT_LOCATION] = meta_props[self.ConfigurationMetaNames.CONFIGURATION_FILEPATH]

        try:
            response_post = requests.post(self._client.service_instance._href_definitions.get_runtimes_href(), json=metadata,
                                          headers=self._client._get_headers())

            details = self._handle_response(201, u'saving runtimes', response_post)
            if self.ConfigurationMetaNames.CONFIGURATION_FILEPATH in meta_props:
                try:
                    runtimes_content_url = self._wml_credentials[u'url'] + details[u'metadata'][u'href'] + '/content'

                    put_header = self._client._get_headers(content_type="text/plain")
                    with open(meta_props[self.ConfigurationMetaNames.CONFIGURATION_FILEPATH], 'rb') as data:
                        response_definition_put = requests.put(runtimes_content_url, data=data,headers=put_header)

                except Exception as e:
                    raise e
                self._handle_response(200, u'saving runtimes content', response_definition_put, False)
        except Exception as e:
            raise WMLClientError('Failure during creation of runtime.', e)
        return details['metadata']['guid']

    def store(self, meta_props):
        """Create a runtime.

        :param meta_props:  metadata of the runtime configuration. To see available meta names use:

            .. code-block:: python

                client.runtimes.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the runtime created
        :rtype: dict

        **Examples**

        Creating a library:

        .. code-block:: python

            lib_meta = {
                client.runtimes.LibraryMetaNames.NAME: "libraries_custom",
                client.runtimes.LibraryMetaNames.DESCRIPTION: "custom libraries for scoring",
                client.runtimes.LibraryMetaNames.FILEPATH: "/home/user/my_lib.zip",
                client.runtimes.LibraryMetaNames.VERSION: "1.0",
                client.runtimes.LibraryMetaNames.PLATFORM: {"name": "python", "versions": ["3.5"]}
            }

            custom_library_details = client.runtimes.store_library(lib_meta)
            custom_library_uid = client.runtimes.get_library_uid(custom_library_details)

        Creating a runtime:

        .. code-block:: python

            runtime_meta = {
                client.runtimes.ConfigurationMetaNames.NAME: "runtime_spec_python_3.5",
                client.runtimes.ConfigurationMetaNames.DESCRIPTION: "test",
                client.runtimes.ConfigurationMetaNames.PLATFORM: {
                    "name": "python",
                    "version": "3.5"
                },
                client.runtimes.ConfigurationMetaNames.LIBRARIES_UIDS: [custom_library_uid] # already existing lib is linked here
            }

            runtime_details = client.runtimes.store(runtime_meta)

        """

        WMLResource._chk_and_block_create_update_for_python36(self)

        self.ConfigurationMetaNames._validate(meta_props)

        custom_libs_list = []

        # if self.ConfigurationMetaNames.LIBRARIES_DEFINITIONS in meta_props:
        #     custom_libs_list.extend(
        #         [self._create_library_from_definition(definition, meta_props) for definition in
        #          meta_props[self.ConfigurationMetaNames.LIBRARIES_DEFINITIONS]]
        #     )

        if self.ConfigurationMetaNames.LIBRARIES_UIDS in meta_props:
            custom_libs_list.extend(meta_props[self.ConfigurationMetaNames.LIBRARIES_UIDS])

        runtime_uid = self._create_runtime_spec(custom_libs_list, meta_props)

        return self.get_details(runtime_uid)

    def get_details(self, runtime_uid=None, pre_defined=False, limit=None):
        """Get metadata of stored runtime(s). If runtime UID is not specified returns all runtimes metadata.

        :param runtime_uid: runtime UID
        :type runtime_uid: str, optional
        :param pre_defined: boolean indicating to display predefined runtimes only
        :type pre_defined: bool, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of runtime(s)
        :rtype:
            - **dict** - if runtime_uid is not None
            - **{"resources": [dict]}** - if runtime_uid is None

        **Examples**

        .. code-block:: python

            runtime_details = client.runtimes.get_details(runtime_uid)
            runtime_details = client.runtimes.get_details(runtime_uid=runtime_uid)
            runtime_details = client.runtimes.get_details()
        """
        Runtimes._validate_type(runtime_uid, u'runtime_uid', str, False)

        # if runtime_uid is not None and not is_uid(runtime_uid):
        #     raise WMLClientError(u'\'runtime_uid\' is not an uid: \'{}\''.format(runtime_uid))

        url = self._client.service_instance._href_definitions.get_runtimes_href()
        if runtime_uid is not None or self._client.default_project_id is not None:
         return self._get_no_space_artifact_details(url, runtime_uid, limit, 'runtime specs', pre_defined="True")
        if pre_defined:
         return self._get_artifact_details(url, runtime_uid, limit, 'runtime specs',pre_defined="True")
        else:
         return self._get_artifact_details(url, runtime_uid, limit, 'runtime specs')

    def get_library_details(self, library_uid=None, limit=None):
        """Get metadata of stored librarie(s). If library UID is not specified returns all libraries metadata.

        :param library_uid: library UID
        :type library_uid: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of library(s)
        :rtype:
            - **dict** - if runtime_uid is not None
            - **{"resources": [dict]}** - if runtime_uid is None

        **Examples**

        .. code-block:: python

            library_details = client.runtimes.get_library_details(library_uid)
            library_details = client.runtimes.get_library_details(library_uid=library_uid)
            library_details = client.runtimes.get_library_details()
        """
        Runtimes._validate_type(library_uid, u'library_uid', str, False)

        if library_uid is not None and not is_uid(library_uid):
            raise WMLClientError(u'\'library_uid\' is not an uid: \'{}\''.format(library_uid))

        url = self._client.service_instance._href_definitions.get_custom_libraries_href()
        if library_uid is not None or self._client.default_project_id is not None:
            return self._get_no_space_artifact_details(url, library_uid, limit, 'libraries')
        return self._get_artifact_details(url, library_uid, limit, 'libraries')

    @staticmethod
    def get_href(details):
        """Get runtime href from runtime details.

        :param details: metadata of the runtime
        :type details: dict

        :return: runtime href
        :rtype: str

        **Example**

        .. code-block:: python

            runtime_details = client.runtimes.get_details(runtime_uid)
            runtime_href = client.runtimes.get_href(runtime_details)
        """

        Runtimes._validate_type(details, u'details', dict, True)
        Runtimes._validate_type_of_details(details, [RUNTIME_SPEC_DETAILS_TYPE, MODEL_DETAILS_TYPE, FUNCTION_DETAILS_TYPE])

        details_type = get_type_of_details(details)

        if details_type == RUNTIME_SPEC_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'runtime_details', ['metadata', 'href'])
        elif details_type == MODEL_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'model_details', ['entity', 'runtime', 'href'])
        elif details_type == FUNCTION_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'function_details', ['entity', 'runtime', 'href'])
        else:
            raise WMLClientError('Unexpected details type: {}'.format(details_type))

    @staticmethod
    def get_uid(details):
        """Get runtime uid from runtime details.

        :param details: metadata of the runtime
        :type details: dict

        :return: runtime UID
        :rtype: str

        **Example**

        .. code-block:: python

            runtime_details = client.runtimes.get_details(runtime_uid)
            runtime_uid = client.runtimes.get_uid(runtime_details)
        """

        Runtimes._validate_type(details, u'details', dict, True)
        Runtimes._validate_type_of_details(details, [RUNTIME_SPEC_DETAILS_TYPE, MODEL_DETAILS_TYPE, FUNCTION_DETAILS_TYPE])

        details_type = get_type_of_details(details)

        if details_type == RUNTIME_SPEC_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'runtime_details', ['metadata', 'guid'])
        elif details_type == MODEL_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'model_details', ['entity', 'runtime', 'href']).split('/')[-1]
        elif details_type == FUNCTION_DETAILS_TYPE:
            return Runtimes._get_required_element_from_dict(details, 'function_details', ['entity', 'runtime', 'href']).split('/')[-1]
        else:
            raise WMLClientError('Unexpected details type: {}'.format(details_type))

    @staticmethod
    def get_library_href(library_details):
        """Get library href from library details.

        :param library_details: metadata of the library
        :type library_details: dict

        :return: library href
        :rtype: str

        **Example**

        .. code-block:: python

            library_details = client.runtimes.get_library_details(library_uid)
            library_url = client.runtimes.get_library_href(library_details)
        """

        Runtimes._validate_type(library_details, u'library_details', dict, True)
        Runtimes._validate_type_of_details(library_details, LIBRARY_DETAILS_TYPE)

        return Runtimes._get_required_element_from_dict(library_details, 'library_details', ['metadata', 'href'])

    @staticmethod
    def get_library_uid(library_details):
        """Get library uid from library details.

        :param library_details: metadata of the library
        :type library_details: dict

        :return: library UID
        :rtype: str

        **Example**

        .. code-block:: python

            library_details = client.runtimes.get_library_details(library_uid)
            library_uid = client.runtimes.get_library_uid(library_details)
        """

        Runtimes._validate_type(library_details, u'library_details', dict, True)
        Runtimes._validate_type_of_details(library_details, LIBRARY_DETAILS_TYPE)

        # TODO error handling
        return Runtimes._get_required_element_from_dict(library_details, 'library_details', ['metadata', 'guid'])

    def _get_runtimes_uids_for_lib(self, library_uid, runtime_details=None):
        # Return list of runtimes which contains library_uid that is passed.
        if runtime_details is None:
            runtime_details = self.get_details()

        return list(map(
            lambda x: x['metadata']['guid'],
            filter(
                lambda x: any(
                    filter(
                        lambda y: library_uid in y['href'],
                        x['entity']['custom_libraries'] if 'custom_libraries' in x['entity'] else [])
                ),
                runtime_details['resources']
            )
        ))

    def delete(self, runtime_uid, with_libraries=False):
        """Delete a runtime.

        :param runtime_uid: runtime UID
        :type runtime_uid: str
        :param with_libraries: boolean value indicating an option to delete the libraries associated with the runtime
        :type with_libraries: bool, optional

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.runtimes.delete(deployment_uid)
        """
        Runtimes._validate_type(runtime_uid, u'runtime_uid', str, True)
        Runtimes._validate_type(with_libraries, u'autoremove', bool, True)

        if runtime_uid is not None and not is_uid(runtime_uid):
            raise WMLClientError(u'\'runtime_uid\' is not an uid: \'{}\''.format(runtime_uid))

        if with_libraries:
            runtime_details = self.get_details(runtime_uid)

        url = self._client.service_instance._href_definitions.get_runtime_href(runtime_uid)

        response_delete = requests.delete(
            url,
            headers=self._client._get_headers())


        if with_libraries:
            if 'custom_libraries' in runtime_details['entity']:
                details = self.get_details()
                custom_libs_uids = map(lambda x: x['href'].split('/')[-1], runtime_details['entity']['custom_libraries'])
                custom_libs_to_remove = filter(
                    lambda x: len(self._get_runtimes_uids_for_lib(x, details)) == 0,
                    custom_libs_uids
                )

                for uid in custom_libs_to_remove:
                    print('Deleting orphaned library \'{}\' during autoremove delete.'.format(uid))
                    delete_status = self.delete_library(uid)
                    print(delete_status)
        return self._handle_response(204, u'runtime deletion', response_delete, False)

    def _delete_orphaned_libraries(self):
        """Delete all custom libraries without runtime.

        **Example**

        .. code-block:: python

            client.runtimes.delete_orphaned_libraries()
        """
        lib_details = self.get_library_details()
        details = self.get_details()
        for lib in lib_details['resources']:
            lib_uid = lib['metadata']['guid']
            if len(self._get_runtimes_uids_for_lib(lib_uid, details)) == 0:
                print('Deleting orphaned \'{}\' library... '.format(lib_uid), end="")
                library_endpoint = self._client.service_instance._href_definitions.get_custom_library_href(lib_uid)
                response_delete = requests.delete(library_endpoint, headers=self._client._get_headers())

                try:
                    self._handle_response(204, u'library deletion', response_delete, False)
                    print('SUCCESS')
                except:
                    pass

    def delete_library(self, library_uid):
        """Delete a library.

        :param library_uid: library UID
        :type library_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.runtimes.delete_library(library_uid)
        """
        Runtimes._validate_type(library_uid, u'library_uid', str, True)
        library_endpoint = self._client.service_instance._href_definitions.get_custom_library_href(library_uid)

        response_delete = requests.delete(library_endpoint, headers=self._client._get_headers())

        return self._handle_response(204, u'library deletion', response_delete, False)

    def list(self, limit=None, pre_defined=False):
        """Print stored runtimes in a table format. If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional
        :param pre_defined: boolean indicating to display predefined runtimes only
        :type pre_defined: bool, optional

        **Example**

        .. code-block:: python

            client.runtimes.list()
            client.runtimes.list(pre_defined=True)
        """
        details = self.get_details(pre_defined=pre_defined)
        resources = details[u'resources']
        values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'metadata'][u'created_at'], m[u'entity'][u'platform']['name']) for m in resources]

        self._list(values, [u'GUID', u'NAME', u'CREATED', u'PLATFORM'], limit, 50)

    def _list_runtimes_for_libraries(self): # TODO make public when the time'll come
        """Print runtimes uids for libraries in a table format.

        **Example**

        .. code-block:: python

            client.runtimes.list_runtimes_for_libraries()
            client.runtimes.list_runtimes_for_libraries(library_uid)
        """
        details = self.get_library_details()
        runtime_details = self.get_details()

        values = [
            (m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'version'],
             ', '.join(self._get_runtimes_uids_for_lib(m[u'metadata'][u'guid'], runtime_details))) for m in
            details['resources']]

        values = sorted(sorted(values, key=lambda x: x[2], reverse=True), key=lambda x: x[1])

        from tabulate import tabulate

        header = [u'GUID', u'NAME', u'VERSION', u'RUNTIME SPECS']
        table = tabulate([header] + values)

        print(table)

    def list_libraries(self, runtime_uid=None, limit=None):
        """Print stored libraries in a table format. If runtime UID is not provided, all libraries are listed else,
        libraries associated with a runtime are listed. If limit is set to None there will be only first 50 records
        shown.

        :param runtime_uid: runtime UID
        :type runtime_uid: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        **Example**

        .. code-block:: python

            client.runtimes.list_libraries()
            client.runtimes.list_libraries(runtime_uid)
        """
        Runtimes._validate_type(runtime_uid, u'runtime_uid', str, False)

        if runtime_uid is None:
            details = self.get_library_details()

            resources = details[u'resources']
            values = [(m[u'metadata'][u'guid'], m[u'entity'][u'name'], m[u'entity'][u'version'], m[u'metadata'][u'created_at'],
                       m[u'entity'][u'platform']['name'], m[u'entity'][u'platform'][u'versions']) for m in
                      resources]

            self._list(values, [u'GUID', u'NAME', u'VERSION', u'CREATED', u'PLATFORM NAME', u'PLATFORM VERSIONS'], limit, 50)
        else:
            details = self.get_details(runtime_uid)

            if 'custom_libraries' not in details['entity'] or len(details['entity']['custom_libraries']) == 0:
                print('No libraries found for this runtime.')
                return

            values = [(m[u'href'].split('/')[-1], u'') for m in details['entity']['custom_libraries']]


            from tabulate import tabulate

            header = [u'GUID']
            table = tabulate([header] + values)

            print(table)

    def download_configuration(self, runtime_uid, filename='runtime_configuration.yaml'):
        """Downloads configuration file for runtime with specified uid.

        :param runtime_uid: UID of runtime
        :type runtime_uid: str
        :param filename: filename of downloaded archive
        :type filename: str, optional

        :return: path to the downloaded runtime configuration
        :rtype: str

        **Example**

        .. code-block:: python

            filename="runtime.yml"
            client.runtimes.download_configuration(runtime_uid, filename=filename)
        """
        Runtimes._validate_type(runtime_uid, u'runtime_uid', str, True)

        if not is_uid(runtime_uid):
            raise WMLClientError(u'\'runtime_uid\' is not an uid: \'{}\''.format(runtime_uid))

        download_url = self._client.service_instance._href_definitions.get_runtime_href(runtime_uid) + '/content'

        response_get = requests.get(
            download_url,
            headers=self._client._get_headers())

        if response_get.status_code == 200:
            with open(filename, "wb") as new_file:
                new_file.write(response_get.content)
                new_file.close()

                print(u'Successfully downloaded runtime configuration file: ' + str(filename))
                return os.getcwd() + "/"+filename
        else:
            if response_get.status_code == 404 and "content_does_not_exist" in str(response_get.text):
                raise WMLClientError(u'Unable to download configuration. download configruation can be invoked '
                                     u'only when CONFIGURATION_FILEPATH meta prop is set during store. ')
            raise WMLClientError(u'Unable to download configuration content: ' + response_get.text)

    def download_library(self, library_uid, filename=None):
        """Downloads library content with specified uid.

        :param library_uid: UID of library
        :type library_uid: str
        :param filename: filename of downloaded archive, default value: `<LIBRARY-NAME>-<LIBRARY-VERSION>.zip`
        :type filename: str, optional

        :return: path to the downloaded library content
        :rtype: str

        **Example**

        .. code-block:: python

            filename="library.tgz"
            client.runtimes.download_library(runtime_uid, filename=filename)
        """
        Runtimes._validate_type(library_uid, u'library_uid', str, True)

        if not is_uid(library_uid):
            raise WMLClientError(u'\'library_uid\' is not an uid: \'{}\''.format(library_uid))

        download_url = self._client.service_instance._href_definitions.get_custom_library_href(library_uid) + '/content'

        response_get = requests.get(
            download_url,
            headers=self._client._get_headers())

        if filename is None:
            details = self.get_library_details(library_uid)
            filename = '{}-{}.zip'.format(details['entity']['name'], details['entity']['version'])

        if response_get.status_code == 200:
            with open(filename, "wb") as new_file:
                new_file.write(response_get.content)
                new_file.close()

                print(u'Successfully downloaded library content: ' + str(filename))
                return os.getcwd() + "/"+filename
        else:
            raise WMLClientError(u'Unable to download library content: ' + response_get.text)

    def update_library(self, library_uid, changes):
        """Updates existing library metadata.

        :param library_uid: UID of library which definition should be updated
        :type library_uid: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated library
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.runtimes.LibraryMetaNames.NAME: "updated_lib"
            }

            library_details = client.runtimes.update_library(library_uid, changes=metadata)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        self._validate_type(library_uid, u'library_uid', str, True)
        self._validate_type(changes, u'changes', dict, True)

        details = self.get_library_details(library_uid)

        patch_payload = self.LibraryMetaNames._generate_patch_payload(details['entity'], changes, with_validation=True)

        url = self._client.service_instance._href_definitions.get_custom_library_href(library_uid)

        response = requests.patch(url, json=patch_payload, headers=self._client._get_headers())

        updated_details = self._handle_response(200, u'library patch', response)

        return updated_details

    def update_runtime(self, runtime_uid, changes):
        """Updates existing runtime metadata.

        :param runtime_uid: UID of runtime which definition should be updated
        :type runtime_uid: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated runtime
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.runtimes.ConfigurationMetaNames.NAME: "updated_runtime"
            }

            runtime_details = client.runtimes.update(runtime_uid, changes=metadata)
        """

        WMLResource._chk_and_block_create_update_for_python36(self)
        self._validate_type(runtime_uid, u'runtime_uid', str, True)
        self._validate_type(changes, u'changes', dict, True)

        details = self.get_details(runtime_uid)

        patch_payload = self.LibraryMetaNames._generate_patch_payload(details['entity'], changes, with_validation=True)

        url = self._client.service_instance._href_definitions.get_runtime_href(runtime_uid)

        response = requests.patch(url, json=patch_payload, headers=self._client._get_headers())

        updated_details = self._handle_response(200, u'library patch', response)

        return updated_details

    def clone_runtime(self, runtime_uid, space_id=None, action="copy", rev_id=None):
        """Create a new runtime identical with the given runtime either in the same space or in a new space.
        All dependent assets will be cloned too.

        :param runtime_uid: UID of the runtime to be cloned
        :type runtime_uid: str
        :param space_id: UID of the space to which the runtime needs to be cloned
        :type space_id: str, optional
        :param action: action specifying "copy" or "move"
        :type action: str, optional
        :param rev_id: revision ID of the runtime
        :type rev_id: str, optional

        :return: metadata of the runtime cloned
        :rtype**: dict

        .. note::

            * If revision id is not specified, all revisions of the artifact are cloned.

            * Space guid is mandatory for "move" action.

        **Example**

        .. code-block:: python

            client.runtimes.clone_runtime(runtime_uid=artifact_id,space_id=space_uid,action="copy")

        """
        Runtimes._validate_type(runtime_uid, 'runtime_uid', str, True)
        clone_meta = {}
        if space_id is not None:
            clone_meta["space"] = {"href": API_VERSION + SPACES + "/" + space_id}
        if action is not None:
            clone_meta["action"] = action
        if rev_id is not None:
            clone_meta["rev"] = rev_id

        url = self._client.service_instance._href_definitions.get_runtime_href(runtime_uid)

        response_post = requests.post(url, json=clone_meta,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=200, operationName=u'cloning runtime',
                                            response=response_post)

        return details

    def clone_library(self, library_uid, space_id=None, action="copy", rev_id=None):
        """Create a new function library with the given library either in the same space or in a new space.
        All dependent assets will be cloned too.

        :param library_uid: UID of the library to be cloned
        :type library_uid: str
        :param space_id: UID of the space to which the library needs to be cloned
        :type space_id: str, optional
        :param action: action specifying "copy" or "move"
        :type action: str, optional
        :param rev_id: revision ID of the library
        :type rev_id: str, optional

        :return: metadata of the library cloned
        :rtype: dict

        .. note::

            * If revision id is not specified, all revisions of the artifact are cloned.

            * Space guid is mandatory for "move" action.

        **Example**

        .. code-block:: python

            client.runtimes.clone_library(library_uid=artifact_id, space_id=space_uid, action="copy")

        """
        Runtimes._validate_type(library_uid, 'library_uid', str, True)
        clone_meta = {}
        if space_id is not None:
            clone_meta["space"] = {"href": API_VERSION + SPACES + "/" + space_id}
        if action is not None:
            clone_meta["action"] = action
        if rev_id is not None:
            clone_meta["rev"] = rev_id

        url = self._client.service_instance._href_definitions.get_custom_library_href(library_uid)

        response_post = requests.post(url, json=clone_meta,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=200, operationName=u'cloning library',
                                            response=response_post)

        return details


