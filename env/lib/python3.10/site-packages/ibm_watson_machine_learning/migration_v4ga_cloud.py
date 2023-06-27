#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.metanames import Migrationv4GACloudMetaNames
import os
import json
from ibm_watson_machine_learning.wml_client_error import WMLClientError, UnexpectedType, ApiRequestFailure
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES
_DEFAULT_LIST_LENGTH = 50


class Migrationv4GACloud(WMLResource):
    """Migration APIs for v4 GA Cloud. This will be applicable only till the migration period.
    Refer to the documentation at 'https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/wml-ai.html'
    for details on migration.
    """

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

        self._client = client
        self.ConfigurationMetaNames = Migrationv4GACloudMetaNames()
        self.skip_msg = False
        self.notification_msg = "NOTE: This migration API is only available during the migration period for moving "\
        "assets to a project or space for use with new machine learning service plans and v4 APIs on Cloud. Refer to "\
        "the documentation at 'https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/wml-ai.html' for "\
        "details on new plans and features.\n"

    def start(self, meta_props):
        """Migration APIs for v4 GA Cloud to migrate assets from v3 or v4 beta. This will be applicable only till the
        migration period. Refer to the documentation at
        'https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/wml-ai.html' for details on migration.

        You will need to have a space or project created before you start migration of assets to that space or project.
        You should have read access to the instance from which migration is required.
        You should have 'editor' or 'admin' access on the target 'space'/'project'

        SKIP_MIGRATED_ASSETS meta prop: If this is `True` (the default) and if the target assets still exist and if
        there are completed jobs (that were not deleted) then any assets that were already migrated will be skipped
        and the details will be returned in the skipped collection in the response. If this is `False` then it is
        possible that duplicate assets will be created in the target space or project if the asset was already migrated.

        :param meta_props: meta data, to see available meta names use
            ``client.v4ga_cloud_migration.ConfigurationMetaNames.get()`
        :type meta_props: str or dict

        :return: initial state of migration
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.v4ga_cloud_migration.MigrationMetaNames.DESCRIPTION: "Migration of assets from v3 to v4ga",
                client.v4ga_cloud_migration.MigrationMetaNames.OLD_INSTANCE_ID: "df40cf1-252f-424b-b52d-5cdd98143aec",
                client.v4ga_cloud_migration.MigrationMetaNames.SPACE_ID: "3fc54cf1-252f-424b-b52d-5cdd9814987f",
                client.v4ga_cloud_migration.MigrationMetaNames.FUNCTION_IDS: ["all"],
                client.v4ga_cloud_migration.MigrationMetaNames.MODEL_IDS: ["afaecb4-254f-689f-4548-9b4298243291"],
                client.v4ga_cloud_migration.MigrationMetaNames.MAPPING: {"dfaecf1-252f-424b-b52d-5cdd98143481": "4fbc211-252f-424b-b52d-5cdd98df310a"}
                client.v4ga_cloud_migration.MigrationMetaNames.SKIP_MIGRATED_ASSETS: True
            }
            details = client.v4ga_cloud_migration.start(meta_props=metadata)
        """

        print(self.notification_msg)

        Migrationv4GACloud._validate_type(meta_props, u'meta_props', dict, True)

        self._validate_input(meta_props)

        migration_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props,
            with_validation=True,
            client=self._client)

        href = self._client.service_instance._href_definitions.v4ga_cloud_migration_href()

        start_response = requests.post(href,
                                       headers=self._client._get_headers(),
                                       json=migration_meta)

        details = self._handle_response(expected_status_code=202,
                                        operationName=u'start migration',
                                        response=start_response)

        print("Migration has been started. The initial state information is below. To monitor the migration "
              "job, you can use client.v4ga_cloud_migration.get_details api where you can provide "
              "the 'migration_id' and 'space_id'/'project_id' to monitor the status of migration. "
              "If you see 'skipped' section in the output from get_details() api, its because the assets listed "
              "in 'skipped' section would already be migrated. Use 'SKIP_MIGRATED_ASSETS' to False in start() "
              "api( by default, its True ) in case you want the assets to be re-migrated again")

        return details

    def _validate_input(self, meta_props):
        if 'old_instance_id' not in meta_props:
            raise WMLClientError("Its mandatory to provide old instance id for migration. Provide it via meta_props "
                                 "Eg: 'client.v4ga_cloud_migration.MigrationMetaNames.OLD_INSTANCE_ID' ")

        if 'space_id' not in meta_props and 'project_id' not in meta_props:
            raise WMLClientError("Its mandatory to provide space_id or project_id for migration. Provide it "
                                 "via meta_props Eg: 'client.v4ga_cloud_migration.MigrationMetaNames.SPACE_ID' ")

        if 'model_ids' not in meta_props and 'function_ids' not in meta_props and\
           'experiment_ids' not in meta_props and 'pipeline_ids' not in meta_props:
            raise WMLClientError("Its mandatory to provide at least one of 'model_ids', 'function_ids',"
                                 " 'experiment_ids', 'pipeline_ids' ' for migration. Provide it via meta_props. Eg: "
                                 "'client.v4ga_cloud_migration.MigrationMetaNames.MODEL_IDS: ['afaecb4-254f-689f-4548-9b4298243291'] ")

        if 'model_ids' in meta_props and not meta_props[u'model_ids']:
            raise WMLClientError("MODEL_IDS is provided but empty. Provide model id(s) to be migrated or 'all' to "
                                 "migrate all models")

        if 'function_ids' in meta_props and not meta_props[u'function_ids']:
            raise WMLClientError("FUNCTION_IDS is provided but empty. Provide function id(s) to be migrated or 'all'"
                                 " to migrate all functions")

        if 'experiment_ids' in meta_props and not meta_props[u'experiment_ids']:
            raise WMLClientError("EXPERIMENT_IDS is provided but empty. Provide experiment id(s) to be migrated or "
                                 "'all' to migrate all experiments")

        if 'pipeline_ids' in meta_props and not meta_props[u'pipeline_ids']:
            raise WMLClientError("PIPELINE_IDS is provided but empty. Provide pipeline id(s) to be migrated or "
                                 "'all' to migrate all pipelines")

    def cancel(self, migration_id, space_id=None, project_id=None):
        """Cancel a migration job. `space_id` or `project_id` has to be provided.

        .. note::
            To delete a migration job, use delete() api.

        :param migration_id: migration identifier
        :type migration_id: str
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.v4ga_cloud_migration.cancel(migration_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                                               space_id='3421cf1-252f-424b-b52d-5cdd981495fe')
        """

        print(self.notification_msg)

        Migrationv4GACloud._validate_type(migration_id, u'migration_id', str, True)

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        href = self._client.service_instance._href_definitions.v4ga_cloud_migration_id_href(migration_id)

        params = {}

        if space_id is not None:
            params.update({'space_id': space_id})
        else:
            params.update({'project_id': project_id})

        cancel_response = requests.delete(href,
                                          params=params,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'cancel migration',
                                        response=cancel_response)

        if "SUCCESS" == details:
            print("Migration job cancelled")

    def delete(self, migration_id, space_id=None, project_id=None):
        """Deletes a migration job. `space_id` or `project_id` has to be provided.

        :param migration_id: migration identifier
        :type migration_id: str
        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.v4ga_cloud_migration.delete(migration_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                                               space_id='3421cf1-252f-424b-b52d-5cdd981495fe')
        """
        print(self.notification_msg)

        Migrationv4GACloud._validate_type(migration_id, u'migration_id', str, True)

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        href = self._client.service_instance._href_definitions.v4ga_cloud_migration_id_href(migration_id)

        params = {'hard_delete': True}

        if space_id is not None:
            params.update({'space_id': space_id})
        else:
            params.update({'project_id': project_id})

        delete_response = requests.delete(href,
                                          params=params,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=204,
                                        operationName=u'delete migration',
                                        response=delete_response)

        if "SUCCESS" == details:
            print("Migration job deleted")

    def get_details(self, migration_id=None, space_id=None, project_id=None, limit=None, asynchronous=False, get_all=False):
        """Get metadata of the migration job function(s). If no `migration_id` is specified all migration jobs
        metadata is returned.

        :param migration_id: migration identifier
        :type migration_id: str, optional
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

        :return: migration(s) metadata
        :rtype: dict (if migration_id is not None) or {"resources": [dict]} (if migration_id is None)\n

        **Example**

        .. code-block:: python

            migration_details = client.v4ga_cloud_migration.get_details(migration_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                                                                        space_id='3421cf1-252f-424b-b52d-5cdd981495fe')
            migration_details = client.v4ga_cloud_migration.get_details(limit=100)
            migration_details = client.v4ga_cloud_migration.get_details(limit=100, get_all=True)
            migration_details = []
            for entry in client.v4ga_cloud_migration.get_details(limit=100, asynchronous=True, get_all=True):
                migration_details.extend(entry)
        """

        if self.skip_msg is False:
            print(self.notification_msg)

        Migrationv4GACloud._validate_type(migration_id, u'migration_id', str, False)

        query_params = {}

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        if space_id is not None:
            query_params.update({'space_id': space_id})
        else:
            query_params.update({'project_id': project_id})

        href = self._client.service_instance._href_definitions.v4ga_cloud_migration_href()

        if migration_id is None:
            return self._get_artifact_details(href, migration_id, limit, 'migration', query_params=query_params,
                                              _async=asynchronous, _all=get_all)

        else:
            return self._get_artifact_details(href, migration_id, limit, 'migration', query_params=query_params)

    def list(self, space_id=None, project_id=None, limit=None):
        """Print the migration jobs in a table format. If limit is set to None there will be only
        first 50 records shown.

        :param space_id: space identifier
        :type space_id: str, optional
        :param project_id: project identifier
        :type project_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        **Example**

        .. code-block:: python

            client.v4ga_cloud_migration.list()
        """

        print(self.notification_msg)

        if space_id is None and project_id is None:
            raise WMLClientError("Its mandatory to provide space_id or project_id")

        self.skip_msg = True

        migration_resources = self.get_details(space_id=space_id, project_id=project_id, limit=limit)[u'resources']

        self.skip_msg = False

        if space_id is not None:
            migration_values = [(m[u'migration_id'],
                                 m[u'status'],
                                 m[u'space_id']) for m in migration_resources]

            self._list(migration_values,
                       [u'ID', u'STATUS', u'SPACE_ID'],
                       limit,
                       _DEFAULT_LIST_LENGTH)
        else:
            migration_values = [(m[u'migration_id'],
                                 m[u'status'],
                                 m[u'project_id']) for m in migration_resources]
            self._list(migration_values,
                       [u'MIGRATION_ID', u'STATUS', u'PROJECT_ID'],
                       limit,
                       _DEFAULT_LIST_LENGTH)
