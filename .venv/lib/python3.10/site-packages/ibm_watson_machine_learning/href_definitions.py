#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import re

TRAINING_MODEL_HREF_PATTERN = u'{}/v4/trainings/{}'
TRAINING_MODELS_HREF_PATTERN = u'{}/v4/trainings'
REPO_MODELS_FRAMEWORKS_HREF_PATTERN = u'{}/v3/models/frameworks'

INSTANCE_ENDPOINT_HREF_PATTERN = u'{}/v3/wml_instance'
INSTANCE_BY_ID_ENDPOINT_HREF_PATTERN = u'{}/v3/wml_instances/{}'
TOKEN_ENDPOINT_HREF_PATTERN = u'{}/v3/identity/token'
CPD_TOKEN_ENDPOINT_HREF_PATTERN = u'{}/icp4d-api/v1/authorize'
CPD_BEDROCK_TOKEN_ENDPOINT_HREF_PATTERN = u'{}/idprovider/v1/auth/identitytoken'
CPD_VALIDATION_TOKEN_ENDPOINT_HREF_PATTERN = u'{}/v1/preauth/validateAuth'
EXPERIMENTS_HREF_PATTERN = u'{}/v4/experiments'
EXPERIMENT_HREF_PATTERN = u'{}/v4/experiments/{}'
EXPERIMENT_RUNS_HREF_PATTERN = u'{}/v3/experiments/{}/runs'
EXPERIMENT_RUN_HREF_PATTERN = u'{}/v3/experiments/{}/runs/{}'

PUBLISHED_MODEL_HREF_PATTERN = u'{}/v4/models/{}'
PUBLISHED_MODELS_HREF_PATTERN = u'{}/v4/models'
LEARNING_CONFIGURATION_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/learning_configuration'
LEARNING_ITERATION_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/learning_iterations/{}'
LEARNING_ITERATIONS_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/learning_iterations'
EVALUATION_METRICS_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/evaluation_metrics'
FEEDBACK_HREF_PATTERN = u'{}/v3/wml_instances/{}/published_models/{}/feedback'

DEPLOYMENTS_HREF_PATTERN = u'{}/v4/deployments'
DEPLOYMENT_HREF_PATTERN = u'{}/v4/deployments/{}'
DEPLOYMENT_JOB_HREF_PATTERN = u'{}/v4/deployment_jobs'
DEPLOYMENT_JOBS_HREF_PATTERN = u'{}/v4/deployment_jobs/{}'
DEPLOYMENT_ENVS_HREF_PATTERN = u'{}/v4/deployments/environments'
DEPLOYMENT_ENV_HREF_PATTERN = u'{}/v4/deployments/environments/{}'

MODEL_LAST_VERSION_HREF_PATTERN = u'{}/v4/models/{}'
DEFINITION_HREF_PATTERN = u'{}/v3/ml_assets/training_definitions/{}'
DEFINITIONS_HREF_PATTERN = u'{}/v3/ml_assets/training_definitions'

FUNCTION_HREF_PATTERN = u'{}/v4/functions/{}'
FUNCTION_LATEST_CONTENT_HREF_PATTERN = u'{}/v4/functions/{}/content'
FUNCTIONS_HREF_PATTERN = u'{}/v4/functions'

RUNTIME_HREF_PATTERN = u'{}/v4/runtimes/{}'
RUNTIMES_HREF_PATTERN = u'{}/v4/runtimes'
CUSTOM_LIB_HREF_PATTERN = u'{}/v4/libraries/{}'
CUSTOM_LIBS_HREF_PATTERN = u'{}/v4/libraries'

IAM_TOKEN_API = u'{}&grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey'
IAM_TOKEN_URL = u'{}/oidc/token'
PROD_SVT_URL = ['https://us-south.ml.cloud.ibm.com',
                'https://eu-gb.ml.cloud.ibm.com',
                'https://eu-de.ml.cloud.ibm.com',
                'https://jp-tok.ml.cloud.ibm.com',
                'https://ibm-watson-ml.mybluemix.net',
                'https://ibm-watson-ml.eu-gb.bluemix.net',
                'https://private.us-south.ml.cloud.ibm.com',
                'https://private.eu-gb.ml.cloud.ibm.com',
                'https://private.eu-de.ml.cloud.ibm.com',
                'https://private.jp-tok.ml.cloud.ibm.com',
                'https://yp-qa.ml.cloud.ibm.com',
                'https://private.yp-qa.ml.cloud.ibm.com',
                'https://yp-cr.ml.cloud.ibm.com',
                'https://private.yp-cr.ml.cloud.ibm.com']

PIPELINES_HREF_PATTERN=u'{}/v4/pipelines'
PIPELINE_HREF_PATTERN=u'{}/v4/pipelines/{}'


SPACES_HREF_PATTERN = u'{}/v4/spaces'
SPACE_HREF_PATTERN = u'{}/v4/spaces/{}'
MEMBER_HREF_PATTERN=u'{}/v4/spaces/{}/members/{}'
MEMBERS_HREF_PATTERN=u'{}/v4/spaces/{}/members'

SPACES_PLATFORM_HREF_PATTERN = u'{}/v2/spaces'
SPACE_PLATFORM_HREF_PATTERN = u'{}/v2/spaces/{}'
SPACES_MEMBERS_HREF_PATTERN = u'{}/v2/spaces/{}/members'
SPACES_MEMBER_HREF_PATTERN = u'{}/v2/spaces/{}/members/{}'

V4_INSTANCE_ID_HREF_PATTERN = u'{}/ml/v4/instances/{}'

API_VERSION = u'/v4'
SPACES=u'/spaces'
PIPELINES=u'/pipelines'
EXPERIMENTS=u'/experiments'
LIBRARIES=u'/libraries'
RUNTIMES=u'/runtimes'
SOFTWARE_SPEC=u'/software_specifications'
DEPLOYMENTS = u'/deployments'
ASSET = u'{}/v2/assets/{}'
ASSETS = u'{}/v2/assets'
ASSET_TYPE = u'{}/v2/asset_types'
ASSET_FILES = u'{}/v2/asset_files/'
ATTACHMENT = u'{}/v2/assets/{}/attachments/{}'
ATTACHMENT_COMPLETE = u'{}/v2/assets/{}/attachments/{}/complete'
ATTACHMENTS = u'{}/v2/assets/{}/attachments'
SEARCH_ASSETS = u'{}/v2/asset_types/{}/search'
SEARCH_MODEL_DEFINITIONS = u'{}/v2/asset_types/wml_model_definition/search'
SEARCH_DATA_ASSETS = u'{}/v2/asset_types/data_asset/search'
SEARCH_SHINY = u'{}/v2/asset_types/shiny_asset/search'
SEARCH_SCRIPT = u'{}/v2/asset_types/script/search'
GIT_BASED_PROJECT_ASSET = u'{}/userfs/v2/assets/{}'
GIT_BASED_PROJECT_ASSETS = u'{}/userfs/v2/assets'
GIT_BASED_PROJECT_ASSET_TYPE = u'{}/userfs/v2/asset_types'
GIT_BASED_PROJECT_ASSET_FILES = u'{}/v2/asset_files/'
GIT_BASED_PROJECT_ATTACHMENT = u'{}/userfs/v2/assets/{}/attachments/{}'
GIT_BASED_PROJECT_ATTACHMENT_COMPLETE = u'{}/userfs/v2/assets/{}/attachments/{}/complete'
GIT_BASED_PROJECT_ATTACHMENTS = u'{}/userfs/v2/assets/{}/attachments'
GIT_BASED_PROJECT_SEARCH_ASSETS = u'{}/userfs/v2/asset_types/{}/search'
GIT_BASED_PROJECT_SEARCH_MODEL_DEFINITIONS = u'{}/userfs/v2/asset_types/wml_model_definition/search'
GIT_BASED_PROJECT_SEARCH_DATA_ASSETS = u'{}/userfs/v2/asset_types/data_asset/search'
GIT_BASED_PROJECT_SEARCH_SHINY = u'{}/userfs/v2/asset_types/shiny_asset/search'
GIT_BASED_PROJECT_SEARCH_SCRIPT = u'{}/userfs/v2/asset_types/script/search'
DATA_SOURCE_TYPE = u'{}/v2/datasource_types'
DATA_SOURCE_TYPE_BY_ID = u'{}/v2/datasource_types/{}'
CONNECTION_ASSET = u'{}/v2/connections'
CONNECTION_ASSET_SEARCH = u'{}/v2/connections'
CONNECTION_BY_ID = u'{}/v2/connections/{}'
CONNECTIONS_FILES = u'{}/v2/connections/files'
CONNECTIONS_FILE = u'{}/v2/connections/files/{}'
SOFTWARE_SPECIFICATION = u'{}/v2/software_specifications/{}'
SOFTWARE_SPECIFICATIONS = u'{}/v2/software_specifications'
HARDWARE_SPECIFICATION = u'{}/v2/hardware_specifications/{}'
HARDWARE_SPECIFICATIONS = u'{}/v2/hardware_specifications'
PACKAGE_EXTENSION = u'{}/v2/package_extensions/{}'
PACKAGE_EXTENSIONS = u'{}/v2/package_extensions'
PROJECT = u'{}/v2/projects/{}'
LIST_SOFTWARE_SPECIFICATIONS = '{}/ml/v4/list/software_specifications'

V4GA_CLOUD_MIGRATION = u'{}/ml/v4/repository'
V4GA_CLOUD_MIGRATION_ID = u'{}/ml/v4/repository/{}'

REMOTE_TRAINING_SYSTEM = u'{}/v4/remote_training_systems'
REMOTE_TRAINING_SYSTEM_ID = u'{}/v4/remote_training_systems/{}'

EXPORTS = u'{}/v2/asset_exports'
EXPORT_ID = u'{}/v2/asset_exports/{}'
EXPORT_ID_CONTENT = u'{}/v2/asset_exports/{}/content'

IMPORTS = u'{}/v2/asset_imports'
IMPORT_ID = u'{}/v2/asset_imports/{}'

VOLUMES = u'{}/zen-data/v3/service_instances'
VOLUME_ID = u'{}/zen-data/v3/service_instances/{}'
VOLUME_SERVICE = u'{}/zen-data/v1/volumes/volume_services/{}'
VOLUME_SERVICE_FILE_UPLOAD = u'{}/zen-volumes/{}/v1/volumes/files/'
VOLUME_MONITOR = u'{}/zen-volumes/{}/v1/monitor'

PROMOTE_ASSET = u'{}/projects/api/rest/catalogs/assets/{}/promote'

DATAPLATFORM_URLS_MAP = {
            'https://wml-fvt.ml.test.cloud.ibm.com': 'https://dataplatform.dev.cloud.ibm.com',
            'https://yp-qa.ml.cloud.ibm.com': 'https://dataplatform.test.cloud.ibm.com',
            'https://private.yp-qa.ml.cloud.ibm.com': 'https://dataplatform.test.cloud.ibm.com',
            'https://yp-cr.ml.cloud.ibm.com': 'https://dataplatform.test.cloud.ibm.com',
            'https://private.yp-cr.ml.cloud.ibm.com': 'https://dataplatform.test.cloud.ibm.com',
            'https://jp-tok.ml.cloud.ibm.com': 'https://jp-tok.dataplatform.cloud.ibm.com',
            'https://eu-gb.ml.cloud.ibm.com': 'https://eu-gb.dataplatform.cloud.ibm.com',
            'https://eu-de.ml.cloud.ibm.com': 'https://eu-de.dataplatform.cloud.ibm.com',
            'https://us-south.ml.cloud.ibm.com': 'https://dataplatform.cloud.ibm.com'
        }

WKC_MODEL_REGISTER = u"{}/v1/aigov/model_inventory/models/{}/model_entry"
WKC_MODEL_LIST_FROM_CATALOG = u"{}/v1/aigov/model_inventory/{}/model_entries"
WKC_MODEL_LIST_ALL = u"{}/v1/aigov/model_inventory/model_entries"
TASK_CREDENTIALS = u"{}/v1/task_credentials/{}"
TASK_CREDENTIALS_ALL = u"{}/v1/task_credentials"

def is_url(s):
    res = re.match('https?:\/\/.+', s)
    return res is not None


def is_uid(s):
    res = re.match('[a-z0-9\-]{36}', s)
    return res is not None


class HrefDefinitions:
    def __init__(self, client, cloud_platform_spaces=False, platform_url=None, cams_url=None,
                 cp4d_platform_spaces=False):
        self._wml_credentials = client.wml_credentials
        self._client = client
        self.cloud_platform_spaces = cloud_platform_spaces
        self.cp4d_platform_spaces = cp4d_platform_spaces
        self.platform_url = platform_url
        self.cams_url = cams_url

        if self.cloud_platform_spaces or self.cp4d_platform_spaces:
            self.prepend = '/ml'
        else:
            self.prepend = ''

    def _is_git_based_project(self):
        return self._client.project_type == "local_git_storage"

    def _get_platform_url_if_exists(self):
        return self.platform_url if self.platform_url else self._wml_credentials['url']

    def _get_cams_url_if_exists(self):
        return self.cams_url if self.cams_url else self._wml_credentials['url']

    def get_training_href(self, model_uid):
        return TRAINING_MODEL_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, model_uid)

    def get_trainings_href(self):
        return TRAINING_MODELS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_repo_models_frameworks_href(self):
        return REPO_MODELS_FRAMEWORKS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_instance_endpoint_href(self):
        return INSTANCE_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_instance_by_id_endpoint_href(self):
        return INSTANCE_BY_ID_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'])

    def get_token_endpoint_href(self):
        return TOKEN_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_cpd_token_endpoint_href(self):
        return CPD_TOKEN_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['url'].replace(':31002', ':31843'))

    def get_cpd_bedrock_token_endpoint_href(self):
        return CPD_BEDROCK_TOKEN_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['bedrock_url'])

    def get_cpd_validation_token_endpoint_href(self):
        return CPD_VALIDATION_TOKEN_ENDPOINT_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_published_model_href(self, model_uid):
        return PUBLISHED_MODEL_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, model_uid)

    def get_published_models_href(self):
        return PUBLISHED_MODELS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_learning_configuration_href(self, model_uid):
        return LEARNING_CONFIGURATION_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid)

    def get_learning_iterations_href(self, model_uid):
        return LEARNING_ITERATIONS_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid)

    def get_learning_iteration_href(self, model_uid, iteration_uid):
        return LEARNING_ITERATION_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid, iteration_uid)

    def get_evaluation_metrics_href(self, model_uid):
        return EVALUATION_METRICS_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid)

    def get_feedback_href(self, model_uid):
        return FEEDBACK_HREF_PATTERN.format(self._wml_credentials['url'], self._wml_credentials['instance_id'], model_uid)

    def get_model_last_version_href(self, artifact_uid):
        return MODEL_LAST_VERSION_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, artifact_uid)

    def get_deployments_href(self):
        return DEPLOYMENTS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_experiments_href(self):
        return EXPERIMENTS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_experiment_href(self, experiment_uid):
        return EXPERIMENT_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, experiment_uid)

    def get_experiment_runs_href(self, experiment_uid):
        return EXPERIMENT_RUNS_HREF_PATTERN.format(self._wml_credentials['url'], experiment_uid)

    def get_experiment_run_href(self, experiment_uid, experiment_run_uid):
        return EXPERIMENT_RUN_HREF_PATTERN.format(self._wml_credentials['url'], experiment_uid, experiment_run_uid)

    def get_deployment_href(self, deployment_uid):
        return DEPLOYMENT_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, deployment_uid)

    def get_definition_href(self, definition_uid):
        return DEFINITION_HREF_PATTERN.format(self._wml_credentials['url'], definition_uid)

    def get_definitions_href(self):
        return DEFINITIONS_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_function_href(self, ai_function_uid):
        return FUNCTION_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, ai_function_uid)

    def get_function_latest_revision_content_href(self, ai_function_uid):
        return FUNCTION_LATEST_CONTENT_HREF_PATTERN.format(self._wml_credentials['url'], ai_function_uid)

    def get_functions_href(self):
        return FUNCTIONS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_runtime_href_v4(self, runtime_uid):
        return u'/v4/runtimes/{}'.format(runtime_uid)

    def get_runtime_href(self, runtime_uid):
        return RUNTIME_HREF_PATTERN.format(self._wml_credentials['url'], runtime_uid)

    def get_runtimes_href(self):
        return RUNTIMES_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_custom_library_href(self, library_uid):
        return CUSTOM_LIB_HREF_PATTERN.format(self._wml_credentials['url'], library_uid)

    def get_custom_libraries_href(self):
        return CUSTOM_LIBS_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_pipeline_href(self, pipeline_uid):
        return PIPELINE_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, pipeline_uid)

    def get_pipelines_href(self):
        return PIPELINES_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_space_href(self, spaces_uid):
        return SPACE_HREF_PATTERN.format(self._wml_credentials['url'], spaces_uid)

    def get_spaces_href(self):
        return SPACES_HREF_PATTERN.format(self._wml_credentials['url'])

    def get_platform_space_href(self, spaces_id):
        return SPACE_PLATFORM_HREF_PATTERN.format(self._get_platform_url_if_exists(), spaces_id)

    def get_platform_spaces_href(self):
        return SPACES_PLATFORM_HREF_PATTERN.format(self._get_platform_url_if_exists())

    def get_platform_spaces_member_href(self, spaces_id, member_id):
        return SPACES_MEMBER_HREF_PATTERN.format(self._get_platform_url_if_exists(), spaces_id, member_id)

    def get_platform_spaces_members_href(self,spaces_id):
        return SPACES_MEMBERS_HREF_PATTERN.format(self._get_platform_url_if_exists(), spaces_id)

    def get_v4_instance_id_href(self):
        return V4_INSTANCE_ID_HREF_PATTERN.format(self._wml_credentials['url'],
                                                  self._wml_credentials['instance_id'])

    def get_async_deployment_job_href(self):
        return DEPLOYMENT_JOB_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend)

    def get_async_deployment_jobs_href(self, job_uid):
        return DEPLOYMENT_JOBS_HREF_PATTERN.format(self._wml_credentials['url'] + self.prepend, job_uid)

    def get_iam_token_api(self):
        return IAM_TOKEN_API.format(self._wml_credentials['apikey'])

    def get_iam_token_url(self):
        if self._wml_credentials['url'] in PROD_SVT_URL:
            return IAM_TOKEN_URL.format('https://iam.cloud.ibm.com')
        else:
            return IAM_TOKEN_URL.format('https://iam.test.cloud.ibm.com')

    def get_member_href(self, spaces_uid,member_id):
        return MEMBER_HREF_PATTERN.format(self._wml_credentials['url'], spaces_uid,member_id)

    def get_members_href(self,spaces_uid):
        return MEMBERS_HREF_PATTERN.format(self._wml_credentials['url'],spaces_uid)

    def get_data_asset_href(self,asset_id):
        return (ASSET if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET)\
            .format(self._get_cams_url_if_exists(), asset_id)

    def get_data_assets_href(self):
        return (ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS)\
            .format(self._get_cams_url_if_exists())

    def get_assets_href(self):
        return (ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS)\
            .format(self._get_cams_url_if_exists())

    def get_asset_href(self,asset_id):
        return (ASSET if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET)\
            .format(self._get_cams_url_if_exists(), asset_id)

    def get_base_asset_href(self,asset_id):
        return (ASSET if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET).format('', asset_id)

    def get_base_assets_href(self):
        return (ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS).format('')

    def get_base_asset_with_type_href(self,asset_type, asset_id):
        return (ASSET if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET).format('', asset_type) + '/' + asset_id

    def get_attachment_href(self,asset_id,attachment_id):
        return (ATTACHMENT if not self._is_git_based_project() else GIT_BASED_PROJECT_ATTACHMENT)\
            .format(self._get_cams_url_if_exists(), asset_id, attachment_id)

    def get_attachments_href(self, asset_id):
        return (ATTACHMENTS if not self._is_git_based_project() else GIT_BASED_PROJECT_ATTACHMENTS)\
            .format(self._get_cams_url_if_exists(), asset_id)

    def get_attachment_complete_href(self,asset_id,attachment_id):
        return (ATTACHMENT_COMPLETE if not self._is_git_based_project() else GIT_BASED_PROJECT_ATTACHMENT_COMPLETE)\
            .format(self._get_cams_url_if_exists(), asset_id, attachment_id)

    def get_search_asset_href(self):
        return (SEARCH_DATA_ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_SEARCH_DATA_ASSETS)\
            .format(self._get_cams_url_if_exists())

    def get_search_shiny_href(self):
        return (SEARCH_SHINY if not self._is_git_based_project() else GIT_BASED_PROJECT_SEARCH_SHINY)\
            .format(self._get_cams_url_if_exists())

    def get_search_script_href(self):
        return (SEARCH_SCRIPT if not self._is_git_based_project() else GIT_BASED_PROJECT_SEARCH_SCRIPT)\
            .format(self._get_cams_url_if_exists())

    def get_model_definition_assets_href(self):
        return (ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS)\
            .format(self._get_cams_url_if_exists())

    def get_model_definition_search_asset_href(self):
        return (SEARCH_MODEL_DEFINITIONS if not self._is_git_based_project() else GIT_BASED_PROJECT_SEARCH_MODEL_DEFINITIONS)\
            .format(self._get_cams_url_if_exists())

    def get_wsd_model_href(self):
        return (ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS).format(self._wml_credentials['url'])

    def get_wsd_model_attachment_href(self):
        return (ASSET_FILES if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET_FILES)\
            .format(self._wml_credentials['url'])

    def get_asset_search_href(self, asset_type):
        return (SEARCH_ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_SEARCH_ASSETS)\
            .format(self._get_cams_url_if_exists(), asset_type)

    def get_wsd_asset_type_href(self):
        return (ASSET_TYPE if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET_TYPE)\
            .format(self._wml_credentials['url'])

    def get_wsd_base_href(self):
        return self._wml_credentials['url']

    def get_connections_href(self):
        return CONNECTION_ASSET.format(self._get_platform_url_if_exists())

    def get_connection_by_id_href(self, connection_id):
        return CONNECTION_BY_ID.format(self._get_platform_url_if_exists(), connection_id)

    def get_connections_files_href(self):
        return CONNECTIONS_FILES.format(self._get_platform_url_if_exists())

    def get_connections_file_href(self, file_name):
        return CONNECTIONS_FILE.format(self._get_platform_url_if_exists(), file_name)

    def get_connection_data_types_href(self):
        return DATA_SOURCE_TYPE.format(self._get_platform_url_if_exists())

    def get_sw_spec_href(self, sw_spec_id):
        return SOFTWARE_SPECIFICATION.format(self._get_platform_url_if_exists(), sw_spec_id)

    def get_sw_specs_href(self):
        return SOFTWARE_SPECIFICATIONS.format(self._get_platform_url_if_exists())

    def get_hw_spec_href(self, hw_spec_id):
        return HARDWARE_SPECIFICATION.format(self._get_platform_url_if_exists(), hw_spec_id)

    def get_hw_specs_href(self):
        return HARDWARE_SPECIFICATIONS.format(self._get_platform_url_if_exists())

    def get_pkg_extn_href(self, pkg_extn_id):
        return PACKAGE_EXTENSION.format(self._get_platform_url_if_exists(), pkg_extn_id)

    def get_pkg_extns_href(self):
        return PACKAGE_EXTENSIONS.format(self._get_platform_url_if_exists())

    def get_project_href(self, project_id):
        return PROJECT.format(self._get_platform_url_if_exists(), project_id)

    def get_software_specifications_list_href(self):
        return LIST_SOFTWARE_SPECIFICATIONS.format(self._wml_credentials['url'])

    def v4ga_cloud_migration_href(self):
        return V4GA_CLOUD_MIGRATION.format(self._wml_credentials['url'])

    def v4ga_cloud_migration_id_href(self, migration_id):
        return V4GA_CLOUD_MIGRATION_ID.format(self._wml_credentials['url'], migration_id)

    def exports_href(self):
        return EXPORTS.format(self._get_platform_url_if_exists())

    def export_href(self, export_id):
        return EXPORT_ID.format(self._get_platform_url_if_exists(), export_id)

    def export_content_href(self, export_id):
        return EXPORT_ID_CONTENT.format(self._get_platform_url_if_exists(), export_id)

    def imports_href(self):
        return IMPORTS.format(self._get_platform_url_if_exists())

    def import_href(self, export_id):
        return IMPORT_ID.format(self._get_platform_url_if_exists(), export_id)

    def remote_training_systems_href(self):
        return REMOTE_TRAINING_SYSTEM.format(self._wml_credentials['url'] + self.prepend)

    def remote_training_system_href(self, remote_training_systems_id):
        return REMOTE_TRAINING_SYSTEM_ID.format(self._wml_credentials['url'] + self.prepend, remote_training_systems_id)

    def volumes_href(self):
        return VOLUMES.format(self._wml_credentials['url'])

    def volume_href(self,volume_id):
        return VOLUME_ID.format(self._wml_credentials['url'], volume_id)

    def volume_service_href(self,volume_name):
        return VOLUME_SERVICE.format(self._wml_credentials['url'], volume_name)

    def volume_upload_href(self, volume_name):
        return VOLUME_SERVICE_FILE_UPLOAD.format(self._wml_credentials['url'], volume_name)

    def volume_monitor_href(self, volume_name):
        return VOLUME_MONITOR.format(self._wml_credentials['url'], volume_name)

    def promote_asset_href(self, asset_id):
        if self.cloud_platform_spaces:
            data_platform_url = DATAPLATFORM_URLS_MAP[self._wml_credentials['url']]
            return PROMOTE_ASSET.format(data_platform_url, asset_id)
        else:
            promote_href = PROMOTE_ASSET.format(self._wml_credentials['url'], asset_id)
            try:
                # note: For CPD older than 4.0 we need to roll back to older endpoint.
                if float(self._wml_credentials.get("version")) < 4.0:
                    promote_href = promote_href.replace("/projects", "")
                # --- end note
            finally:
                return promote_href

    def get_wkc_model_register_href(self, model_id) -> str:
        return WKC_MODEL_REGISTER.format(self._get_platform_url_if_exists(), model_id)

    def get_wkc_model_list_from_catalog_href(self, catalog_id) -> str:
        return WKC_MODEL_LIST_FROM_CATALOG.format(self._get_platform_url_if_exists(), catalog_id)

    def get_wkc_model_list_all_href(self) -> str:
        return WKC_MODEL_LIST_ALL.format(self._get_platform_url_if_exists())

    def get_wkc_model_delete_href(self, asset_id) -> str:
        return WKC_MODEL_REGISTER.format(self._get_platform_url_if_exists(), asset_id)

    def get_task_credentials_href(self, task_credentials_uid) -> str:
        return TASK_CREDENTIALS.format(self._get_platform_url_if_exists(), task_credentials_uid)

    def get_task_credentials_all_href(self) -> str:
        return TASK_CREDENTIALS_ALL.format(self._get_platform_url_if_exists())


    # def get_envs_href(self):
    #     return DEPLOYMENT_ENVS_HREF_PATTERN.format(self._wml_credentials['url'])
    #
    # def get_env_href(self, env_id):
    #     return DEPLOYMENT_ENV_HREF_PATTERN.format(self._wml_credentials['url'],env_id)
