#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import re
import os
import sys
import pkg_resources
import shutil
import tarfile
import logging
import importlib.util

from typing import Optional
from subprocess import check_call
from packaging import version

import ibm_watson_machine_learning._wrappers.requests as requests

from ibm_watson_machine_learning.wml_client_error import WMLClientError, CannotInstallLibrary

INSTANCE_DETAILS_TYPE = u'instance_details_type'
PIPELINE_DETAILS_TYPE = u'pipeline_details_type'
DEPLOYMENT_DETAILS_TYPE = u'deployment_details_type'
EXPERIMENT_RUN_DETAILS_TYPE = u'experiment_run_details_type'
MODEL_DETAILS_TYPE = u'model_details_type'
DEFINITION_DETAILS_TYPE = u'definition_details_type'
EXPERIMENT_DETAILS_TYPE = u'experiment_details_type'
TRAINING_RUN_DETAILS_TYPE = u'training_run_details_type'
FUNCTION_DETAILS_TYPE = u'function_details_type'
DATA_ASSETS_DETAILS_TYPE = u'data_assets_details_type'
SW_SPEC_DETAILS_TYPE = u'sw_spec_details_type'
HW_SPEC_DETAILS_TYPE = u'hw_spec_details_type'
RUNTIME_SPEC_DETAILS_TYPE = u'runtime_spec_details_type'
LIBRARY_DETAILS_TYPE = u'library_details_type'
SPACES_DETAILS_TYPE = u'spaces_details_type'
MEMBER_DETAILS_TYPE = u'member_details_type'
CONNECTION_DETAILS_TYPE = u'connection_details_type'
PKG_EXTN_DETAILS_TYPE = u'pkg_extn_details_type'
UNKNOWN_ARRAY_TYPE = u'resource_type'
UNKNOWN_TYPE = u'unknown_type'
SPACES_IMPORTS_DETAILS_TYPE = u'spaces_imports_details_type'
SPACES_EXPORTS_DETAILS_TYPE = u'spaces_exports_details_type'

SPARK_MLLIB = u'mllib'
SPSS_FRAMEWORK = u'spss-modeler'
TENSORFLOW_FRAMEWORK = u'tensorflow'
XGBOOST_FRAMEWORK = u'xgboost'
SCIKIT_LEARN_FRAMEWORK = u'scikit-learn'
PMML_FRAMEWORK = u'pmml'


def is_python_2():
    return sys.version_info[0] == 2


def get_url(url, headers, params=None, isIcp=False):
    import ibm_watson_machine_learning._wrappers.requests as requests

    if isIcp:
        return requests.get(url, headers=headers, params=params)
    else:
        return requests.get(url, headers=headers, params=params)


def print_text_header_h1(title):
    print(u'\n\n' + (u'#' * len(title)) + u'\n')
    print(title)
    print(u'\n' + (u'#' * len(title)) + u'\n\n')


def print_text_header_h2(title):
    print(u'\n\n' + (u'-' * len(title)))
    print(title)
    print((u'-' * len(title)) + u'\n\n')


def get_type_of_details(details):
    if 'resources' in details:
        return UNKNOWN_ARRAY_TYPE
    elif details is None:
        raise WMLClientError('Details doesn\'t exist.')
    else:
        try:
            plan = 'plan' in details[u'entity']

            if plan:
                return INSTANCE_DETAILS_TYPE

            if re.search(u'\/wml_instances\/[^\/]+$', details[u'metadata'][u'url']) is not None:
                return INSTANCE_DETAILS_TYPE
        except:
            pass
        try:
            if re.search(u'\/pipelines\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return PIPELINE_DETAILS_TYPE
        except:
            pass
        try:
            if 'href' in details[u'metadata'] and re.search(u'\/deployments\/[^\/]+$', details[u'metadata'][u'href']) is not None \
                    or re.search(u'\/deployments\/[^\/]+$', details[u'metadata'][u'id']) is not None \
                    or u'virtual_deployment_downloads' in details[u'entity'][u'status']:
                return DEPLOYMENT_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/experiments\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return EXPERIMENT_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/trainings\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return TRAINING_RUN_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/models\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return MODEL_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/functions\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return FUNCTION_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/runtimes\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return RUNTIME_SPEC_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/libraries\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return LIBRARY_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/spaces\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return SPACES_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/members\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return MEMBER_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/members\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return MEMBER_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/assets\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return DATA_ASSETS_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/software_specifications\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return SW_SPEC_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/hardware_specifications\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return HW_SPEC_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/package_extension\/[^\/]+$', details[u'entity'][u'package_extension'][u'href']) is not None:
                return PKG_EXTN_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/imports\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return SPACES_IMPORTS_DETAILS_TYPE
        except:
            pass

        try:
            if re.search(u'\/exports\/[^\/]+$', details[u'metadata'][u'href']) is not None:
                return SPACES_EXPORTS_DETAILS_TYPE
        except:
            pass

        return UNKNOWN_TYPE


def load_model_from_directory(framework, directory_path):
    if "mllib" in framework:
     from pyspark.ml import PipelineModel
     return PipelineModel.read().load(directory_path)
    if "spss" in framework:
     pass
    if "tensorflow" in framework:
     pass
    if "scikit" in framework or "xgboost" in framework:
        try:
            try:
                from sklearn.externals import joblib
            except ImportError:
                import joblib
            pkl_files = [x for x in os.listdir(directory_path) if x.endswith('.pkl')]

            if len(pkl_files) < 1:
                raise WMLClientError('No pkl files in directory.')

            model_id = pkl_files[0]
            return joblib.load(os.path.join(directory_path, model_id))
        except Exception as e:
            raise WMLClientError('Cannot load model from pkl file.', e)
    if "pmml" in framework:
     pass
    else:
        raise WMLClientError(u'Invalid framework specified: \'{}\'.'.format(framework))


# def load_model_from_directory(framework, directory_path):
#     if framework == SPARK_MLLIB:
#         from pyspark.ml import PipelineModel
#         return PipelineModel.read().load(directory_path)
#     elif framework == SPSS_FRAMEWORK:
#         pass
#     elif framework == TENSORFLOW_FRAMEWORK:
#         pass
#     elif framework == SCIKIT_LEARN_FRAMEWORK or framework == XGBOOST_FRAMEWORK:
#         from sklearn.externals import joblib
#         model_id = directory_path[directory_path.rfind('/') + 1:] + ".pkl"
#         return joblib.load(os.path.join(directory_path, model_id))
#     elif framework == PMML_MODEL:
#         pass
#     else:
#         raise WMLClientError('Invalid framework specified: \'{}\'.'.format(framework))


def save_model_to_file(model, framework, base_path, filename):
    if filename.find('.') != -1:
        base_name = filename[:filename.find('.') + 1]
        file_extension = filename[filename.find('.'):]
    else:
        base_name = filename
        file_extension = 'tar.gz'

    if framework == SPARK_MLLIB:
        model.write.overwrite.save(os.path.join(base_path, base_name))
    elif framework == SPSS_FRAMEWORK:
        pass
    elif framework == TENSORFLOW_FRAMEWORK:
        pass
    elif framework == XGBOOST_FRAMEWORK:
        pass
    elif framework == SCIKIT_LEARN_FRAMEWORK:
        os.makedirs(os.path.join(base_path, base_name))
        try:
            from sklearn.externals import joblib
        except ImportError:
            import joblib
        joblib.dump(model, os.path.join(base_path, base_name, base_name + ".pkl"))
    elif framework == PMML_FRAMEWORK:
        pass
    else:
        raise WMLClientError(u'Invalid framework specified: \'{}\'.'.format(framework))


def format_metrics(latest_metrics_list):
    formatted_metrics = u''

    for i in latest_metrics_list:

        values = i[u'values']

        if len(values) > 0:
            sorted_values = sorted(values, key=lambda k: k[u'name'])
        else:
            sorted_values = values

        for j in sorted_values:
            formatted_metrics = formatted_metrics + i[u'phase'] + ':' + j[u'name']+'='+'{0:.4f}'.format(j[u'value']) + '\n'

    return formatted_metrics


def inherited_docstring(f, mapping={}):
    def dec(obj):
        if not obj.__doc__:
            possible_types = {'model': 'model',
                              'function': 'function',
                              'space': 'space',
                              'pipeline': 'pipeline',
                              'experiment': 'experiment',
                              'member': 'space'}

            available_metanames = {'model': 'ModelMetaNames',
                                   'experiment': 'ExperimentMetaNames',
                                   'function': 'FunctionMetaNames',
                                   'pipeline': 'PipelineMetaNames'}

            actual_type = None

            for t in possible_types:
                if t in obj.__name__:
                    actual_type = possible_types[t]

            docs = f.__doc__

            if actual_type:
                docs = docs.replace(f'client.{actual_type}s.{f.__name__}', 'client.repository.' + obj.__name__)
                docs = docs.replace(f'client._{actual_type}s.{f.__name__}', 'client.repository.' + obj.__name__)

                if actual_type in available_metanames:
                    repository_meta_names = available_metanames[actual_type]
                    docs = docs.replace(f'_{actual_type}s.ConfigurationMetaNames', f'repository.{repository_meta_names}')
                    docs = docs.replace(f'{actual_type}s.ConfigurationMetaNames', f'repository.{repository_meta_names}')
                    docs = docs.replace('ConfigurationMetaNames', repository_meta_names)

                for k in mapping:
                    docs = docs.replace(k, mapping[k])
            obj.__doc__ = docs
        return obj
    return dec


def group_metrics(metrics):
    grouped_metrics = []

    if len(metrics) > 0:
        import collections
        grouped_metrics = collections.defaultdict(list)
        for d in metrics:
            k = d[u'phase']
            grouped_metrics[k].append(d)

    return grouped_metrics


class StatusLogger:
    def __init__(self, initial_state):
        self.last_state = initial_state
        print(initial_state, end='')

    def log_state(self, state):
        if state == self.last_state:
            print('.', end='')
        else:
            print('\n{}'.format(state), end='')
            self.last_state = state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_file_from_cos(cos_credentials):
    import ibm_boto3
    from ibm_botocore.client import Config

    client_cos = ibm_boto3.client(service_name='s3',
        ibm_api_key_id=cos_credentials['IBM_API_KEY_ID'],
        ibm_auth_endpoint=cos_credentials['IBM_AUTH_ENDPOINT'],
        config=Config(signature_version='oauth'),
        endpoint_url=cos_credentials['ENDPOINT'])

    streaming_body = client_cos.get_object(Bucket=cos_credentials['BUCKET'], Key=cos_credentials['FILE'])['Body']
    training_definition_bytes = streaming_body.read()
    streaming_body.close()
    filename = cos_credentials['FILE']
    f = open(filename, 'wb')
    f.write(training_definition_bytes)
    f.close()

    return filename


def extract_model_from_repository(model_uid, client):
    """Download and extract archived model from wml repository.

    :param model_uid: UID of model
    :type model_uid: str
    :param client: client instance
    :type client: APIClient

    :return: extracted directory path
    :rtype: str
    """
    create_empty_directory(model_uid)
    current_dir = os.getcwd()

    os.chdir(model_uid)
    model_dir = os.getcwd()

    fname = 'downloaded_' + model_uid + '.tar.gz'
    client.repository.download(model_uid, filename=fname)

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
    else:
        raise WMLClientError('Invalid type. Expected tar.gz')

    os.chdir(current_dir)
    return model_dir


def extract_mlmodel_from_archive(archive_path, model_uid):
    """Extract archived model under model uid directory.

    :param model_uid: UID of model
    :type model_uid: str
    :param archive_path: path to archived model
    :type archive_path: str

    :return: extracted directory path
    :rtype: str
    """
    create_empty_directory(model_uid)
    current_dir = os.getcwd()

    os.rename(archive_path, os.path.join(model_uid, archive_path))

    os.chdir(model_uid)
    model_dir = os.getcwd()

    if archive_path.endswith("tar.gz"):
        tar = tarfile.open(archive_path)
        tar.extractall()
        tar.close()
    else:
        raise WMLClientError('Invalid type. Expected tar.gz')

    os.chdir(current_dir)
    return os.path.join(model_uid, 'model.mlmodel')


def get_model_filename(directory, model_extension):
    logger = logging.getLogger(__name__)
    model_filepath = None

    for file in os.listdir(directory):
        if file.endswith(model_extension):
            if model_filepath is None:
                model_filepath = os.path.join(directory, file)
            else:
                logger.warning('More than one file with extension \'{}\'.'.format(model_extension))

    if model_filepath is None:
        raise WMLClientError('No file with extension \'{}\'.'.format(model_extension))

    return model_filepath


def delete_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


def create_empty_directory(directory):
    delete_directory(directory)
    os.makedirs(directory)


def install_package(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])


def is_ipython():
    # checks if the code is run in the notebook
    try:
        get_ipython
        return True
    except Exception:
        return False


def create_download_link(file_path, title="Download file."):
    # creates download link for binary files on notebook filesystem (Watson Studio)

    if is_ipython():
        from IPython.display import HTML
        import base64

        filename = os.path.basename(file_path)

        with open(file_path, 'rb') as file:
            b_model = file.read()
        b64 = base64.b64encode(b_model)
        payload = b64.decode()
        html = '<a download="{file_path}" href="data:binary;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(payload=payload, title=title, file_path=filename)

        return HTML(html)


def convert_metadata_to_parameters(meta_data):
    parameters = []

    if meta_data is not None:
        if is_python_2():
            for key, value in meta_data.iteritems():
                parameters.append({'name': str(key), 'value': value})
        else:
            for key, value in meta_data.items():
                parameters.append({'name': str(key), 'value': value})

    return parameters


def is_of_python_basic_type(el):
    if type(el) in [int, float, bool, str]:
        return True
    elif type(el) in [list, tuple]:
        return all([is_of_python_basic_type(t) for t in el])
    elif type(el) is dict:
        if not all(type(k) == str for k in el.keys()):
            return False

        return is_of_python_basic_type(list(el.values()))
    else:
        return False


class NextResourceGenerator:
    """Generator class to produce next list of resources from REST API."""

    def __init__(self, wml_client: 'APIClient', url: str, href: str, params: dict = None, _all=False,
                 _filter_func=None) -> None:
        """
        :param wml_client: WML Client Instance
        :type wml_client: APIClient

        :param href: href to the resource
        :type href: str
        """
        self.wml_client = wml_client
        self.url = url
        self.next_href = href
        self.params = params
        self.all = _all
        self._filter_func = _filter_func

    def __iter__(self):
        if self.next_href is not None:
            response = requests.get(
                url=f"{self.url}/{self.next_href}",
                headers=self.wml_client._get_headers(),
                params=self.params if self.params is not None else self.wml_client.params()
            )
            details_json = self.wml_client.training._handle_response(200, "Get next details", response)

            if self.all:
                self.next_href = details_json.get('next', {'href': None})['href']

            else:
                self.next_href = None

            if 'resources' in details_json:
                resources = details_json['resources']

            elif 'metadata' in details_json:
                resources = [details_json]

            else:
                resources = details_json.get('results', [])

            yield {'resources': self._filter_func(resources) if self._filter_func else resources}

        else:
            raise StopIteration


class DisableWarningsLogger:
    """Class which disables logging warnings (for example for silent handling WMLClientErrors in try except).

    **Example**

    .. code-block:: python

        try:
            with DisableWarningsLogger():
                throw_wml_error()
        except WMLClientError:
            success = False

    """
    def __enter__(self):
        logging.disable(logging.WARNING)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def is_lib_installed(lib_name: str, minimum_version: Optional[str] = None, install: Optional[bool] = False) -> bool:
    """Check if provided library is installed on user environment. If not, tries to install it.

    :param lib_name: library name to check
    :type lib_name: str

    :param minimum_version: minimum version of library to check, default: None - check if library is installed in overall
    :type minimum_version: str, optional

    :param install: indicates to install missing or to low version library
    :type install: bool, optional

    :return: information if library is installed: `True` is library is installed, `False` otherwise
    :rtype: bool
    """
    if lib_name in sys.modules:
        installed = True

    elif importlib.util.find_spec(lib_name) is not None:
        installed = True

    else:
        installed = False

    if installed:
        installed_module_version = get_module_version(lib_name)

        if minimum_version is not None:
            if version.parse(installed_module_version) < version.parse(minimum_version):
                if install:
                    install_library(lib_name=lib_name, version=minimum_version, strict=False)

    else:
        if install:
            install_library(lib_name=lib_name, version=minimum_version, strict=False)
            installed = True

    return installed


def install_library(lib_name: str, version: Optional[str] = None, strict: Optional[bool] = False) -> None:
    """Try to install library.

    :param lib_name: library name to install
    :type lib_name: str

    :param version: version of the library to install
    :type version: str, optional

    :param strict: indicates if we want to install specific version or higher version if available
    :type strict: bool, optional
    """
    try:
        if version is not None:
            check_call([sys.executable, "-m", "pip", "install", f"{lib_name}{'==' if strict else '>='}{version}"])

        else:
            check_call([sys.executable, "-m", "pip", "install", lib_name])

    except Exception as e:
        raise CannotInstallLibrary(lib_name=lib_name, reason=str(e))


def get_module_version(lib_name: str) -> str:
    """Use only when you need to check package version by package name with pip."""
    from importlib.metadata import version
    try:
        return version(lib_name)
    except:
        return pkg_resources.get_distribution(lib_name).version


def prepare_interaction_props_for_cos(source_params: dict, file_name: str) -> dict:
    """If user specified properties for dataset as sheet_name, delimiter etc. we need to
    pass them as interaction properties for Flight Service.

    :param source_params: data source parameters describe data (eg. excel_sheet, encoding etc.)
    :type source_params: dict

    :param file_name: name of the file to download, should consist of file extension
    :type file_name: str

    :return: COS interaction properties for Flight Service
    :rtype: dict
    """
    interaction_properties = {}
    file_format = None

    encoding = source_params.get("encoding", None)

    if ".xls" in file_name or ".xlsx" in file_name:
        file_format = "excel"
        if source_params.get("excel_sheet"):
            interaction_properties["sheet_name"] = str(source_params.get("excel_sheet"))

    elif '.csv' in file_name:
        if source_params.get("quote_character"):
            interaction_properties["quote_character"] = str(source_params.get("quote_character"))
        if encoding is not None:
            interaction_properties["encoding"] = encoding

        input_file_separator = source_params.get("input_file_separator", ",")
        if input_file_separator != ",":
            file_format = "delimited"
            interaction_properties["field_delimiter"] = input_file_separator
        else:
            file_format = "csv"

    elif '.parquet' in file_name or '.prq' in file_name:
        file_format = 'parquet'

    if file_format is not None:
        interaction_properties["file_format"] = file_format

    return interaction_properties


def modify_details_for_script_and_shiny(details_from_get: dict) -> dict:
    """Add the href and id of and asset to the same position as it is returned from the POST method
    it allows the `get_id`/`get_href` method to work with details returned by GET method.

    :param details_from_get: details of script/shiny app acquired using GET method
    :type details_from_get: dict

    :return: details with 'guid' and 'href' key added to 'metadata'
    :rtype: dict
    """
    try:
        details_from_get[u'metadata'][u'href'] = details_from_get[u'href']
        details_from_get[u'metadata'][u'guid'] = details_from_get[u'metadata'][u'asset_id']
    except KeyError:
        pass

    return details_from_get


def is_lale_pipeline(pipeline):
    return type(pipeline).__module__ == 'lale.operators' and type(pipeline).__qualname__ == 'TrainedPipeline'
