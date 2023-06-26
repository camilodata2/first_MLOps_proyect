#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json
import os
from configparser import ConfigParser
from typing import Union, Dict

from ..utils.autoai.watson_studio import get_project, get_wmls_credentials_and_space_ids

__all__ = [
    "get_credentials_from_config",
    "pipeline_to_script",
    'get_wmls_configuration'
]


def get_credentials_from_config(env_name, credentials_name, config_path="./config.ini"):
    """Load credentials from config file.

    ::

        [DEV_LC]

        wml_credentials = { }
        cos_credentials = { }

    :param env_name: the name of [ENV] defined in config file
    :type env_name: str
    :param credentials_name: name of credentials
    :type credentials_name: str
    :param config_path: path to the config file
    :type config_path: str
    :return: loaded credentials
    :rtype: dict

    **Example**

    .. code-block:: python

        get_credentials_from_config(env_name='DEV_LC', credentials_name='wml_credentials')

    """
    config = ConfigParser()
    config.read(config_path)

    return json.loads(config.get(env_name, credentials_name))


def pipeline_to_script(pipeline) -> Union['str', 'HTML']:
    """Create a python script based on a passed pipeline model. (Python code representation of pipeline model)

    :param pipeline: pipeline model to be written as script
    :type pipeline: Pipeline or TrainedPipeline

    :return: information about script location
    :rtype: str or html

    **Example**

    .. code-block:: python

        pipeline_to_script(pipeline=best_pipeline)
    """
    from lale.helpers import import_from_sklearn_pipeline
    from sklearn.pipeline import Pipeline
    from ibm_watson_machine_learning.utils.autoai.utils import is_ipython
    from ibm_watson_machine_learning.utils import create_download_link
    import os
    script_name = "pipeline_script.py"

    if isinstance(pipeline, Pipeline):
        pipeline = import_from_sklearn_pipeline(pipeline)

    script = pipeline.pretty_print()

    with open(script_name, 'w') as f:
        f.write(script)

    script_location = f"{os.path.abspath('.')}/{script_name}"

    if is_ipython():
        return create_download_link(script_location)
    else:
        return f"Pipeline python script location: {script_location}"


def get_wmls_configuration() -> Dict[str, Union[Dict, None, str]]:
    """Try to find credentials and space_ids on Watson Studio Desktop automatically.

    :return: list of dictionaries with wml_credentials, project_id, and space_id
    :rtype: dict
    """
    project = get_project()
    project_id = project.get_metadata()["metadata"]["guid"]

    path_to_wmls_credentials = f"{os.path.abspath('.')}/{project_id}/project.json"
    credentials, space_ids = get_wmls_credentials_and_space_ids(path_to_wmls_credentials)

    found_data = [{'wml_credentials': creds,
                   'project_id': None,
                   'space_id': space_id} for creds, space_id in zip(credentials, space_ids)]

    return found_data[0]
