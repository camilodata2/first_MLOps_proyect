#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

__all__ = [
    'get_project',
    'get_wmls_credentials_and_space_ids'
]


import base64
import json
from typing import Any, Dict, Tuple, List, TYPE_CHECKING

import ibm_watson_machine_learning._wrappers.requests as requests

from ibm_watson_machine_learning.href_definitions import IAM_TOKEN_API, HrefDefinitions
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from .errors import NotInWatsonStudio, CredentialsNotFound

if TYPE_CHECKING:
    from ibm_watson_machine_learning.workspace import WorkSpace


def get_project() -> Any:
    """Try to import project_lib and get user corresponding project."""
    try:
        from project_lib import Project

    except ModuleNotFoundError:
        raise NotInWatsonStudio(reason="You are not in Watson Studio or Watson Studio Desktop environment. "
                                       "Cannot access to project metadata.")

    try:
        access = Project.access()

    except RuntimeError:
        raise CredentialsNotFound(reason="Your WSD environment does not have correctly configured "
                                         "connection to WML Server or you are not in WSD environment. "
                                         "In that case, please provide WMLS credentials and space_id.")

    return access


def get_wmls_credentials_and_space_ids(local_path: str) -> Tuple[List[Dict], List[str]]:
    """Parse project.json file and get WMLS credentials associated with WSD project.

    :param local_path: path to project.json file with project configuration in WSD
    :type local_path: str

    :return: two lists with WMLS credentials and corresponding space_ids
    :rtype: tuple[list[dict], list[str]]
    """
    try:
        with open(local_path, 'r') as f:
            data = json.load(f)

        credentials = [instance['credentials'] for instance in data['compute']]
        for instance in credentials:
            instance['version'] = "2.0"
            instance['instance_id'] = "wml_local"
            instance['password'] = base64.decodebytes(bytes(instance['password'].encode())).decode()

        space_ids = [instance['properties']['space_guid'] for instance in data['compute']]

    except (FileNotFoundError, KeyError):
        raise CredentialsNotFound(reason="Your WSD environment does not have correctly configured "
                                         "connection to WML Server or you are not in WSD environment. "
                                         "In that case, please provide WMLS credentials and space_id.")

    return credentials, space_ids
