#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
__all__ = [
    'is_run_id_exists'
]

from typing import Dict, Optional

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure


def is_run_id_exists(wml_credentials: Dict, run_id: str, space_id: Optional[str] = None) -> bool:
    """Check if specified run_id exists for WML client initialized with passed credentials.

    :param wml_credentials: WML Service Instance credentials
    :type wml_credentials: dict

    :param run_id: training run id of AutoAI experiment
    :type run_id: str

    :param space_id: optional space id for WMLS and CP4D
    :type space_id: str, optional
    """
    client = APIClient(wml_credentials)

    if space_id is not None:
        client.set.default_space(space_id)

    try:
        client.training.get_details(run_id, _internal=True)

    except ApiRequestFailure as e:
        if 'Status code: 404' in str(e):
            return False

        else:
            raise e

    return True
