#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
from .utils import *

wml_credentials = get_wml_credentials()

if 'flight_url' in wml_credentials:
    url_parts = wml_credentials['flight_url'].replace('https://', '').replace('http://', '').split(':')
    location = url_parts[0]
    port = url_parts[1] if len(url_parts) > 1 else '443'

    if "FLIGHT_SERVICE_LOCATION" not in os.environ:
        os.environ['FLIGHT_SERVICE_LOCATION'] = location

    if "FLIGHT_SERVICE_PORT" not in os.environ:
        os.environ['FLIGHT_SERVICE_PORT'] = port
