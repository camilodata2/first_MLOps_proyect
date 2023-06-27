#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import ibm_watson_machine_learning._wrappers.requests as requests
import json
import base64
import logging
from datetime import datetime, timedelta
from ibm_watson_machine_learning.wml_client_error import NoWMLCredentialsProvided, ApiRequestFailure, WMLClientError, \
                                                         CannotAutogenerateBedrockUrl
from ibm_watson_machine_learning.href_definitions import HrefDefinitions


class ServiceInstanceNewPlan:
    """Connect, get details and check usage of Watson Machine Learning service instance."""

    def __init__(self, client):
        self._logger = logging.getLogger(__name__)
        self._client = client
        self._wml_credentials = client.wml_credentials
        self._expiration_datetime = None

        if self._client.ICP_PLATFORM_SPACES:
            if self.get_instance_id() == 'openshift' or self.get_instance_id() == 'wml_local':
                self._wml_credentials[u'url'] = self.get_url()
            else:
                self._wml_credentials[u'url'] = self.get_url() + ':31843'
            # TODO: Check if this is used anywhere.. from initial searches, doesn't seem like
            # self._wml_credentials[u'instance_id'] = "999"

        # This is used in connections.py
        self._href_definitions = HrefDefinitions(self._client,
                                                 self._client.CLOUD_PLATFORM_SPACES,
                                                 self._client.PLATFORM_URL,
                                                 self._client.CAMS_URL,
                                                 self._client.ICP_PLATFORM_SPACES)

        self._client.wml_token = self._create_token()
        # self._logger.info(u'Successfully prepared token: ' + self._client.wml_token)
        # ml_repository_client is initialized in repo
        self.details = None

    def get_instance_id(self):
        """Get instance id of Watson Machine Learning service.

        :return: instance id
        :rtype: str

        **Example**

        .. code-block:: python

            instance_details = client.service_instance.get_instance_id()
        """
        if self._wml_credentials['instance_id'] == 'invalid':
            raise WMLClientError('instance_id for this plan is picked up from the space with which'
                                 'this instance_id is associated with. Set the space with associated'
                                 'instance_id to be able to use this function')
        return self._wml_credentials['instance_id']

    def get_api_key(self):
        """Get api key of Watson Machine Learning service.

        :return: api key
        :rtype: str

        **Example**

        .. code-block:: python

            instance_details = client.service_instance.get_api_key()
        """
        return self._wml_credentials['apikey']

    def get_url(self):
        """Get instance url of Watson Machine Learning service.

        :return: instance url
        :rtype: str

        **Example**

        .. code-block:: python

            instance_details = client.service_instance.get_url()
        """
        return self._wml_credentials['url']

    def get_username(self):
        """Get username for Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data.

        :return: username
        :rtype: str

        **Example**

        .. code-block:: python

            instance_details = client.service_instance.get_username()
        """
        if self._client.ICP_PLATFORM_SPACES:
            try:
                return self._wml_credentials['username']
            except:
                raise WMLClientError('`username` missing in wml_credentials.')
        else:
            raise WMLClientError("Not applicable for Cloud")

    def get_password(self):
        """Get password for Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data.

        :return: password
        :rtype: str

        **Example**

        .. code-block:: python

            instance_details = client.service_instance.get_password()
        """
        if self._client.ICP_PLATFORM_SPACES:
            try:
                return self._wml_credentials['password']
            except:
                raise WMLClientError('`password` missing in wml_credentials.')
        else:
            raise WMLClientError("Not applicable for Cloud")

    def get_details(self):
        """Get information about Watson Machine Learning instance.

        :return: metadata of service instance
        :rtype: dict

        **Example**

        .. code-block:: python

            instance_details = client.service_instance.get_details()
        """

        if not self._client.ICP:
            if self._wml_credentials is not None:

                if self._wml_credentials['instance_id'] == 'invalid':
                    raise WMLClientError('instance_id for this plan is picked up from the space with which '
                                         'this instance_id is associated with. Set the space with associated '
                                         'instance_id to be able to use this function')

                    # /ml/v4/instances will need either space_id or project_id as mandatory params
                # We will enable this service instance class only during create space or
                # set space/project. So, space_id/project_id would have been populated at this point
                headers = self._client._get_headers()

                del headers[u'X-WML-User-Client']
                if 'ML-Instance-ID' in headers:
                    headers.pop('ML-Instance-ID')
                headers.pop(u'x-wml-internal-switch-to-new-v4')
                # params = {'version': self._client.version_param}
                response_get_instance = requests.get(
                    self._href_definitions.get_v4_instance_id_href(),
                    params=self._client._params(skip_space_project_chk=True),
                    # params={'version': self._client.version_param},
                    headers=headers
                )

                if response_get_instance.status_code == 200:
                    return response_get_instance.json()
                else:
                    raise ApiRequestFailure(u'Getting instance details failed.', response_get_instance)
            else:
                raise NoWMLCredentialsProvided
        else:
            return {}

    def _get_token(self):
        if self._client.wml_token is None:
            self._create_token()
            self._client.repository._refresh_repo_client()
        elif self._is_token_refresh_possible():
            if self._client.ICP:
                if self._get_expiration_datetime():
                    if self._get_expiration_datetime() - timedelta(minutes=50) < datetime.now():
                        self._client.wml_token = self._get_cpd_token_from_request()
                        self._client.repository._refresh_repo_client()
                else:
                    self._client.wml_token = self._get_cpd_token_from_request()
                    self._client.repository._refresh_repo_client()
            elif self._client._is_IAM():
                if self._get_expiration_datetime() - timedelta(minutes=15) < datetime.now():
                    self._client.wml_token = self._get_IAM_token()
                    self._client.repository._refresh_repo_client()
            elif self._get_expiration_datetime() - timedelta(minutes=30) < datetime.now():
                self._client.repository._refresh_repo_client()
                self._refresh_token()

        return self._client.wml_token

    def _create_token(self):

        if self._client.proceed is True:
            return self._wml_credentials["token"]

        if not self._client.ICP_PLATFORM_SPACES:
            if self._client._is_IAM():
                return self._get_IAM_token()
            else:
                raise WMLClientError('apikey for IAM token is not provided in credentials for the client.')
        else:
            return self._get_cpd_token_from_request()

    def _refresh_token(self):
        if self._client.proceed is True:
            self._client.wml_token = self._wml_credentials["token"]

        self._client.wml_token = self._get_cpd_token_from_request()

    def _get_expiration_datetime(self):
        if self._expiration_datetime is not None:
            return self._expiration_datetime

        token_parts = self._client.wml_token.split('.')
        token_padded = token_parts[1] + '=' * (len(token_parts[1]) % 4)
        token_info = json.loads(base64.b64decode(token_padded).decode('utf-8'))
        token_expire = token_info.get('exp')

        return datetime.fromtimestamp(token_expire)

    def _is_iam(self):
        try:
            token_parts = self._client.wml_token.split('.')
            token_padded = token_parts[1] + '=' * (len(token_parts[1]) % 4)
            token_info = json.loads(base64.b64decode(token_padded).decode('utf-8'))
            instanceId = token_info.get('instanceId')

            return instanceId
        except:
            return False

    def _get_IAM_token(self):
        if self._client.proceed is True:
            return self._wml_credentials["token"]
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic Yng6Yng='
        }

        mystr = 'apikey=' + self._href_definitions.get_iam_token_api()
        response = requests.post(
            self._href_definitions.get_iam_token_url(),
            data=mystr,
            headers=headers
        )

        if response.status_code == 200:
            token = response.json().get(u'access_token')
            self._expiration_datetime = None
        else:
            raise WMLClientError(u'Error getting IAM Token.', response)
        return token

    def _is_token_refresh_possible(self):
        """Checks if necessary credentials were passed for token refresh.
        For CP4D we need (username & password)/(username & api_key).
        For Cloud we need api_key.

        :return: `True` if token refresh can be performed `False` otherwise
        :rtype: bool
        """
        if self._client._is_IAM():
            return 'apikey' in self._wml_credentials
        else:
            return 'username' in self._wml_credentials and (
                    'password' in self._wml_credentials or 'apikey' in self._wml_credentials)

    def _get_cpd_auth_pair(self):
        """Get a pair of credentials required for generation of token.

        :return: string representing a dictionary of authentication credentials
                         (username & password) or (username & api_key).
        :rtype: str
        """
        if "apikey" in self._wml_credentials:
            return f'{{\"username\": \"{self.get_username()}\", \"api_key\": \"{self.get_api_key()}\"}}'
        else:
            return f'{{\"username\": \"{self.get_username()}\", \"password\": \"{self.get_password()}\"}}'

    def _get_cpd_bedrock_auth_data(self):
        """Get data required for generation of token.

        :return: string representing a dictionary of authentication credentials
        :rtype: str
        """
        return f'grant_type=password&username={self.get_username()}&password={self.get_password()}&scope=openid'

    def _get_cpd_token_from_request_old_auth_flow(self):
        token_url = self._href_definitions.get_cpd_token_endpoint_href()
        response = requests.post(token_url,
                                 headers={
                                     'Content-Type': 'application/json'
                                 },
                                 data=self._get_cpd_auth_pair())

        if response.status_code == 200:
            self._expiration_datetime = None
            return response.json().get(u'token')
        else:
            raise WMLClientError(u'Error refreshing the token.', response)

    def _get_cpd_token_from_request_new_auth_flow(self):
        bedrock_url = self._href_definitions.get_cpd_bedrock_token_endpoint_href()
        response = requests.post(bedrock_url,
                                 headers={
                                     'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
                                 },
                                 data=self._get_cpd_bedrock_auth_data())

        if response.status_code != 200:
            raise WMLClientError(u'Error refreshing the token.', response, logg_messages=False)

        iam_token = response.json()['access_token']
        self._expiration_datetime = datetime.now() + timedelta(seconds=response.json()['expires_in'])
        # refresh_token = response.json()['refresh_token']

        token_url = self._href_definitions.get_cpd_validation_token_endpoint_href()
        response = requests.get(token_url,
                                headers={
                                    'username': self.get_username(),
                                    'iam-token': iam_token
                                })

        if response.status_code == 200:
            return response.json()['accessToken']
        else:
            raise WMLClientError(u'Error refreshing the token.', response)

    def _get_cpd_token_from_request(self):
        """Send a request for token on CPD.

        :return: newly created token is returned if no errors occurred
        :rtype: str
        """
        if (self._client.ICP_40 or self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47) and 'bedrock_url' in self._client.wml_credentials and 'password' in self._client.wml_credentials:
            try:
                return self._get_cpd_token_from_request_new_auth_flow()
            except Exception as e1:
                if not hasattr(self._client, '_is_bedrock_url_autogenerated'):
                    raise e1

                try:
                    res = self._get_cpd_token_from_request_old_auth_flow()
                    # if it worked then iamintegration=False, then removing bedrock_url will shorten the path
                    del self._client.wml_credentials['bedrock_url']
                    return res
                except Exception as e2:
                    if hasattr(self._client, '_is_bedrock_url_autogenerated') and self._client._is_bedrock_url_autogenerated:
                        raise CannotAutogenerateBedrockUrl(e1, e2)
                    else:
                        raise e2
        else:
            return self._get_cpd_token_from_request_old_auth_flow()
