#  (C) Copyright IBM Corp. 2023.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure

_DEFAULT_LIST_LENGTH = 50


class TaskCredentials(WMLResource):
    """Store and manage your task credentials."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

    def get_details(self, task_credentials_uid=None, project_id=None, space_id=None):
        """Get task credentials details. If no task_credentials_uid is passed, details for all task credentials
        will be returned.

        :param task_credentials_uid: UID of task credentials to be fetched
        :type task_credentials_uid: str, optional

        :param project_id: UID of project to be used for filtering
        :type project_id: str, optional

        :param space_id: UID of space to be used for filtering
        :type space_id: str, optional

        :return: created task credentials details
        :rtype: dict (if task_credentials_uid is not None) or {"resources": [dict]} (if task_credentials_uid is None)

        **Example**

        .. code-block:: python

            task_credentials_details = client.task_credentials.get_details(task_credentials_uid)

        """
        if self._ICP:
            raise WMLClientError(u'Task Credentials API is supported on Cloud only.')

        # TaskCredentials._validate_type(task_credentials_uid, u'task_credentials_uid', STR_TYPE, False)

        if task_credentials_uid:
            response = requests.get(
                self._client.service_instance._href_definitions.get_task_credentials_href(task_credentials_uid),
                headers=self._client._get_headers())

            return self._handle_response(200, u'get task credentials details', response)
        else:
            params = {}

            if project_id:
                params['scope.project_id'] = project_id
            elif space_id:
                params['scope.space_id'] = space_id

            response = requests.get(self._client.service_instance._href_definitions.get_task_credentials_all_href(),
                                    params=params,
                                    headers=self._client._get_headers())

            return {'resources': self._handle_response(200, u'get task credentials details',
                                                       response).get('credentials', {})}

    def store(self, project_id=None, space_id=None):
        """Store current credentials using Task Credentials API to use with long run tasks. Supported only on Cloud.

        :param project_id: UID of project which become a scope for saved credentials
        :type project_id: str, optional

        :param space_id: UID of space which become a scope for saved credentials
        :type space_id: str, optional

        :return: metadata of the stored task credentials
        :rtype: dict

        **Example**

        .. code-block:: python

            task_credentials_details = client.task_credentials.store()

        """
        if self._ICP:
            raise WMLClientError(u'Task Credentials API is supported on Cloud only.')

        href = self._client.service_instance._href_definitions.get_task_credentials_all_href()

        scope = {}

        if project_id:
            scope['project_id'] = project_id
        elif space_id:
            scope['space_id'] = space_id

        creation_response = requests.post(href,
                                          params=self._client._params(),
                                          headers=self._client._get_headers(),
                                          json={"name": "Python API generated task credentials",
                                                "description": "Python API generated task credentials.",
                                                "type": "iam_api_key",
                                                "scope": scope,
                                                "secret": {
                                                    "api_key": self._client.wml_credentials['apikey']
                                                }})

        return self._handle_response(201, u'creating task credentials', creation_response)

    def list(self, limit=None, project_id=None, space_id=None, return_as_df=True):
        """Print task credentials in table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param project_id: UID of project to be used for filtering
        :type project_id: str, optional

        :param space_id: UID of space to be used for filtering
        :type space_id: str, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed assets or None if return_as_df is False
        :rtype: pandas.DataFrame or None


        **Example**

        .. code-block:: python

            client.task_credentials.list()

        """
        if self._ICP:
            raise WMLClientError(u'Task Credentials API is supported on Cloud only.')

        details = self.get_details(project_id=project_id, space_id=space_id)

        task_credentials_details = details["resources"]
        task_credentials_values = [
            (m[u'name'], m[u'id'], m[u'scope']) for
            m in task_credentials_details]

        table = self._list(task_credentials_values, [u'NAME', u'ASSET_ID', u'TYPE'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    @staticmethod
    def get_id(task_credentials_details):
        """Get Unique Id of task credentials.

        :param task_credentials_details: metadata of the task credentials
        :type task_credentials_details: dict

        :return: Unique Id of task credentials
        :rtype: str

        **Example**

        .. code-block:: python

            task_credentials_uid = client.task_credentials.get_id(task_credentials_details)

        """
        return task_credentials_details['id']

    def delete(self, task_credentials_uid):
        """Delete a software specification.

        :param task_credentials_uid: Unique Id of task credentials
        :type task_credentials_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.task_credentials.delete(task_credentials_uid)

        """
        if self._ICP:
            raise WMLClientError(u'Task Credentials API is supported on Cloud only.')

        TaskCredentials._validate_type(task_credentials_uid, u'task_credentials_uid', str, True)

        response = requests.delete(self._client.service_instance._href_definitions.get_task_credentials_href(task_credentials_uid),
                                   params=self._client._params(),
                                   headers=self._client._get_headers())

        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, u'delete task credentials', response)
