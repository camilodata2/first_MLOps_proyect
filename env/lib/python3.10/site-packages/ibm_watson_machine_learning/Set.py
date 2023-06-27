#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.wml_client_error import  WMLClientError, CannotSetProjectOrSpace
from ibm_watson_machine_learning.instance_new_plan import ServiceInstanceNewPlan


_DEFAULT_LIST_LENGTH = 50


class Set(WMLResource):
    """Set a space_id/project_id to be used in the subsequent actions."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)
        self._ICP = client.ICP

    def default_space(self, space_uid):
        """Set a space ID.

        :param space_uid: UID of the space to be used
        :type space_uid: str

        :return: status ("SUCCESS" if succeeded)
        :rtype: str

        **Example**

        .. code-block:: python

            client.set.default_space(space_uid)

        """
        if self._client.WSD:
            raise WMLClientError(u'Spaces API are not supported in Watson Studio Desktop.')

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_PLATFORM_SPACES:
            space_endpoint = self._client.service_instance._href_definitions.get_platform_space_href(space_uid)
        else:
            space_endpoint = self._client.service_instance._href_definitions.get_space_href(space_uid)

        space_details = requests.get(space_endpoint, headers=self._client._get_headers())

        if space_details.status_code == 404:
            error_msg = "Space with id '{}' does not exist".format(space_uid)
            raise CannotSetProjectOrSpace(reason=error_msg)

        elif space_details.status_code == 200:
            self._client.default_space_id = space_uid
            if self._client.default_project_id is not None:
                print("Unsetting the project_id ...")
            self._client.default_project_id = None
            self._client.project_type = None

            if self._client.CLOUD_PLATFORM_SPACES:
                if 'compute' in space_details.json()['entity'].keys():
                    instance_id = space_details.json()['entity']['compute'][0]['guid']
                    self._client.wml_credentials[u'instance_id'] = instance_id
                    self._client.service_instance = ServiceInstanceNewPlan(self._client)
                    self._client.service_instance.details = self._client.service_instance.get_details()

                else:
                    # Its possible that a previous space is used in the context of
                    # this client which had compute but this space doesn't have
                    self._client.wml_credentials[u'instance_id'] = 'invalid'
                    self._client.service_instance = ServiceInstanceNewPlan(self._client)
                    self._client.service_instance.details = None

            return "SUCCESS"
        else:
            raise CannotSetProjectOrSpace(reason=space_details.text)

    ##Setting project ID
    def default_project(self, project_id):
        """Set a project ID.

        :param project_id: UID of the project
        :type project_id: str

        :return: status ("SUCCESS" if succeeded)
        :rtype: str

        **Example**

        .. code-block:: python

            client.set.default_project(project_id)
        """

        if self._client.ICP and '1.1' == self._client.wml_credentials[u'version'].lower():
            raise WMLClientError(u'Project APIs are not supported in Watson Studio Local. Set space_id for the subsequent actions.')

        if self._client.ICP or self._client.WSD or self._client.CLOUD_PLATFORM_SPACES:
            if project_id is not None:
                self._client.default_project_id = project_id

                if self._client.default_space_id is not None:
                    print("Unsetting the space_id ...")
                self._client.default_space_id = None

                project_endpoint = self._client.service_instance._href_definitions.get_project_href(project_id)
                project_details = requests.get(project_endpoint, headers=self._client._get_headers())
                if project_details.status_code != 200 and project_details.status_code != 204:
                    raise CannotSetProjectOrSpace(reason=project_details.text)
                else:
                    self._client.project_type = project_details.json()['entity']['storage']['type']
                    if self._client.CLOUD_PLATFORM_SPACES:
                        instance_id = "not_found"
                        if 'compute' in project_details.json()['entity'].keys():
                            for comp_obj in project_details.json()['entity']['compute']:
                                if comp_obj['type'] == 'machine_learning':
                                    instance_id = comp_obj['guid']
                                    break
                            self._client.wml_credentials[u'instance_id'] = instance_id
                            self._client.service_instance = ServiceInstanceNewPlan(self._client)
                            self._client.service_instance.details = self._client.service_instance.get_details()
                        else:
                            # Its possible that a previous project is used in the context of
                            # this client which had compute but this project doesn't have
                            self._client.wml_credentials[u'instance_id'] = 'invalid'
                            self._client.service_instance = ServiceInstanceNewPlan(self._client)
                            self._client.service_instance.details = None

                    else:
                        self._client.service_instance = ServiceInstanceNewPlan(self._client)

                    return "SUCCESS"

            else:
                error_msg = "Project id cannot be None."
                raise CannotSetProjectOrSpace(reason=error_msg)
        else:
            self._client.default_project_id = project_id

            if self._client.default_space_id is not None:
                print("Unsetting the space_id ...")
            self._client.default_space_id = None
            
            return "SUCCESS"
