#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from warnings import warn

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.wml_client_error import MissingArgument, WrongEnvironmentVersion
from ..utils.autoai.enums import TShirtSize
from ..utils.autoai.errors import TShirtSizeNotSupported, SetIDFailed
from ..utils.autoai.utils import is_ipython


class WorkSpace:
    """WorkSpace class for WML authentication and project/space manipulation."""

    def __init__(self, wml_credentials: dict, project_id: str = None, space_id: str = None, verify=None) -> None:
        """
        :param wml_credentials: credentials to Watson Machine Learning instance
        :type wml_credentials: dict

        :param project_id: ID of the Watson Studio project
        :type project_id: str, optional

        :param space_id: ID of the Watson Studio Space
        :type space_id: str, optional

        **Example**

        .. code-block: python

            from ibm_watson_machine_learning.workspace import WorkSpace

            ws = WorkSpace(
                wml_credentials={
                    "apikey": "...",
                    "iam_apikey_description": "...",
                    "iam_apikey_name": "...",
                    "iam_role_crn": "...",
                    "iam_serviceid_crn": "...",
                    "instance_id": "...",
                    "url": "https://us-south.ml.cloud.ibm.com"
                },
                project_id="...",
                space_id="...")
        """
        self.wml_credentials = wml_credentials.copy()
        self.project_id = project_id
        self.space_id = space_id
        self.WMLS = False

        # note: base client changed versioning for WMLS
        if wml_credentials.get('instance_id', '').lower() == 'wml_local':
            supported_versions = ('2.0')
            if wml_credentials[u'version'] not in supported_versions:
                raise WrongEnvironmentVersion(wml_credentials[u'version'], wml_credentials[u'instance_id'].lower(),
                                              supported_versions)

            else:
                self.wml_credentials['version'] = '3.0.0'
        # --- end note

        self.wml_client = APIClient(wml_credentials=self.wml_credentials, verify=verify)

        if not wml_credentials.get('instance_id'):
            if self.space_id is not None:
                self.wml_client.set.default_space(self.space_id)

            elif self.project_id is not None:
                self.wml_client.set.default_project(self.project_id)

            else:
                raise SetIDFailed(
                    f'project_id and space_id',
                    reason=f'project_id and space_id cannot be None at the same time.')

        elif wml_credentials.get('instance_id', '').lower() == 'wml_local':
            print("Connecting to WML Server...")
            if self.space_id is None:
                raise MissingArgument(
                    'space_id',
                    reason="These credentials are from WML Server environment, "
                           "please specify the \"space_id\"")
            else:
                outcome = self.wml_client.set.default_space(self.space_id)
                if outcome == 'FAILURE':
                    raise SetIDFailed(
                        'space_id',
                        reason=f"This space_id: {self.space_id} cannot be found in current environment.")
                self.WMLS = True

        elif wml_credentials.get('instance_id', '').lower() in ('icp', 'openshift'):
            if self.project_id is None and self.space_id is None:
                raise MissingArgument(
                    'project_id',
                    reason="These credentials are from CP4D environment, "
                           "please specify \"project_id\"")
            else:
                if self.project_id is not None:
                    outcome = self.wml_client.set.default_project(self.project_id)

                else:
                    outcome = self.wml_client.set.default_space(self.space_id)

                if outcome == 'FAILURE':
                    raise SetIDFailed(
                        f'{"project_id" if self.project_id is not None else "space_id"}',
                        reason=f'This {"project_id" if self.project_id is not None else "space_id"}: '
                               f'{self.project_id if self.project_id is not None else self.space_id} '
                               f'cannot be found in current environment.')

            supported_versions = ('3.0.0', '3.0.1', '3.5', '4.0', '4.5', '4.6', '4.7')
            if wml_credentials[u'version'] not in supported_versions:
                raise WrongEnvironmentVersion(wml_credentials[u'version'], wml_credentials[u'instance_id'].lower(),
                                              supported_versions)

    def __str__(self):
        return f'wml_credentials: {self.wml_credentials} project_id: {self.project_id} space_id = {self.space_id}'

    def __repr__(self):
        return self.__str__()

    def restrict_pod_size(self, t_shirt_size: 'TShirtSize') -> 'TShirtSize':
        """Check t_shirt_size for AutoAI POD. Restrict sizes per environment.

        :param t_shirt_size: TShirt size to be validated and restricted
        :type t_shirt_size: TShirtSize

        :return: validated and restricted TShirt size
        :rtype: TShirtSize
        """
        # note: for testing purposes
        if self.wml_credentials.get('development', False):
            return t_shirt_size
        # --- end note

        default_cloud = TShirtSize.L
        default_cp4d = TShirtSize.M
        default_server = TShirtSize.M

        supported_cp4d = (TShirtSize.S, TShirtSize.M, TShirtSize.L, TShirtSize.XL)
        supported_server = (TShirtSize.S, TShirtSize.M, TShirtSize.L)

        # note: check CP4D and Server pod sizes
        if self.wml_client.ICP:
            if self.WMLS and t_shirt_size not in supported_server:
                message = (f"This t-shirt-size: \"{t_shirt_size}\" is not supported in WMLS. "
                           f"Supported sizes: {supported_server} "
                           f"Continuing work with default size {default_server}")

                warn(message=message)
                if is_ipython():
                    print(message)

                return default_server

            elif t_shirt_size not in supported_cp4d:
                message = (f"This t-shirt-size: \"{t_shirt_size}\" is not supported in CP4D. "
                           f"Supported sizes: {supported_cp4d} "
                           f"Continuing work with default size {default_cp4d}")

                warn(message=message)
                if is_ipython():
                    print(message)

                return default_cp4d

            else:
                return t_shirt_size

        else:
            # note: allow every size in test envs
            if 'test' in self.wml_credentials.get('url', ''):
                return t_shirt_size

            else:
                # note: raise an error for cloud if pod size is different, other just return
                if t_shirt_size != default_cloud:
                    raise TShirtSizeNotSupported(
                        t_shirt_size,
                        reason=f"This t-shirt size is not supported. Please use the default one: {default_cloud}")

                else:
                    return default_cloud
                # --- end note
