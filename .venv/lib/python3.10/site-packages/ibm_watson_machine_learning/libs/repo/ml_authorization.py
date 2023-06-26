#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import certifi
import json
from ibm_watson_machine_learning.libs.repo.swagger_client.rest import ApiException

try:
    import urllib3
except ImportError:
    raise ImportError('urllib3 is missing')

try:
    # for python3
    from urllib.parse import urlencode
except ImportError:
    # for python2
    from urllib import urlencode

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)


class MLAuthorization(object):
    """
    Class used to authorize user.

    MLAuthorization shouldn't be used alone. It is created to be used as a base class by MLRepositoryClient.
    """
    def __init__(self, api_client):
        self.api_client = api_client
        self.stage_url = "https://iam.stage1.bluemix.net/oidc/token"
        self.prod_url = "https://iam.bluemix.net/oidc/token"


    def _add_header(self, key, value):
        self.api_client.set_default_header(key, value)
        self.api_client.set_default_header(key, value)


    def iam_authorize(self,vcap):
        url         =   vcap.get('url')
        api_key     =   vcap.get('api_key')
        instance_id =   vcap.get('instance_id')
        user        =   vcap.get('username')
        password    =   vcap.get('password')

        if api_key is None:
            if user is None:
                raise TypeError("Watson ML service credentials: username not defined ")
            elif password is None:
                raise TypeError("Watson ML service credentials: password  not defined")
            else:
                #Authorize with ML Token
                self.authorize(user,password)
        elif instance_id is None:
            raise TypeError("Watson ML service credentials: instance_id not defined")
        else:
            #Authorize IAM Token
            self.generate_iam_token(url,api_key,instance_id)


    def get_iam_url(self, base_service_url) :
        path = base_service_url

        paramList = path.split('.')
        if paramList[1] == "stage1":
            iam_url= self.stage_url
        else:
            iam_url= self.prod_url
        return iam_url

    def generate_iam_token(self,base_url, api_key,instance_id):
        oidc_data = {'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
                     'response_type': 'cloud_iam',
                     'apikey': api_key}
        headers = {'Content-Type':'application/x-www-form-urlencoded'}
        url = self.get_iam_url(base_url)

        try:
            response = http.request(
                'POST',
                url,
                fields=oidc_data,
                encode_multipart=False,
                headers=headers
            )
            if response.status == 200:
                iam_token = json.loads(response.data.decode('UTF-8')).get('access_token')
                self.authorize_with_iamtoken(iam_token,instance_id)
            elif response.status == 401:
                raise ApiException(
                    status=401,
                    reason="The incoming request did not contain a valid authentication information"
                )
            elif response.status == 403:
                raise ApiException(
                    status=403,
                    reason="The incoming request did not contain a valid authentication information."
                )
            elif response.status == 400:
                raise ApiException(
                    status=400,
                    reason="Parameter validation failed."
                )
            elif response.status == 500:
                raise ApiException(
                    status=500,
                    reason="Internal Server error."
                )
            else:
                msg = "Unable to authenticate due to error :\n" + "{0}\n{1}".format(response.status, response.text)
                raise ApiException(
                    status=404,
                    reason=msg
                )
        except ApiException as e:
            raise e

        except BaseException as e:
            msg = "Unable to authenticate due to error :\n" + "{0}\n{1}".format(type(e).__name__, str(e))
            raise ApiException(
                status=404,
                reason=msg
            )

    def authorize(self, user, password):
        """
        Authorizes user by username and password

        Both username and password should be credentials taken from Watson Machine Learning instance.

        :param str user: username taken from WML instance
        :param str password: password corresponding with user name taken from WML instance

        """
        headers = urllib3.util.make_headers(basic_auth='{}:{}'.format(user, password))
        url = '{}/v2/identity/token'.format(self.api_client.repository_path)
        try:
            response = http.request(
                      'GET',
                      url,
                      headers=headers
                     )
            if response.status == 200:
                token = json.loads(response.data.decode('UTF-8')).get('token')
                self.authorize_with_token(token)
            elif response.status == 401:
                raise ApiException(
                    status=404,
                    reason="Invalid credentials. Authorization failed."
                    )
            elif response.status == 403:
                raise ApiException(
                    status=404,
                    reason="Authentication failed due to insufficient permission"
                    )
            else:
                msg = "Unable to authenticate due to error :\n" + "{0}\n{1}".format(response.status, response.text)
                raise ApiException(
                    status=404,
                    reason=msg
                   )
        except ApiException as e:
            raise e

        except BaseException as e:
            msg = "Unable to authenticate due to error :\n" + "{0}\n{1}".format(type(e).__name__, str(e))
            raise ApiException(
                status=404,
                reason=msg
            )

    def authorize_with_token(self, token):
        """
            Authorizes user by token.

            Token should be generated during basic authentication using username and password from WML service instance.

            :param str token: token

            """
        self._add_header('Authorization', 'Bearer {}'.format(token))


    def authorize_with_iamtoken(self, iamtoken,instance_id, platform_spaces=False):

        self._add_header('Authorization', 'Bearer {}'.format(iamtoken))
        # Cloud Convergence
        if not platform_spaces:
           self._add_header('ML-Instance-ID', instance_id)
