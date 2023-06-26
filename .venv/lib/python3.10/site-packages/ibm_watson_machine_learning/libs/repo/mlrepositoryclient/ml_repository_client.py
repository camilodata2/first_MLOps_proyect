#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from .model_collection import ModelCollection
from .experiment_collection import ExperimentCollection
from .wml_experiment_collection import WmlExperimentCollection

from .function_collection import FunctionCollection
from .libraries_collection import LibrariesCollection
from ibm_watson_machine_learning.libs.repo.ml_api_client import MLApiClient
from ibm_watson_machine_learning.libs.repo.ml_authorization import MLAuthorization
from ibm_watson_machine_learning.libs.repo.mlrepository.meta_props import MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepositoryclient.ml_repository_api import MLRepositoryApi
from .runtimes_collection import RuntimesCollection
import sys
if sys.version_info < (3, 0):
    from urlparse import urlparse
else:
    from urllib.parse import urlparse


class MLRepositoryClient(MLAuthorization):
    """
    Main class used as client to connect to Repository sevice.

    :param str base_path: base url to Watson Machine Learning instance
    :param MLApiClient api_client: customized api_client provided by user

    :ivar ModelCollection models: provides client to operate on models

    """
    def __init__(self, base_path=None, api_client=None):

        if base_path is not None:
            if api_client is None:
                api_client = MLApiClient(base_path)

            if not isinstance(base_path, str) and not isinstance(base_path, unicode):
                raise ValueError('Invalid type for base_path: {}'.format(base_path.__class__.__name__))

            if not isinstance(api_client, MLApiClient):
                raise ValueError('Invalid type for api_client: {}'.format(api_client.__class__.__name__))

            self.repository_api = MLRepositoryApi(api_client)
            super(MLRepositoryClient, self).__init__(api_client)

            self.models = ModelCollection(base_path, self.repository_api, self)
            self.pipelines = ExperimentCollection(base_path, self.repository_api, self)
            self.experiments = WmlExperimentCollection(base_path, self.repository_api, self)
            self.functions = FunctionCollection(base_path, self.repository_api, self)
            self.libraries = LibrariesCollection(base_path, self.repository_api, self)
            self.runtimes = RuntimesCollection(base_path, self.repository_api, self)

        else:
            self.repository_api = None
            super(MLRepositoryClient, self).__init__(api_client)
            self.models = None
            self.pipelines = None
            self.experiments = None
            self.functions = None


    def iam_connect(self, wml_vcap):

        if not isinstance(wml_vcap, dict):
            raise TypeError("Expecting object of type : %s" % dict+ " but got %s" % type(wml_vcap))

        base_path = wml_vcap.get('url')
        if base_path is None:
            raise TypeError("Watson ML service credentials: url not defined")

        parsed = urlparse(base_path)
        if parsed.scheme == '' or parsed.netloc == '':
            raise TypeError("Watson ML service credentials: Invalid URL")
        api_client = MLApiClient(base_path)
        self.repository_api = MLRepositoryApi(api_client)
        self.api_client = api_client
        self.models = ModelCollection(base_path, self.repository_api, self)
        self.pipelines = ExperimentCollection(base_path, self.repository_api, self)
        self.experiments = WmlExperimentCollection(base_path, self.repository_api, self)
        self.functions = FunctionCollection(base_path, self.repository_api, self)

        self.iam_authorize(wml_vcap)


    @staticmethod
    def meta(self):
        """returns meta object

    >>> model_artifact = MLRepositoryArtifact(model, name='test-model-name', training_data=training, meta_props=MetaProps({
    >>> MetaNames.EVALUATION_METRICS: json.dumps([{
    >>>   "name": "accuracy",
    >>>    "value": 0.95,
    >>>    "threshold": 0.9}]) }))
    >>> created_model_artifact = ml_repository_client.models.save(model_artifact)
    >>> created_model_metaprops = created_model_artifact.meta.prop(MetaNames.EVALUATION_METRICS)

        """
        return MetaProps({})

def connect(vcap):
    """
    Authorizes user to connect by providing the vcap

   :param dict vcap: WML Credentials with API key or WML credentials

   """

    if not isinstance(vcap, dict):
        raise TypeError("Expecting object of type : %s" % dict+ " but got %s" % type(vcap))
    iam_client = MLRepositoryClient()
    iam_client.iam_connect(vcap)
    return iam_client
