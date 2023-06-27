#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import print_function
import ibm_watson_machine_learning._wrappers.requests as requests
from ibm_watson_machine_learning.utils import PIPELINE_DETAILS_TYPE
from ibm_watson_machine_learning.metanames import PipelineMetanames
from ibm_watson_machine_learning.wml_resource import WMLResource
from ibm_watson_machine_learning.href_definitions import API_VERSION, SPACES, PIPELINES, LIBRARIES, EXPERIMENTS, RUNTIMES, DEPLOYMENTS
from ibm_watson_machine_learning.wml_client_error import WMLClientError
_DEFAULT_LIST_LENGTH = 50
import json


class Pipelines(WMLResource):
    """Store and manage pipelines."""

    ConfigurationMetaNames = PipelineMetanames()
    """MetaNames for pipelines creation."""

    def __init__(self, client):
        WMLResource.__init__(self, __name__, client)

        self.WSD = client.WSD
        self._ICP = client.ICP

    def _generate_pipeline_document(self, meta_props):
        if self._client.ICP:
            doc = {
                "doc_type": "pipeline",
                "version": "2.0",
                "primary_pipeline": "wmla_only",
                "pipelines": [
                    {
                        "id": "wmla_only",
                        "runtime_ref": "hybrid",
                        "nodes": [
                            {
                                "id": "training",
                                "type": "model_node",
                                "op": "dl_train",
                                "runtime_ref": "DL_WMLA",
                                "inputs": [
                                ],
                                "outputs": [],
                                "parameters": {
                                    "name": "pipeline",
                                    "description": "Pipeline - Python client"
                                }
                            }
                        ]
                    }
                ],
                "schemas": [
                    {
                        "id": "schema1",
                        "fields": [
                            {
                                "name": "text",
                                "type": "string"
                            }
                        ]
                    }
                ]
            }
        else:
            doc = {
                "doc_type": "pipeline",
                "version": "2.0",
                "primary_pipeline": "dlaas_only",
                "pipelines": [
                  {
                    "id": "dlaas_only",
                    "runtime_ref": "hybrid",
                    "nodes": [
                      {
                        "id": "training",
                        "type": "model_node",
                        "op": "dl_train",
                        "runtime_ref": "DL",
                        "inputs": [
                        ],
                        "outputs": [],
                        "parameters": {
                          "name": "tf-mnist",
                          "description": "Simple MNIST model implemented in TF"
                        }
                      }
                    ]
                  }
                ],
                "schemas": [
                  {
                    "id": "schema1",
                    "fields": [
                      {
                        "name": "text",
                        "type": "string"
                      }
                    ]
                  }
                ]
              }
        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
            if self.ConfigurationMetaNames.COMMAND in meta_props:
                doc["pipelines"][0]["nodes"][0]["parameters"]["command"] = meta_props[self.ConfigurationMetaNames.COMMAND]
            if self.ConfigurationMetaNames.RUNTIMES in meta_props:
                doc["runtimes"] = meta_props[self.ConfigurationMetaNames.RUNTIMES]
                if self._client.ICP:
                    doc["runtimes"][0]["id"] = "DL_WMLA"
                else:
                    doc["runtimes"][0]["id"] = "DL"
            if self.ConfigurationMetaNames.LIBRARY_UID in meta_props:
                if self._client.CAMS:
                    type_uid = self._check_if_lib_or_def(meta_props[self.ConfigurationMetaNames.LIBRARY_UID])
                    doc["pipelines"][0]["nodes"][0]["parameters"]["training_lib_href"] = type_uid
                else:
                    doc["pipelines"][0]["nodes"][0]["parameters"]["training_lib_href"] = "/v4/libraries/" + meta_props[
                        self.ConfigurationMetaNames.LIBRARY_UID]
            if self.ConfigurationMetaNames.COMPUTE in meta_props:
                doc["pipelines"][0]["nodes"][0]["parameters"]["compute"] = meta_props[self.ConfigurationMetaNames.COMPUTE]
            return doc
        else:
            return doc

    def store(self, meta_props, **kwargs):
        """Create a pipeline.

        :param meta_props: meta data of the pipeline configuration. To see available meta names use:

            .. code-block:: python

                client.pipelines.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: stored pipeline metadata
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.pipelines.ConfigurationMetaNames.NAME: 'my_training_definition',
                client.pipelines.ConfigurationMetaNames.DOCUMENT: {"doc_type":"pipeline",
                                                                   "version": "2.0",
                                                                   "primary_pipeline": "dlaas_only",
                                                                   "pipelines": [{"id": "dlaas_only",
                                                                                  "runtime_ref": "hybrid",
                                                                                  "nodes": [{"id": "training",
                                                                                             "type": "model_node",
                                                                                             "op": "dl_train",
                                                                                             "runtime_ref": "DL",
                                                                                             "inputs": [],
                                                                                             "outputs": [],
                                                                                             "parameters": {"name": "tf-mnist",
                                                                                                            "description": "Simple MNIST model implemented in TF",
                                                                                                            "command": "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000",
                                                                                                            "compute": {"name": "k80","nodes": 1},
                                                                                                            "training_lib_href": "/v4/libraries/64758251-bt01-4aa5-a7ay-72639e2ff4d2/content"
                                                                                             },
                                                                                             "target_bucket": "wml-dev-results"
                                                                                  }]
                                                                   }]
                }
            }
            pipeline_details = client.pipelines.store(training_definition_filepath, meta_props=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        Pipelines._validate_type(meta_props, u'meta_props', dict, True)

        if self.ConfigurationMetaNames.DOCUMENT in meta_props:
            pipeline_meta = self.ConfigurationMetaNames._generate_resource_metadata(
                meta_props,
                with_validation=True,
                client=self._client

            )
        else:
            document = self._generate_pipeline_document(meta_props)
            meta_props[self.ConfigurationMetaNames.DOCUMENT] = document
            pipeline_meta = self.ConfigurationMetaNames._generate_resource_metadata(
                meta_props,
                with_validation=True,
                client=self._client
            )
        if self._client.CAMS:
            if self._client.default_space_id is not None:
                pipeline_meta['space'] = {'href': "/v4/spaces/"+self._client.default_space_id}
            elif self._client.default_project_id is not None:
                pipeline_meta['project'] = {'href': "/v2/projects/"+self._client.default_project_id}
            else:
                raise WMLClientError("It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
            if self._client.default_space_id is not None:
                pipeline_meta['space_id'] = self._client.default_space_id
            elif self._client.default_project_id is not None:
                pipeline_meta['project_id'] = self._client.default_project_id
            else:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed.")

        # add kwargs into optimization section at the very end of preparing payload
        try:
            for p in pipeline_meta[self.ConfigurationMetaNames.DOCUMENT]['pipelines']:
                for n in p['nodes']:
                    params = n['parameters']['optimization']
                    params.update(kwargs)
                    n['parameters']['optimization'] = params
        except:
            pass

        if not self._ICP and not self._client.WSD:
            if self._client.CLOUD_PLATFORM_SPACES:
                creation_response = requests.post(
                    self._client.service_instance._href_definitions.get_pipelines_href(),
                    headers=self._client._get_headers(),
                    params=self._client._params(skip_for_create=True),
                    json=pipeline_meta
                 )
            else:
                creation_response = requests.post(
                    self._wml_credentials['url'] + '/v4/pipelines',
                    headers=self._client._get_headers(),
                    json=pipeline_meta
                )

        else:
            if self._client.WSD:
                if "space_uid" in meta_props:
                    raise WMLClientError(
                        u'Invalid input SPACE_UID in meta_props. SPACE_UID not supported for Watson Studio Desktop.')

                pipeline_url = self._wml_credentials['url'] + '/v4/pipelines'
                if self._client.default_project_id is not None:
                    pipeline_meta['project'] = {'href': "/v2/projects/" + self._client.default_project_id}
                else:
                    raise WMLClientError(
                        "It is mandatory to set the project id. Use client.set.default_project(<PROJECT_UID>) to proceed.")

                import copy
                cams_entity = copy.deepcopy(pipeline_meta)
                pipeline_url = self._client.service_instance._href_definitions.get_wsd_model_href()
                #cams_entity.pop('name')
                if cams_entity.get('description') is not None:
                    cams_entity.pop('description')
                cams_payload = self._create_wsd_payload('wml_pipeline', cams_entity, pipeline_meta)

                atype_body = {
                    "name": "wml_pipeline"
                }
                atype_payload = json.dumps(atype_body, separators=(',', ':'))
                atype_url = self._client.service_instance._href_definitions.get_wsd_asset_type_href()
                aheaders = {
                    'Content-Type': "application/json"
                }
                asset_type_response = requests.post(
                    atype_url,
                    params=self._client._params(),
                    data=atype_payload,
                    headers=aheaders
                )

                if asset_type_response.status_code != 200 and \
                        asset_type_response.status_code != 201 and asset_type_response.status_code != 409:
                    raise WMLClientError("Failed to create asset type. Try again.")

                creation_response = requests.post(
                    pipeline_url,
                    params=self._client._params(),
                    headers=self._client._get_headers(),
                    json=cams_payload
                )
                pipeline_details = self._handle_response(201, u'creating new pipeline', creation_response)
                return WMLResource._wsd_get_required_element_from_response(self, pipeline_details)

            elif self._client.ICP_35 or self._client.ICP_40 or self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                creation_response = requests.post(
                    self._client.service_instance._href_definitions.get_pipelines_href(),
                    headers=self._client._get_headers(),
                    params=self._client._params(skip_for_create=True),
                    json=pipeline_meta
                 )
            else:
                creation_response = requests.post(
                    self._wml_credentials['url'] + '/v4/pipelines',
                    headers=self._client._get_headers(),
                    json=pipeline_meta
                )
        pipeline_details = self._handle_response(201, u'creating new pipeline', creation_response)

        return pipeline_details

    def create_revision(self, pipeline_uid):
        """Create a new pipeline revision.

        :param pipeline_uid: Unique pipeline ID
        :type pipeline_uid: str

        :return: pipeline revision details
        :rtype: dict

        **Example**

        .. code-block:: python

            client.pipelines.create_revision(pipeline_uid)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        Pipelines._validate_type(pipeline_uid, u'pipeline_uid', str, False)

        if not any([self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40,
                    self._client.ICP_45, self._client.ICP_46, self._client.ICP_47]):
            raise WMLClientError(
                u'Revision support is not there in this version WML server. It is supported only from 3.1.0 onwards.')
        else:
            url = self._client.service_instance._href_definitions.get_pipelines_href()
            #return self. _create_revision_artifact_for_assets(pipeline_uid, 'pipelines')
            return self._create_revision_artifact(url, pipeline_uid, 'pipelines')

    def update(self, pipeline_uid, changes, rev_id = None):
        """Updates existing pipeline metadata.

        :param pipeline_uid: Unique Id of pipeline which definition should be updated
        :type pipeline_uid: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of updated pipeline
        :rtype: dict

        **Example**

        .. code-block:: python

            metadata = {
                client.pipelines.ConfigurationMetaNames.NAME: "updated_pipeline"
            }
            pipeline_details = client.pipelines.update(pipeline_uid, changes=metadata)

        """
        WMLResource._chk_and_block_create_update_for_python36(self)
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        if self._client.WSD:
            raise WMLClientError(" Update pipeline operation is not supported in WSD. ")

        self._validate_type(pipeline_uid, u'pipeline_uid', str, True)
        self._validate_type(changes, u'changes', dict, True)

        details = self._client.pipelines.get_details(pipeline_uid)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(details['entity'], changes,
                                                                            with_validation=True)

        url = self._client.service_instance._href_definitions.get_pipeline_href(pipeline_uid)

        response = requests.patch(url, json=patch_payload, params=self._client._params(),headers=self._client._get_headers())

        updated_details = self._handle_response(200, u'pipeline patch', response)

        return updated_details

    def delete(self, pipeline_uid):
        """Delete a stored pipeline.

        :param pipeline_uid: Unique Id of pipeline
        :type pipeline_uid: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example**

        .. code-block:: python

            client.pipelines.delete(pipeline_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Pipelines._validate_type(pipeline_uid, u'pipeline_uid', str, True)
        if self._client.WSD:
            pipeline_endpoint = self._client.service_instance._href_definitions.get_model_definition_assets_href() + "/" + pipeline_uid
        else:
            pipeline_endpoint = self._client.service_instance._href_definitions.get_pipeline_href(pipeline_uid)

        response_delete = requests.delete(pipeline_endpoint, params=self._client._params(), headers=self._client._get_headers())

        return self._handle_response(204, u'pipeline deletion', response_delete, False)

    def get_details(self, pipeline_uid=None, limit=None, asynchronous=False, get_all=False):
        """Get metadata of stored pipeline(s). If pipeline UID is not specified returns all pipelines metadata.

        :param pipeline_uid: Pipeline UID
        :type pipeline_uid: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of pipeline(s)
        :rtype: dict (if UID is not None) or {"resources": [dict]} (if UID is None)

        **Example**

        .. code-block:: python

            pipeline_details = client.pipelines.get_details(pipeline_uid)
            pipeline_details = client.pipelines.get_details()
            pipeline_details = client.pipelines.get_details(limit=100)
            pipeline_details = client.pipelines.get_details(limit=100, get_all=True)
            pipeline_details = []
            for entry in client.pipelines.get_details(limit=100, asynchronous=True, get_all=True):
                pipeline_details.extend(entry)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Pipelines._validate_type(pipeline_uid, u'pipeline_uid', str, False)
        Pipelines._validate_type(limit, u'limit', int, False)
        if self._client.WSD:
            url = self._client.service_instance._href_definitions.get_model_definition_assets_href()
            response = self._get_artifact_details(url, pipeline_uid, limit, 'pipeline' if pipeline_uid else 'pipelines', summary="False")
            return self._wsd_get_required_element_from_response(response)
        else:
            url = self._client.service_instance._href_definitions.get_pipelines_href()

        if pipeline_uid is None:
            return self._get_artifact_details(url, pipeline_uid, limit, 'pipelines', summary="False",
                                              _async=asynchronous, _all=get_all)
        else:
            return self._get_artifact_details(url, pipeline_uid, limit, 'pipeline',summary="False")

    def get_revision_details(self, pipeline_uid, rev_uid):
        """Get metadata of pipeline revision.

        :param pipeline_uid:  stored pipeline UID
        :type pipeline_uid: str

        :param rev_uid:  stored pipeline revision ID
        :type rev_uid: str

        :return: stored pipeline revision metadata
        :rtype: dict

        **Example:**

        .. code-block:: python

            pipeline_details = client.pipelines.get_revision_details(pipeline_uid, rev_uid)

        .. note::
            `rev_uid` parameter is not applicable in Cloud platform.
         """
        Pipelines._validate_type(pipeline_uid, u'pipeline_uid', str, True)
        Pipelines._validate_type(rev_uid, u'rev_uid', int, True)
        if not any([self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40,
                    self._client.ICP_45, self._client.ICP_46, self._client.ICP_47]):
            raise WMLClientError(
                'Not supported. Revisions APIs are supported only for IBM Cloud Pak® for Data 3.0 and above.')
        else:
            url = self._client.service_instance._href_definitions.get_pipeline_href(pipeline_uid)
        return self._get_with_or_without_limit(url, limit=None, op_name="pipeline",
                                               summary=None, pre_defined=None, revision=rev_uid)

    @staticmethod
    def get_href(pipeline_details):
        """Get href from pipeline details.

        :param pipeline_details: metadata of the stored pipeline
        :type pipeline_details: dict

        :return: pipeline href
        :rtype: str

        **Example**

        .. code-block:: python

            pipeline_details = client.pipelines.get_details(pipeline_uid)
            pipeline_href = client.pipelines.get_href(pipeline_details)

        """
        Pipelines._validate_type(pipeline_details, u'pipeline_details', object, True)

        if 'asset_type' in pipeline_details['metadata']:
            return WMLResource._get_required_element_from_dict(pipeline_details, u'pipeline_details',
                                                               [u'metadata', u'href'])
        else:
            if 'href' in pipeline_details['metadata']:
                Pipelines._validate_type_of_details(pipeline_details, PIPELINE_DETAILS_TYPE)
                return WMLResource._get_required_element_from_dict(pipeline_details, u'pipeline_details', [u'metadata', u'href'])
            else:
                pipeline_id = WMLResource._get_required_element_from_dict(pipeline_details, u'pipeline_details', [u'metadata', u'id'])
                return "/ml/v4/pipelines/" + pipeline_id


    @staticmethod
    def get_uid(pipeline_details):
        """Get pipeline_uid from pipeline details.

        :param pipeline_details: metadata of the stored pipeline
        :type pipeline_details: dict

        :return: Unique Id of pipeline
        :rtype: str

        **Example**

        .. code-block:: python

            pipeline_uid = client.pipelines.get_uid(pipeline_details)

        """
        Pipelines._validate_type(pipeline_details, u'pipeline_details', object, True)
        if 'asset_id' in pipeline_details['metadata']:
            return WMLResource._get_required_element_from_dict(pipeline_details, u'pipeline_details',
                                                               [u'metadata', u'asset_id'])
        else:
            if 'id' not in pipeline_details[u'metadata']:
                Pipelines._validate_type_of_details(pipeline_details, PIPELINE_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(pipeline_details, u'pipeline_details', [u'metadata', u'id'])

    @staticmethod
    def get_id(pipeline_details):
        """Get pipeline id from pipeline details.

        :param pipeline_details: metadata of the stored pipeline
        :type pipeline_details: dict

        :return: Unique Id of pipeline
        :rtype: str

        **Example**

        .. code-block:: python

            pipeline_uid = client.pipelines.get_id(pipeline_details)

        """
        Pipelines._validate_type(pipeline_details, u'pipeline_details', object, True)
        if 'asset_id' in pipeline_details['metadata']:
            return WMLResource._get_required_element_from_dict(pipeline_details, u'pipeline_details',
                                                               [u'metadata', u'asset_id'])
        else:
            if 'id' not in pipeline_details[u'metadata']:
                Pipelines._validate_type_of_details(pipeline_details, PIPELINE_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(pipeline_details, u'pipeline_details',
                                                               [u'metadata', u'id'])

    def list(self, limit=None, return_as_df=True):
        """Print stored pipelines in a table format. If limit is set to None there will be only first 50 records shown.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed pipelines or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.pipelines.list()

        """
        table = None
        if self._client.WSD:
            Pipelines._wsd_list_assets(self, "wml_pipeline", limit)
        else:
            ##For CP4D, check if either spce or project ID is set
            self._client._check_if_either_is_set()

            pipeline_resources = self.get_details(limit=limit)[u'resources']
            if self._client.CLOUD_PLATFORM_SPACES or self._client.ICP_35 or self._client.ICP_40 or \
                    self._client.ICP_45 or self._client.ICP_46 or self._client.ICP_47:
                pipeline_values = [(m[u'metadata'][u'id'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at']) for m
                                   in pipeline_resources]

                table = self._list(pipeline_values, [u'ID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
            else:
                pipeline_values = [(m[u'metadata'][u'id'], m[u'entity'][u'name'], m[u'metadata'][u'created_at']) for m in pipeline_resources]

                table = self._list(pipeline_values, [u'GUID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def list_revisions(self, pipeline_uid, limit=None, return_as_df=True):
        """Print all revision for the given pipeline uid in a table format.

        :param pipeline_uid: Unique id of stored pipeline
        :type pipeline_uid: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param return_as_df: determinate if table should be returned as pandas.DataFrame object, default: True
        :type return_as_df: bool, optional

        :return: pandas.DataFrame with listed revisions or None if return_as_df is False
        :rtype: pandas.DataFrame or None

        **Example**

        .. code-block:: python

            client.pipelines.list_revisions(pipeline_uid)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Pipelines._validate_type(pipeline_uid, u'pipeline_uid', str, True)
        table = None
        if not any([self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40,
                    self._client.ICP_45, self._client.ICP_46, self._client.ICP_47]):
            raise WMLClientError(
                u'Revision support is not there in this WML server. It is supported only from 3.1.0 onwards.')
        else:
            url = self._client.service_instance._href_definitions.get_pipeline_href(pipeline_uid)
            pipeline_resources = self._get_artifact_details(url, "revisions", limit, 'pipeline revisions')[u'resources']
            pipeline_values = [
                (m[u'metadata'][u'rev'], m[u'metadata'][u'name'], m[u'metadata'][u'created_at']) for m in
                pipeline_resources]

            table = self._list(pipeline_values, [u'GUID', u'NAME', u'CREATED'], limit, _DEFAULT_LIST_LENGTH)
        if return_as_df:
            return table

    def clone(self, pipeline_uid, space_id=None, action="copy", rev_id=None):
        """Create a new pipeline identical with the given pipeline either in the same space or in a new space.
        All dependent assets will be cloned too.

        :param pipeline_uid: uid of the pipeline to be cloned
        :type pipeline_uid: str
        :param space_id: uid of the space to which the pipeline needs to be cloned
        :type space_id: str, optional
        :param action: action specifying "copy" or "move"
        :type action: str, optional
        :param rev_id: revision ID of the pipeline
        :type rev_id: str, optional

        :return: metadata of the pipeline cloned
        :rtype: dict

        **Example**

        .. code-block:: python

            client.pipelines.clone(pipeline_uid=artifact_id,space_id=space_uid,action="copy")

        .. note::
            * If revision id is not specified, all revisions of the artifact are cloned.

            * Space id is mandatory for "move" action.

        """
        if any([self._client.WSD, self._client.CLOUD_PLATFORM_SPACES, self._client.ICP_35, self._client.ICP_40,
                self._client.ICP_45, self._client.ICP_46, self._client.ICP_47]):
            raise WMLClientError(" Clone pipeline operation is not supported. ")
        Pipelines._validate_type(pipeline_uid, 'exp_id', str, True)
        clone_meta = {}
        if space_id is not None:
            clone_meta["space"] = {"href": API_VERSION + SPACES + "/" + space_id}
        if action is not None:
            clone_meta["action"] = action
        if rev_id is not None:
            clone_meta["rev"] = rev_id

        url = self._client.service_instance._href_definitions.get_pipeline_href(pipeline_uid)

        response_post = requests.post(url, json=clone_meta,
                                          headers=self._client._get_headers())

        details = self._handle_response(expected_status_code=200, operationName=u'cloning pipeline',
                                            response=response_post)

        return details