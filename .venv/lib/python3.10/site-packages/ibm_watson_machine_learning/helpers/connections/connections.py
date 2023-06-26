__all__ = [
    "DataConnection",
    "S3Connection",
    "ConnectionAsset",
    "S3Location",
    "FSLocation",
    "AssetLocation",
    "CP4DAssetLocation",
    "WMLSAssetLocation",
    "WSDAssetLocation",
    "CloudAssetLocation",
    "DeploymentOutputAssetLocation",
    "NFSConnection",
    "NFSLocation",
    'ConnectionAssetLocation',
    "DatabaseLocation",
    "ContainerLocation"
]

#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import io
import os
import uuid
import copy
import sys
from copy import deepcopy
from typing import Union, Tuple, List, TYPE_CHECKING, Optional
from warnings import warn

from ibm_boto3 import resource
from ibm_botocore.client import ClientError
from pandas import DataFrame
import pandas as pd
import ibm_watson_machine_learning._wrappers.requests as requests

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, DataConnectionTypes
from ibm_watson_machine_learning.utils.autoai.errors import (
    MissingAutoPipelinesParameters, UseWMLClient, MissingCOSStudioConnection, MissingProjectLib,
    HoldoutSplitNotSupported, InvalidCOSCredentials, MissingLocalAsset, InvalidIdType, NotWSDEnvironment,
    NotExistingCOSResource, InvalidDataAsset, CannotReadSavedRemoteDataBeforeFit, NoAutomatedHoldoutSplit
)

import numpy as np
from ibm_watson_machine_learning.utils.autoai.utils import all_logging_disabled, try_import_autoai_libs, \
    try_import_autoai_ts_libs
from ibm_watson_machine_learning.utils.autoai.watson_studio import get_project
from ibm_watson_machine_learning.data_loaders.datasets.experiment import DEFAULT_SAMPLING_TYPE, DEFAULT_SAMPLE_SIZE_LIMIT
from ibm_watson_machine_learning.wml_client_error import MissingValue, ApiRequestFailure, WMLClientError
from ibm_watson_machine_learning.utils.autoai.errors import ContainerTypeNotSupported
from .base_connection import BaseConnection
from .base_data_connection import BaseDataConnection
from .base_location import BaseLocation

if TYPE_CHECKING:
    from ibm_watson_machine_learning.workspace import WorkSpace


class DataConnection(BaseDataConnection):
    """Data Storage Connection class needed for WML training metadata (input data).

    :param connection: connection parameters of specific type
    :type connection: NFSConnection or ConnectionAsset, optional

    :param location: required location parameters of specific type
    :type location: Union[S3Location, FSLocation, AssetLocation]

    :param data_join_node_name: name(s) for node(s):

        - `None` - data file name will be used as node name
        - str - it will became node name
        - list[str] - multiple names passed, several nodes will have the same data connection
          (used for excel files with multiple sheets)
    :type data_join_node_name:  None or str or list[str], optional

    :param data_asset_id: data asset ID if DataConnection should be pointing out to data asset
    :type data_asset_id: str, optional
    """

    def __init__(self,
                 location: Union['S3Location',
                                 'FSLocation',
                                 'AssetLocation',
                                 'CP4DAssetLocation',
                                 'WMLSAssetLocation',
                                 'WSDAssetLocation',
                                 'CloudAssetLocation',
                                 'NFSLocation',
                                 'DeploymentOutputAssetLocation',
                                 'ConnectionAssetLocation',
                                 'DatabaseLocation',
                                 'ContainerLocation'] = None,
                 connection: Optional[Union['S3Connection', 'NFSConnection', 'ConnectionAsset']] = None,
                 data_join_node_name: Union[str, List[str]] = None,
                 data_asset_id: str = None,
                 connection_asset_id: str = None,
                 **kwargs):
        if data_asset_id is None and location is None:
            raise MissingValue('location or data_asset_id', reason="Provide 'location' or 'data_asset_id'.")

        elif data_asset_id is not None and location is not None:
            raise ValueError("'data_asset_id' and 'location' cannot be specified together.")

        elif data_asset_id is not None:
            location = AssetLocation(asset_id=data_asset_id)

            if kwargs.get('model_location') is not None:
                location._model_location = kwargs['model_location']

            if kwargs.get('training_status') is not None:
                location._training_status = kwargs['training_status']

        elif connection_asset_id is not None and isinstance(location, (S3Location, DatabaseLocation, NFSLocation)):
            connection = ConnectionAsset(connection_id=connection_asset_id)

        super().__init__()

        self.connection = connection
        self.location = location

        # TODO: remove S3 implementation
        if isinstance(connection, S3Connection):
            self.type = DataConnectionTypes.S3

        elif isinstance(connection, ConnectionAsset):
            self.type = DataConnectionTypes.CA
            # note: We expect a `file_name` keyword for CA pointing to COS or NFS.
            if isinstance(self.location, (S3Location, NFSLocation)):
                self.location.file_name = self.location.path
                del self.location.path
                if isinstance(self.location, NFSLocation):
                    del self.location.id
            # --- end note

        elif isinstance(location, FSLocation):
            self.type = DataConnectionTypes.FS

        elif isinstance(location, ContainerLocation):
            self.type = DataConnectionTypes.CN

        elif isinstance(location, (AssetLocation, CP4DAssetLocation, WMLSAssetLocation, CloudAssetLocation,
                                   WSDAssetLocation, DeploymentOutputAssetLocation)):
            self.type = DataConnectionTypes.DS

        self.auto_pipeline_params = {}  # note: needed parameters for recreation of autoai holdout split
        self._wml_client = None
        self.__wml_client = None  # only for getter/setter for AssetLocation href
        self._run_id = None
        self._obm = False
        self._obm_cos_path = None
        self._test_data = False
        self._user_holdout_exists = False

        # note: make data connection id as a location path for OBM + KB
        if data_join_node_name is None:
            # TODO: remove S3 implementation
            if self.type == DataConnectionTypes.S3 or (
                    self.type == DataConnectionTypes.CA and hasattr(location, 'file_name')):
                self.id = location.get_location()

            else:
                self.id = None

        else:
            self.id = data_join_node_name
        # --- end note

    # note: client as property and setter for dynamic href creation for AssetLocation
    @property
    def _wml_client(self):
        return self.__wml_client

    @_wml_client.setter
    def _wml_client(self, var):
        self.__wml_client = var
        if isinstance(self.location, (AssetLocation, WSDAssetLocation)):
            self.location.wml_client = self.__wml_client

        if getattr(var, 'project_type', None) == 'local_git_storage':
            self.location.userfs = True

    def set_client(self, wml_client):
        """Set initialized wml client in connection to enable write/read operations with connection to service.

        :param wml_client: WML client to connect to service
        :type wml_client: APIClient

        **Example**

        .. code-block:: python

            DataConnection.set_client(wml_client)
        """
        self._wml_client = wml_client

    # --- end note

    @classmethod
    def from_studio(cls, path: str) -> List['DataConnection']:
        """Create DataConnections from the credentials stored (connected) in Watson Studio. Only for COS.

        :param path: path in COS bucket to the training dataset
        :type path: str

        :return: list with DataConnection objects
        :rtype: list[DataConnection]

        **Example**

        .. code-block:: python

            data_connections = DataConnection.from_studio(path='iris_dataset.csv')
        """
        try:
            from project_lib import Project

        except ModuleNotFoundError:
            raise MissingProjectLib("Missing project_lib package.")

        else:
            data_connections = []
            for name, value in globals().items():
                if isinstance(value, Project):
                    connections = value.get_connections()

                    if connections:
                        for connection in connections:
                            asset_id = connection['asset_id']
                            connection_details = value.get_connection(asset_id)

                            if ('url' in connection_details and 'access_key' in connection_details and
                                    'secret_key' in connection_details and 'bucket' in connection_details):
                                data_connections.append(
                                    cls(connection=ConnectionAsset(id=connection_details['id']),
                                        location=ConnectionAssetLocation(bucket=connection_details['bucket'],
                                                                         file_name=path))
                                )

            if data_connections:
                return data_connections

            else:
                raise MissingCOSStudioConnection(
                    "There is no any COS Studio connection. "
                    "Please create a COS connection from the UI and insert "
                    "the cell with project API connection (Insert project token)")

    def _subdivide_connection(self):
        if type(self.id) is str or not self.id:
            return [self]
        else:
            def cpy(new_id):
                child = copy.copy(self)
                child.id = new_id
                return child

            return [cpy(id) for id in self.id]

    def _to_dict(self) -> dict:
        """Convert DataConnection object to dictionary representation.

        :return: DataConnection dictionary representation
        :rtype: dict
        """

        if self.id and type(self.id) is list:
            raise InvalidIdType(list)

        _dict = {"type": self.type}

        # note: for OBM (id of DataConnection if an OBM node name)
        if self.id is not None:
            _dict['id'] = self.id
        # --- end note

        if self.connection is not None:
            _dict['connection'] = deepcopy(self.connection.to_dict())
        else:
            _dict['connection'] = {}

        try:
            _dict['location'] = deepcopy(self.location.to_dict())

        except AttributeError:
            _dict['location'] = {}

        # note: convert userfs to string - training service requires it as string
        if hasattr(self.location, 'userfs'):
            _dict['location']['userfs'] = str(getattr(self.location, 'userfs', False)).lower()
        # end note

        return _dict

    def __repr__(self):
        return str(self._to_dict())

    def __str__(self):
        return str(self._to_dict())

    @classmethod
    def _from_dict(cls, _dict: dict) -> 'DataConnection':
        """Create a DataConnection object from dictionary.

        :param _dict: a dictionary data structure with information about data connection reference
        :type _dict: dict

        :return: DataConnection object
        :rtype: DataConnection
        """
        # TODO: remove S3 implementation
        if _dict['type'] == DataConnectionTypes.S3:
            warn(message="S3 DataConnection is deprecated! Please use data_asset_id instead.")

            data_connection: 'DataConnection' = cls(
                connection=S3Connection(
                    access_key_id=_dict['connection']['access_key_id'],
                    secret_access_key=_dict['connection']['secret_access_key'],
                    endpoint_url=_dict['connection']['endpoint_url']
                ),
                location=S3Location(
                    bucket=_dict['location']['bucket'],
                    path=_dict['location']['path']
                )
            )
        elif _dict['type'] == DataConnectionTypes.FS:
            data_connection: 'DataConnection' = cls(
                location=FSLocation._set_path(path=_dict['location']['path'])
            )
        elif _dict['type'] == DataConnectionTypes.CA:
            if _dict['location'].get('file_name') is not None and _dict['location'].get('bucket'):
                data_connection: 'DataConnection' = cls(
                    connection_asset_id=_dict['connection']['id'],
                    location=S3Location(
                        bucket=_dict['location']['bucket'],
                        path=_dict['location']['file_name']
                    )
                )

            elif _dict['location'].get('path') is not None and _dict['location'].get('bucket'):
                data_connection: 'DataConnection' = cls(
                    connection_asset_id=_dict['connection']['id'],
                    location=S3Location(
                        bucket=_dict['location']['bucket'],
                        path=_dict['location']['path']
                    )
                )

            elif _dict['location'].get('schema_name') and _dict['location'].get('table_name'):
                data_connection: 'DataConnection' = cls(
                    connection_asset_id=_dict['connection']['id'],
                    location=DatabaseLocation(schema_name=_dict['location']['schema_name'],
                                              table_name=_dict['location']['table_name'])
                )

            else:
                if 'asset_id' in _dict['connection']:
                    data_connection: 'DataConnection' = cls(
                        connection=NFSConnection(asset_id=_dict['connection']['asset_id']),
                        location=NFSLocation(path=_dict['location']['path'])
                    )
                else:
                    if _dict['location'].get('file_name') is not None:
                        data_connection: 'DataConnection' = cls(
                            connection_asset_id=_dict['connection']['id'],
                            location=NFSLocation(path=_dict['location']['file_name'])
                        )
                    else:
                        data_connection: 'DataConnection' = cls(
                            connection_asset_id=_dict['connection']['id'],
                            location=NFSLocation(path=_dict['location']['path'])
                        )
        elif _dict['type'] == DataConnectionTypes.CN:
            data_connection: 'DataConnection' = cls(
                location=ContainerLocation(path=_dict['location']['path'])
            )

        else:
            data_connection: 'DataConnection' = cls(
                location=AssetLocation._set_path(href=_dict['location']['href'])
            )

        if _dict.get('id'):
            data_connection.id = _dict['id']

        if _dict['location'].get('userfs'):
            if str(_dict['location'].get('userfs', 'false')).lower() in ['true', '1']:
                data_connection.location.userfs = True
            else:
                data_connection.location.userfs = False

        return data_connection

    def _recreate_holdout(
            self,
            data: 'DataFrame',
            with_holdout_split: bool = True
    ) -> Union[Tuple['DataFrame', 'DataFrame'], Tuple['DataFrame', 'DataFrame', 'DataFrame', 'DataFrame']]:
        """This method tries to recreate holdout data."""

        if self.auto_pipeline_params.get('prediction_columns') is not None:
            # timeseries
            try_import_autoai_ts_libs()
            from autoai_ts_libs.utils.holdout_utils import make_holdout_split

            # Note: When lookback window is auto detected there is need to get the detected value from training details
            if self.auto_pipeline_params.get('lookback_window') == -1 or self.auto_pipeline_params.get('lookback_window') is None:
                ts_metrics = self._wml_client.training.get_details(self.auto_pipeline_params.get('run_id'), _internal=True)['entity']['status']['metrics']
                final_ts_state_name = "after_final_pipelines_generation"


                for metric in ts_metrics:
                    if metric['context']['intermediate_model']['process'] == final_ts_state_name:
                        self.auto_pipeline_params['lookback_window'] = metric['context']['timeseries']['lookback_window']
                        break

            # Note: imputation is not supported
            X_train, X_holdout, y_train, y_holdout, _, _, _, _ = make_holdout_split(
                dataset=data,
                target_columns=self.auto_pipeline_params.get('prediction_columns'),
                learning_type="forecasting",
                test_size=self.auto_pipeline_params.get('holdout_size'),
                lookback_window=self.auto_pipeline_params.get('lookback_window'),
                feature_columns=self.auto_pipeline_params.get('feature_columns'),
                timestamp_column=self.auto_pipeline_params.get('timestamp_column_name'),
                # n_jobs=None,
                # tshirt_size=None,
                return_only_holdout=False
            )

            X_columns = self.auto_pipeline_params.get('feature_columns') if self.auto_pipeline_params.get('feature_columns') else self.auto_pipeline_params['prediction_columns']

            X_train = DataFrame(X_train, columns=X_columns)
            X_holdout = DataFrame(X_holdout, columns=X_columns)
            y_train = DataFrame(y_train, columns=self.auto_pipeline_params['prediction_columns'])
            y_holdout = DataFrame(y_holdout, columns=self.auto_pipeline_params['prediction_columns'])

            return X_train, X_holdout, y_train, y_holdout
        elif self.auto_pipeline_params.get('feature_columns') is not None:
            # timeseries anomaly detection
            try_import_autoai_ts_libs()
            from autoai_ts_libs.utils.holdout_utils import make_holdout_split
            from autoai_ts_libs.utils.constants import LEARNING_TYPE_TIMESERIES_ANOMALY_PREDICTION

            # Note: imputation is not supported
            X_train, X_holdout, y_train, y_holdout, _, _, _, _ = make_holdout_split(
                dataset=data,
                learning_type=LEARNING_TYPE_TIMESERIES_ANOMALY_PREDICTION,
                test_size=self.auto_pipeline_params.get('holdout_size'),
                # lookback_window=self.auto_pipeline_params.get('lookback_window'),
                feature_columns=self.auto_pipeline_params.get('feature_columns'),
                timestamp_column=self.auto_pipeline_params.get('timestamp_column_name'),
                # n_jobs=None,
                # tshirt_size=None,
                return_only_holdout=False
            )

            X_columns = self.auto_pipeline_params['feature_columns']
            y_column = ['anomaly_label']

            X_train = DataFrame(X_train, columns=X_columns)
            X_holdout = DataFrame(X_holdout, columns=X_columns)
            y_train = DataFrame(y_train, columns=y_column)
            y_holdout = DataFrame(y_holdout, columns=y_column)

            return X_train, X_holdout, y_train, y_holdout

        else:
            if sys.version_info >= (3, 10):
                try_import_autoai_libs(minimum_version='1.14.0')
            else:
                try_import_autoai_libs(minimum_version='1.12.14')

            from autoai_libs.utils.holdout_utils import make_holdout_split, numpy_split_on_target_values
            from autoai_libs.utils.sampling_utils import numpy_sample_rows

            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.drop_duplicates(inplace=True)
            data.dropna(subset=[self.auto_pipeline_params['prediction_column']], inplace=True)
            dfy = data[self.auto_pipeline_params['prediction_column']]
            data.drop(columns=[self.auto_pipeline_params['prediction_column']], inplace=True)

            y_column = [self.auto_pipeline_params['prediction_column']]
            X_columns = data.columns

            if self._test_data or not with_holdout_split:
                return data, dfy

            else:
                ############################
                #   REMOVE MISSING ROWS    #
                from autoai_libs.utils.holdout_utils import numpy_remove_missing_target_rows
                # Remove (and save) the rows of X and y for which the target variable has missing values
                data, dfy, _, _, _, _ = numpy_remove_missing_target_rows(
                    y=dfy, X=data
                )
                #   End of REMOVE MISSING ROWS    #
                ###################################

                #################
                #   SAMPLING    #
                # Get a sample of the rows if requested and applicable
                # (check for sampling is performed inside this function)
                try:
                    data, dfy, _ = numpy_sample_rows(
                        X=data,
                        y=dfy,
                        train_sample_rows_test_size=self.auto_pipeline_params['train_sample_rows_test_size'],
                        learning_type=self.auto_pipeline_params['prediction_type'],
                        return_sampled_indices=True
                    )

                # Note: we have a silent error here (the old core behaviour)
                # sampling is not performed as 'train_sample_rows_test_size' is bigger than data rows count
                # TODO: can we throw an error instead?
                except ValueError as e:
                    if 'between' in str(e):
                        pass

                    else:
                        raise e
                #   End of SAMPLING    #
                ########################

                # Perform holdout split
                X_train, X_holdout, y_train, y_holdout, _, _ = make_holdout_split(
                    x=data,
                    y=dfy,
                    learning_type=self.auto_pipeline_params['prediction_type'],
                    fairness_info=self.auto_pipeline_params.get('fairness_info', None),
                    test_size=self.auto_pipeline_params.get('holdout_size') if self.auto_pipeline_params.get('holdout_size') is not None else 0.1,
                    return_only_holdout=False
                )

                X_train = DataFrame(X_train, columns=X_columns)
                X_holdout = DataFrame(X_holdout, columns=X_columns)
                y_train = DataFrame(y_train, columns=y_column)
                y_holdout = DataFrame(y_holdout, columns=y_column)

                return X_train, X_holdout, y_train, y_holdout

    def read(self,
             with_holdout_split: bool = False,
             csv_separator: str = ',',
             excel_sheet: Union[str, int] = None,
             encoding: Optional[str] = 'utf-8',
             raw: Optional[bool] = False,
             binary: Optional[bool] = False,
             read_to_file: Optional[str] = None,
             number_of_batch_rows: Optional[int] = None,
             sampling_type: Optional[str] = None,
             sample_size_limit: Optional[int] = None,
             sample_rows_limit: Optional[int] = None,
             sample_percentage_limit: Optional[float] = None,
             **kwargs) -> Union['DataFrame', Tuple['DataFrame', 'DataFrame'], bytes]:
        """Download dataset stored in remote data storage. Returns batch up to 1GB.

        :param with_holdout_split: if `True`, data will be split to train and holdout dataset as it was by AutoAI
        :type with_holdout_split: bool, optional

        :param csv_separator: separator / delimiter for CSV file
        :type csv_separator: str, optional

        :param excel_sheet: excel file sheet name to use, only use when xlsx file is an input,
            support for number of the sheet is deprecated
        :type excel_sheet: str, optional

        :param encoding: encoding type of the CSV
        :type encoding: str, optional

        :param raw: if `False` there wil be applied simple data preprocessing (the same as in the backend),
            if `True`, data will be not preprocessed
        :type raw: bool, optional

        :param binary: indicates to retrieve data in binary mode, the result will be a python binary type variable
        :type binary: bool, optional

        :param read_to_file: stream read data to file under path specified as value of this parameter,
            use this parameter to prevent keeping data in-memory
        :type read_to_file: str, optional

        :param number_of_batch_rows: number of rows to read in each batch when reading from flight connection
        :type number_of_batch_rows: int, optional

        :param sampling_type: a sampling strategy how to read the data
        :type sampling_type: str, optional

        :param sample_size_limit: upper limit for overall data that should be downloaded in bytes, default: 1 GB
        :type sample_size_limit: int, optional

        :param sample_rows_limit: upper limit for overall data that should be downloaded in number of rows
        :type sample_rows_limit: int, optional

        :param sample_percentage_limit: upper limit for overall data that should be downloaded
            in percent of all dataset, this parameter is ignored, when `sampling_type` parameter is set
            to `first_n_records`, must be a float number between 0 and 1
        :type sample_percentage_limit: float, optional

        .. note::

            If more than one of: `sample_size_limit`, `sample_rows_limit`, `sample_percentage_limit` are set,
            then downloaded data is limited to the lowest threshold.

        :return: one of:

            - pandas.DataFrame contains dataset from remote data storage : Xy_train
            - Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame] : X_train, X_holdout, y_train, y_holdout
            - Tuple[pandas.DataFrame, pandas.DataFrame] : X_test, y_test containing training data and holdout data from
              remote storage
            - bytes object, auto holdout split from backend (only train data provided)

        **Examples**

        .. code-block:: python

            train_data_connections = optimizer.get_data_connections()

            data = train_data_connections[0].read() # all train data

            # or

            X_train, X_holdout, y_train, y_holdout = train_data_connections[0].read(with_holdout_split=True) # train and holdout data

        User provided train and test data:

        .. code-block:: python

            optimizer.fit(training_data_reference=[DataConnection],
                          training_results_reference=DataConnection,
                          test_data_reference=DataConnection)

            test_data_connection = optimizer.get_test_data_connections()
            X_test, y_test = test_data_connection.read() # only holdout data

            # and

            train_data_connections = optimizer.get_data_connections()
            data = train_connections[0].read() # only train data
        """

        # enables flight automatically for CP4D 4.0.x, 4.5.x, for 3.0 and 3.5 we do not have a flight service there
        try:
            use_flight = kwargs.get(
                'use_flight',
                bool((self._wml_client is not None or 'USER_ACCESS_TOKEN' in os.environ or 'RUNTIME_ENV_ACCESS_TOKEN_FILE' in os.environ) and (
                        self._wml_client.ICP_40 or self._wml_client.ICP_45 or self._wml_client.ICP_46 or self._wml_client.ICP_47)))
        except:
            use_flight = False

        return_data_as_iterator = kwargs.get('return_data_as_iterator', False)
        sampling_type = sampling_type if sampling_type is not None else DEFAULT_SAMPLING_TYPE
        enable_sampling = kwargs.get('enable_sampling', True)
        total_size_limit = sample_size_limit if sample_size_limit is not None else kwargs.get('total_size_limit', DEFAULT_SAMPLE_SIZE_LIMIT)
        total_nrows_limit = sample_rows_limit
        total_percentage_limit = sample_percentage_limit if sample_percentage_limit is not None else 1.0

        # Deprecation of excel_sheet as number:
        if isinstance(excel_sheet, int):
            warn(
                message="Support for excel sheet as number of the sheet (int) is deprecated! Please set excel sheet with name of the sheet.")

        flight_parameters = kwargs.get('flight_parameters', {})
        impersonate_header = kwargs.get('impersonate_header', None)

        if with_holdout_split and self._user_holdout_exists:  # when this connection is training one
            raise NoAutomatedHoldoutSplit(reason="Experiment was run based on user defined holdout dataset.")

        # note: experiment metadata is used only in autogen notebooks
        experiment_metadata = kwargs.get('experiment_metadata')
        # note: process subsampling stats flag
        _return_subsampling_stats = kwargs.get("_return_subsampling_stats", False)

        if experiment_metadata is not None:
            self.auto_pipeline_params['train_sample_rows_test_size'] = experiment_metadata.get(
                'train_sample_rows_test_size')
            self.auto_pipeline_params['prediction_column'] = experiment_metadata.get('prediction_column')
            self.auto_pipeline_params['prediction_columns'] = experiment_metadata.get('prediction_columns')
            self.auto_pipeline_params['holdout_size'] = experiment_metadata.get('holdout_size')
            self.auto_pipeline_params['prediction_type'] = experiment_metadata['prediction_type']
            self.auto_pipeline_params['fairness_info'] = experiment_metadata.get('fairness_info')
            self.auto_pipeline_params['lookback_window'] = experiment_metadata.get('lookback_window')
            self.auto_pipeline_params['timestamp_column_name'] = experiment_metadata.get('timestamp_column_name')
            self.auto_pipeline_params['feature_columns'] = experiment_metadata.get('feature_columns')

            # note: check for cloud
            if 'training_result_reference' in experiment_metadata:
                if isinstance(experiment_metadata['training_result_reference'].location, (S3Location, AssetLocation)):
                    run_id = experiment_metadata['training_result_reference'].location._training_status.split('/')[-2]
                # WMLS
                else:
                    run_id = experiment_metadata['training_result_reference'].location.path.split('/')[-3]
                self.auto_pipeline_params['run_id'] = run_id

            if self._test_data:
                csv_separator = experiment_metadata.get('test_data_csv_separator', csv_separator)
                excel_sheet = experiment_metadata.get('test_data_excel_sheet', excel_sheet)
                encoding = experiment_metadata.get('test_data_encoding', encoding)

            else:
                csv_separator = experiment_metadata.get('csv_separator', csv_separator)
                excel_sheet = experiment_metadata.get('excel_sheet', excel_sheet)
                encoding = experiment_metadata.get('encoding', encoding)

        if self.type == DataConnectionTypes.DS or self.type == DataConnectionTypes.CA:
            if self._wml_client is None:
                try:
                    from project_lib import Project

                except ModuleNotFoundError:
                    raise ConnectionError(
                        "This functionality can be run only on Watson Studio or with wml_client passed to connection. "
                        "Please initialize WML client using `DataConnection.set_client(wml_client)` function "
                        "to be able to use this functionality.")

        if (with_holdout_split or self._test_data) and not self.auto_pipeline_params.get('prediction_type', False):
            raise MissingAutoPipelinesParameters(
                self.auto_pipeline_params,
                reason=f"To be able to recreate an original holdout split, you need to schedule a training job or "
                       f"if you are using historical runs, just call historical_optimizer.get_data_connections()")

        # note: allow to read data at any time
        elif (('csv_separator' not in self.auto_pipeline_params and 'encoding' not in self.auto_pipeline_params)
              or csv_separator != ',' or encoding != 'utf-8'):
            self.auto_pipeline_params['csv_separator'] = csv_separator
            self.auto_pipeline_params['encoding'] = encoding
        # --- end note
        # note: excel_sheet in params only if it is not None (not specified):
        if excel_sheet:
            self.auto_pipeline_params['excel_sheet'] = excel_sheet
        # --- end note

        # note: set default quote character for flight (later applicable only for csv files stored in S3)
        self.auto_pipeline_params['quote_character'] = 'double_quote'
        # --- end note

        data = DataFrame()

        headers = None
        if self._wml_client is None:
            token = self._get_token_from_environment()
            if token is not None:
                headers = {'Authorization': f'Bearer {token}'}
        elif impersonate_header is not None:
            headers = self._wml_client._get_headers()

            headers['impersonate'] = impersonate_header
        if self.type == DataConnectionTypes.S3:
            raise ConnectionError(
                f"S3 DataConnection is deprecated! Please use data_asset_id instead.")

        elif self.type == DataConnectionTypes.DS:
            if use_flight and not self._obm:
                from ibm_watson_machine_learning.utils.utils import is_lib_installed

                is_lib_installed(lib_name='pyarrow', minimum_version='3.0.0', install=True)

                from pyarrow.flight import FlightError

                _iam_id = None
                if headers and headers.get('impersonate'):
                    _iam_id = headers.get('impersonate', {}).get('iam_id')

                self._wml_client._iam_id = _iam_id

                try:
                    if self._check_if_connection_asset_is_s3():
                        # note: update flight parameters only if `connection_properties` was not set earlier
                        #       (e.x. by wml/autoi)
                        if not flight_parameters.get('connection_properties'):
                            flight_parameters = self._update_flight_parameters_with_connection_details(flight_parameters)

                    data = self._download_data_from_flight_service(data_location=self,
                                                                   binary=binary,
                                                                   read_to_file=read_to_file,
                                                                   flight_parameters=flight_parameters,
                                                                   headers=headers,
                                                                   enable_sampling=enable_sampling,
                                                                   sampling_type=sampling_type,
                                                                   number_of_batch_rows=number_of_batch_rows,
                                                                   return_data_as_iterator=return_data_as_iterator,
                                                                   _return_subsampling_stats=_return_subsampling_stats,
                                                                   total_size_limit=total_size_limit,
                                                                   total_nrows_limit=total_nrows_limit,
                                                                   total_percentage_limit=total_percentage_limit
                                                                   )
                except (ConnectionError, FlightError, ApiRequestFailure) as download_data_error:
                    # note: try to download normal data asset either directly from cams or from mounted NFS
                    #       to keep backward compatibility
                    if self._wml_client and ((self._is_data_asset_normal() and self._is_size_acceptable()) or self._is_data_asset_nfs()) :
                        import warnings
                        warnings.warn(str(download_data_error), Warning)
                        data = self._download_training_data_from_data_asset_storage()
                    else:
                        raise download_data_error

            # backward compatibility
            else:
                try:
                    with all_logging_disabled():
                        if self._check_if_connection_asset_is_s3():
                            cos_client = self._init_cos_client()

                            if self._obm:
                                data = self._download_obm_data_from_cos(cos_client=cos_client)

                            else:
                                data = self._download_data_from_cos(cos_client=cos_client)
                        else:
                            data = self._download_training_data_from_data_asset_storage()

                except NotImplementedError as e:
                    raise e

                except FileNotFoundError as e:
                    raise e

                except Exception as e:
                    # do not try Flight if we are on the cloud
                    if self._wml_client is not None:
                        if not self._wml_client.ICP:
                            raise e

                    elif os.environ.get('USER_ACCESS_TOKEN') is None and os.environ.get('RUNTIME_ENV_ACCESS_TOKEN_FILE') is None:
                        raise CannotReadSavedRemoteDataBeforeFit()

                    data = self._download_data_from_flight_service(data_location=self,
                                                                   binary=binary,
                                                                   read_to_file=read_to_file,
                                                                   flight_parameters=flight_parameters,
                                                                   headers=headers,
                                                                   enable_sampling=enable_sampling,
                                                                   sampling_type=sampling_type,
                                                                   number_of_batch_rows=number_of_batch_rows,
                                                                   return_data_as_iterator=return_data_as_iterator,
                                                                   _return_subsampling_stats=_return_subsampling_stats,
                                                                   total_size_limit=total_size_limit,
                                                                   total_nrows_limit=total_nrows_limit,
                                                                   total_percentage_limit=total_percentage_limit)

        elif self.type == DataConnectionTypes.FS:

            if self._obm:
                data = self._download_obm_data_from_file_system()
            else:
                data = self._download_training_data_from_file_system()

        elif self.type == DataConnectionTypes.CA or self.type == DataConnectionTypes.CN:
            if getattr(self._wml_client, 'ICP', False) and self.type == DataConnectionTypes.CN:
                raise ContainerTypeNotSupported()  # block Container type on CPD

            if use_flight and not self._obm:
                # Workaround for container connection type, we need to fetch COS details from space/project
                if self.type == DataConnectionTypes.CN:
                    # note: update flight parameters only if `connection_properties` was not set earlier
                    #       (e.x. by wml/autoi)
                    if not flight_parameters.get('connection_properties'):
                        flight_parameters = self._update_flight_parameters_with_connection_details(flight_parameters)

                data = self._download_data_from_flight_service(data_location=self,
                                                               binary=binary,
                                                               read_to_file=read_to_file,
                                                               flight_parameters=flight_parameters,
                                                               headers=headers,
                                                               enable_sampling=enable_sampling,
                                                               sampling_type=sampling_type,
                                                               number_of_batch_rows=number_of_batch_rows,
                                                               return_data_as_iterator=return_data_as_iterator,
                                                               _return_subsampling_stats=_return_subsampling_stats,
                                                               total_size_limit=total_size_limit,
                                                               total_nrows_limit=total_nrows_limit,
                                                               total_percentage_limit=total_percentage_limit)

            else:  # backward compatibility
                try:
                    with all_logging_disabled():
                        if self._check_if_connection_asset_is_s3():
                            cos_client = self._init_cos_client()
                            try:
                                if self._obm:
                                    data = self._download_obm_data_from_cos(cos_client=cos_client)

                                else:
                                    data = self._download_data_from_cos(cos_client=cos_client)

                            except Exception as cos_access_exception:
                                raise ConnectionError(
                                    f"Unable to access data object in cloud object storage with credentials supplied. "
                                    f"Error: {cos_access_exception}")
                        else:
                            data = self._download_data_from_nfs_connection()

                except Exception as e:
                    # do not try Flight is we are on the cloud
                    if self._wml_client is not None:
                        if not self._wml_client.ICP:
                            raise e

                    elif os.environ.get('USER_ACCESS_TOKEN') is None and os.environ.get('RUNTIME_ENV_ACCESS_TOKEN_FILE') is None:
                        raise CannotReadSavedRemoteDataBeforeFit()

                    data = self._download_data_from_flight_service(data_location=self,
                                                                   binary=binary,
                                                                   read_to_file=read_to_file,
                                                                   flight_parameters=flight_parameters,
                                                                   headers=headers,
                                                                   enable_sampling=enable_sampling,
                                                                   sampling_type=sampling_type,
                                                                   number_of_batch_rows=number_of_batch_rows,
                                                                   _return_subsampling_stats=_return_subsampling_stats,
                                                                   total_size_limit=total_size_limit,
                                                                   total_nrows_limit=total_nrows_limit,
                                                                   total_percentage_limit=total_percentage_limit)

        if getattr(self._wml_client, '_internal', False):
            pass  # don't remove additional params if client is used internally
        else:
            # note: remove additional params and inline credentials added by _check_if_connection_asset_is_s3:
            [delattr(self.connection, attr) for attr in
             ['secret_access_key', 'access_key_id', 'endpoint_url', 'cos_type'] if
             hasattr(self.connection, attr)]
            # end note

        # create data statistics if data were not downloaded with flight:
        if not isinstance(data, tuple) and _return_subsampling_stats:
            data = (data, {"data_batch_size": sys.getsizeof(data),
                           "data_batch_nrows": len(data)})

        if binary:
            return data

        if raw or (self.auto_pipeline_params.get('prediction_column') is None
                   and self.auto_pipeline_params.get('prediction_columns') is None
                   and self.auto_pipeline_params.get('feature_columns') is None):
            return data

        else:
            if with_holdout_split:  # when this connection is training one
                if return_data_as_iterator:
                    raise WMLClientError("The flags `return_data_as_iterator` and `with_holdout_split` cannot be set both in the same time.")

                if _return_subsampling_stats:
                    X_train, X_holdout, y_train, y_holdout = self._recreate_holdout(data=data[0])
                    return X_train, X_holdout, y_train, y_holdout, data[1]
                else:
                    X_train, X_holdout, y_train, y_holdout = self._recreate_holdout(data=data)
                    return X_train, X_holdout, y_train, y_holdout

            else:  # when this data connection is a test / holdout one
                if return_data_as_iterator:
                    return data

                if _return_subsampling_stats:
                    if self.auto_pipeline_params.get('prediction_columns') or \
                            not self.auto_pipeline_params.get('prediction_column') or \
                            (self.auto_pipeline_params.get('prediction_column') and self.auto_pipeline_params.get(
                                'prediction_column') not in data[0].columns):
                        # timeseries dataset does not have prediction columns. Whole data set is returned:
                        test_X = data
                        return test_X
                    else:
                        test_X, test_y = self._recreate_holdout(data=data[0], with_holdout_split=False)
                        test_X[self.auto_pipeline_params.get('prediction_column', 'prediction_column')] = test_y
                        return test_X, data[1]

                else: # when this data connection is a test / holdout one and no subsampling stats are needed
                    if self.auto_pipeline_params.get('prediction_columns') or \
                            not self.auto_pipeline_params.get('prediction_column') or \
                            (self.auto_pipeline_params.get('prediction_column') and self.auto_pipeline_params.get(
                                'prediction_column') not in data.columns):
                        # timeseries dataset does not have prediction columns. Whole data set is returned:
                        test_X = data
                    else:
                        test_X, test_y = self._recreate_holdout(data=data, with_holdout_split=False)
                        test_X[self.auto_pipeline_params.get('prediction_column', 'prediction_column')] = test_y
                    return test_X  # return one dataframe

    def write(self, data: Union[str, 'DataFrame'], remote_name: str = None, **kwargs) -> None:
        """Upload file to a remote data storage.

        :param data: local path to the dataset or pandas.DataFrame with data
        :type data: str

        :param remote_name: name that dataset should be stored with in remote data storage
        :type remote_name: str
        """
        # enables flight automatically for CP4D 4.0.x, for 3.0 and 3.5 we do not have a flight service there
        use_flight = kwargs.get(
            'use_flight',
            bool((self._wml_client is not None or 'USER_ACCESS_TOKEN' in os.environ or 'RUNTIME_ENV_ACCESS_TOKEN_FILE' in os.environ) and (
                    self._wml_client.ICP_40 or self._wml_client.ICP_45 or self._wml_client.ICP_46 or self._wml_client.ICP_47)))

        flight_parameters = kwargs.get('flight_parameters', {})

        impersonate_header = kwargs.get('impersonate_header', None)

        headers = None
        if self._wml_client is None:
            token = self._get_token_from_environment()
            if token is None:
                raise ConnectionError("WML client missing. Please initialize WML client and pass it to "
                                      "DataConnection._wml_client property to be able to use this functionality.")

            else:
                headers = {'Authorization': f'Bearer {token}'}
        elif impersonate_header is not None:
            headers = self._wml_client._get_headers()
            headers['impersonate'] = impersonate_header

        # TODO: Remove S3 implementation
        if self.type == DataConnectionTypes.S3:
            raise ConnectionError("S3 DataConnection is deprecated! Please use data_asset_id instead.")

        elif self.type == DataConnectionTypes.CA or self.type == DataConnectionTypes.CN:
            if getattr(self._wml_client, 'ICP', False) and self.type == DataConnectionTypes.CN:
                raise ContainerTypeNotSupported()  # block Container type on CPD

            if self._check_if_connection_asset_is_s3():
                # do not try Flight if we are on the cloud
                if self._wml_client is not None and not self._wml_client.ICP and not use_flight:  # CLOUD
                    updated_remote_name = self._get_path_with_remote_name(self._to_dict(), remote_name)
                    cos_resource_client = self._init_cos_client()
                    if isinstance(data, str):
                        with open(data, "rb") as file_data:
                            cos_resource_client.Object(self.location.bucket, updated_remote_name).upload_fileobj(
                                Fileobj=file_data)

                    elif isinstance(data, DataFrame):
                        # note: we are saving csv in memory as a file and stream it to the COS
                        buffer = io.StringIO()
                        data.to_csv(buffer, index=False)
                        buffer.seek(0)

                        with buffer as f:
                            cos_resource_client.Object(self.location.bucket, updated_remote_name).upload_fileobj(
                                Fileobj=io.BytesIO(bytes(f.read().encode())))

                    else:
                        raise TypeError("data should be either of type \"str\" or \"pandas.DataFrame\"")
                # CP4D
                else:
                    # Workaround for container connection type, we need to fetch COS details from space/project
                    if self.type == DataConnectionTypes.CN:
                        # note: update flight parameters only if `connection_properties` was not set earlier
                        #       (e.x. by wml/autoi)
                        if not flight_parameters.get('connection_properties'):
                            flight_parameters = self._update_flight_parameters_with_connection_details(flight_parameters)

                    if isinstance(data, str):
                        self._upload_data_via_flight_service(file_path=data,
                                                             data_location=self,
                                                             remote_name=remote_name,
                                                             flight_parameters=flight_parameters,
                                                             headers=headers)

                    elif isinstance(data, DataFrame):
                        # note: we are saving csv in memory as a file and stream it to the COS
                        self._upload_data_via_flight_service(data=data,
                                                             data_location=self,
                                                             remote_name=remote_name,
                                                             flight_parameters=flight_parameters,
                                                             headers=headers)

                    else:
                        raise TypeError("data should be either of type \"str\" or \"pandas.DataFrame\"")

            else:
                if self._wml_client is not None and not self._wml_client.ICP and not use_flight:  # CLOUD
                    raise ConnectionError("Connections other than COS are not supported on a cloud yet.")
                # CP4D
                else:
                    if isinstance(data, str):
                        self._upload_data_via_flight_service(file_path=data,
                                                             data_location=self,
                                                             remote_name=remote_name,
                                                             flight_parameters=flight_parameters,
                                                             headers=headers)

                    elif isinstance(data, DataFrame):
                        # note: we are saving csv in memory as a file and stream it to the COS
                        self._upload_data_via_flight_service(data=data,
                                                             data_location=self,
                                                             remote_name=remote_name,
                                                             flight_parameters=flight_parameters,
                                                             headers=headers)

                    else:
                        raise TypeError("data should be either of type \"str\" or \"pandas.DataFrame\"")

            if getattr(self._wml_client, '_internal', False):
                pass  # don't remove additional params if client is used internally
            else:
                # note: remove additional params and inline credentials added by _check_if_connection_asset_is_s3:
                [delattr(self.connection, attr) for attr in ['secret_access_key', 'access_key_id', 'endpoint_url', 'cos_type'] if
                    hasattr(self.connection, attr)]
                # end note
        elif self.type == DataConnectionTypes.DS:
            if self._wml_client is not None and not self._wml_client.ICP and not use_flight:  # CLOUD
                raise ConnectionError("Write of data for Data Asset is not supported on Cloud.")

            elif self._wml_client is not None:
                if isinstance(data, str):
                    self._upload_data_via_flight_service(file_path=data,
                                                         data_location=self,
                                                         remote_name=remote_name,
                                                         flight_parameters=flight_parameters,
                                                         headers=headers)

                elif isinstance(data, DataFrame):
                    # note: we are saving csv in memory as a file and stream it to the COS
                    self._upload_data_via_flight_service(data=data,
                                                         data_location=self,
                                                         remote_name=remote_name,
                                                         flight_parameters=flight_parameters,
                                                         headers=headers)

                else:
                    raise TypeError("data should be either of type \"str\" or \"pandas.DataFrame\"")

            else:
                self._upload_data_via_flight_service(data=data,
                                                     data_location=self,
                                                     remote_name=remote_name,
                                                     flight_parameters=flight_parameters,
                                                     headers=headers)

    def _init_cos_client(self) -> 'resource':
        """Initiate COS client for further usage."""
        from ibm_botocore.client import Config
        if hasattr(self.connection, 'auth_endpoint') and hasattr(self.connection, 'api_key'):
            cos_client = resource(
                service_name='s3',
                ibm_api_key_id=self.connection.api_key,
                ibm_auth_endpoint=self.connection.auth_endpoint,
                config=Config(signature_version="oauth"),
                endpoint_url=self.connection.endpoint_url
            )

        else:
            cos_client = resource(
                service_name='s3',
                endpoint_url=self.connection.endpoint_url,
                aws_access_key_id=self.connection.access_key_id,
                aws_secret_access_key=self.connection.secret_access_key
            )
        return cos_client

    def _validate_cos_resource(self):
        cos_client = self._init_cos_client()
        try:
            files = cos_client.Bucket(self.location.bucket).objects.all()
            next(x for x in files if x.key == self.location.path)
        except Exception as e:
            raise NotExistingCOSResource(self.location.bucket, self.location.path)

    def _update_flight_parameters_with_connection_details(self, flight_parameters):
        with all_logging_disabled():
            self._check_if_connection_asset_is_s3()
            connection_properties = {
                "bucket": self.location.bucket,
                "url": self.connection.endpoint_url
            }
            if hasattr(self.connection, 'auth_endpoint') and hasattr(self.connection, 'api_key'):
                connection_properties["iam_url"] = self.connection.auth_endpoint
                connection_properties["api_key"] = self.connection.api_key
                connection_properties["resource_instance_id"] = self.connection.resource_instance_id

            else:
                connection_properties["secret_key"] = self.connection.secret_access_key
                connection_properties["access_key"] = self.connection.access_key_id

            flight_parameters.update({"connection_properties": connection_properties})
            flight_parameters.update({"datasource_type": {"entity": {"name": self._datasource_type}}})

        return flight_parameters


# TODO: Remove S3 Implementation for connection
class S3Connection(BaseConnection):
    """Connection class to COS data storage in S3 format.

    :param endpoint_url: S3 data storage url (COS)
    :type endpoint_url: str

    :param access_key_id: access_key_id of the S3 connection (COS)
    :type access_key_id: str, optional

    :param secret_access_key: secret_access_key of the S3 connection (COS)
    :type secret_access_key: str, optional

    :param api_key: API key of the S3 connection (COS)
    :type api_key: str, optional

    :param service_name: service name of the S3 connection (COS)
    :type service_name: str, optional

    :param auth_endpoint: authentication endpoint url of the S3 connection (COS)
    :type auth_endpoint: str, optional
    """

    def __init__(self, endpoint_url: str, access_key_id: str = None, secret_access_key: str = None,
                 api_key: str = None, service_name: str = None, auth_endpoint: str = None,
                 resource_instance_id: str = None, _internal_use=False) -> None:

        if not _internal_use:
            warn(message="S3 DataConnection is deprecated! Please use data_asset_id instead.")

        if (access_key_id is None or secret_access_key is None) and (api_key is None or auth_endpoint is None):
            raise InvalidCOSCredentials(reason='You need to specify (access_key_id and secret_access_key) or'
                                               '(api_key and auth_endpoint)')

        if secret_access_key is not None:
            self.secret_access_key = secret_access_key

        if api_key is not None:
            self.api_key = api_key

        if service_name is not None:
            self.service_name = service_name

        if auth_endpoint is not None:
            self.auth_endpoint = auth_endpoint

        if access_key_id is not None:
            self.access_key_id = access_key_id

        if endpoint_url is not None:
            self.endpoint_url = endpoint_url

        if resource_instance_id is not None:
            self.resource_instance_id = resource_instance_id


class S3Location(BaseLocation):
    """Connection class to COS data storage in S3 format.

    :param bucket: COS bucket name
    :type bucket: str

    :param path: COS data path in the bucket
    :type path: str

    :param excel_sheet: name of excel sheet if pointed dataset is excel file used for Batched Deployment scoring
    :type excel_sheet: str, optional

    :param model_location: path to the pipeline model in the COS
    :type model_location: str, optional

    :param training_status: path to the training status json in COS
    :type training_status: str, optional
    """

    def __init__(self, bucket: str, path: str, **kwargs) -> None:
        self.bucket = bucket
        self.path = path

        if kwargs.get('model_location') is not None:
            self._model_location = kwargs['model_location']

        if kwargs.get('training_status') is not None:
            self._training_status = kwargs['training_status']

        if kwargs.get('excel_sheet') is not None:
            self.sheet_name = kwargs['excel_sheet']
            self.file_format = "xls"

    def _get_file_size(self, cos_resource_client: 'resource') -> 'int':
        try:
            size = cos_resource_client.Object(self.bucket, self.path).content_length
        except ClientError:
            size = 0
        return size

    def get_location(self) -> str:
        if hasattr(self, "file_name"):
            return self.file_name
        else:
            return self.path


class ContainerLocation(BaseLocation):
    """Connection class to default COS in user Project/Space."""

    def __init__(self, path: Optional[str] = None, **kwargs) -> None:
        if path is None:
            self.path = "default_autoai_out"

        else:
            self.path = path

        self.bucket = None

        if kwargs.get('model_location') is not None:
            self._model_location = kwargs['model_location']

        if kwargs.get('training_status') is not None:
            self._training_status = kwargs['training_status']

    def to_dict(self) -> dict:
        _dict = super().to_dict()

        if 'bucket' in _dict and _dict['bucket'] is None:
            del _dict['bucket']

        return _dict

    @classmethod
    def _set_path(cls, path: str) -> 'ContainerLocation':
        location = cls()
        location.path = path
        return location

    def _get_file_size(self):
        pass


class FSLocation(BaseLocation):
    """Connection class to File Storage in CP4D."""

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            self.path = "/{option}/{id}" + f"/assets/auto_ml/auto_ml.{uuid.uuid4()}/wml_data"

        else:
            self.path = path

    @classmethod
    def _set_path(cls, path: str) -> 'FSLocation':
        location = cls()
        location.path = path
        return location

    def _save_file_as_data_asset(self, workspace: 'WorkSpace') -> 'str':

        asset_name = self.path.split('/')[-1]
        if self.path:
            data_asset_details = workspace.wml_client.data_assets.create(asset_name, self.path)
            return workspace.wml_client.data_assets.get_uid(data_asset_details)
        else:
            raise MissingValue('path', reason="Incorrect initialization of class FSLocation")

    def _get_file_size(self, workspace: 'WorkSpace') -> 'int':
        # note if path is not file then returned size is 0
        try:
            # note: try to get file size from remote server
            url = workspace.wml_client.service_instance._href_definitions.get_wsd_model_attachment_href() \
                  + f"/{self.path.split('/assets/')[-1]}"
            path_info_response = requests.head(url, headers=workspace.wml_client._get_headers(),
                                               params=workspace.wml_client._params())
            if path_info_response.status_code != 200:
                raise ApiRequestFailure(u"Failure during getting path details", path_info_response)
            path_info = path_info_response.headers
            if 'X-Asset-Files-Type' in path_info and path_info['X-Asset-Files-Type'] == 'file':
                size = path_info['X-Asset-Files-Size']
            else:
                size = 0
            # -- end note
        except (ApiRequestFailure, AttributeError):
            # note try get size of file from local fs
            size = os.stat(path=self.path).st_size if os.path.isfile(path=self.path) else 0
            # -- end note
        return size


class AssetLocation(BaseLocation):

    def __init__(self, asset_id: str) -> None:
        self._wsd = self._is_wsd()
        self.href = None
        self._initial_asset_id = asset_id
        self.__wml_client = None

        if self._wsd:
            self._asset_name = None
            self._asset_id = None
            self._local_asset_path = None
        else:
            self.id = asset_id

    def _get_bucket(self, client) -> str:
        """Try to get bucket from data asset."""
        connection_id = self._get_connection_id(client)
        conn_details = client.connections.get_details(connection_id)
        bucket = conn_details.get('entity', {}).get('properties', {}).get('bucket')

        if bucket is None:
            asset_details = client.data_assets.get_details(self.id)
            connection_path = asset_details['entity'].get('folder_asset', {}).get('connection_path')
            if connection_path is None:
                attachment_content = self._get_attachment_details(client)
                connection_path = attachment_content.get('connection_path')

            bucket = connection_path.split('/')[1]

        return bucket

    def _get_attachment_details(self, client) -> dict:
        if self.id is None and self.href:
            items = self.href.split('/')
            self.id = items[-1].split('?')[0]

        asset_details = client.data_assets.get_details(self.id)

        if 'attachment_id' in asset_details.get('metadata'):
            attachment_id = asset_details['metadata']['attachment_id']

        else:
            attachment_id = asset_details['attachments'][0]['id']

        attachment_url = client.service_instance._href_definitions.get_data_asset_href(self.id)
        attachment_url = f"{attachment_url}/attachments/{attachment_id}"

        if client.ICP:
            attachment = requests.get(attachment_url, headers=client._get_headers(),
                                      params=client._params())

        else:
            attachment = requests.get(attachment_url, headers=client._get_headers(),
                                      params=client._params())

        if attachment.status_code != 200:
            raise ApiRequestFailure(u"Failure during getting attachment details", attachment)

        return attachment.json()

    def _get_connection_id(self, client) -> str:
        attachment_content = self._get_attachment_details(client)

        return attachment_content.get('connection_id')

    @classmethod
    def _is_wsd(cls):
        if os.environ.get('USER_ACCESS_TOKEN') or os.environ.get('RUNTIME_ENV_ACCESS_TOKEN_FILE'):
            return False

        try:
            from project_lib import Project
            try:
                with all_logging_disabled():
                    access = Project.access()
                return True
            except RuntimeError:
                pass
        except ModuleNotFoundError:
            pass

        return False

    @classmethod
    def _set_path(cls, href: str) -> 'AssetLocation':
        items = href.split('/')
        _id = items[-1].split('?')[0]
        location = cls(_id)
        location.href = href
        return location

    def _get_file_size(self, workspace: 'WorkSpace', *args) -> 'int':
        if self._wsd:
            return self._wsd_get_file_size()
        else:
            asset_info_response = requests.get(
                workspace.wml_client.service_instance._href_definitions.get_data_asset_href(self.id),
                params=workspace.wml_client._params(),
                headers=workspace.wml_client._get_headers())
            if asset_info_response.status_code != 200:
                raise ApiRequestFailure(u"Failure during getting asset details", asset_info_response)
            return asset_info_response.json()['metadata'].get('size')

    def _wsd_setup_local_asset_details(self) -> None:
        if not self._wsd:
            raise NotWSDEnvironment()

        # note: set local asset file from asset_id
        project = get_project()
        project_id = project.get_metadata()["metadata"]["guid"]

        local_assets = project.get_files()

        # note: reuse local asset_id when object is reused more times
        if self._asset_id is None:
            local_asset_id = self._initial_asset_id

        else:
            local_asset_id = self._asset_id
        # --- end note

        if local_asset_id not in str(local_assets):
            raise MissingLocalAsset(local_asset_id, reason="Provided asset_id cannot be found on WS Desktop.")

        else:
            for asset in local_assets:
                if asset['asset_id'] == local_asset_id:
                    asset_name = asset['name']
                    self._asset_name = asset_name
                    self._asset_id = local_asset_id

            local_asset_path = f"{os.path.abspath('.')}/{project_id}/assets/data_asset/{asset_name}"
            self._local_asset_path = local_asset_path

    def _wsd_move_asset_to_server(self, workspace: 'WorkSpace') -> None:
        if not self._wsd:
            raise NotWSDEnvironment()

        if not self._local_asset_path or self._asset_name or self._asset_id:
            self._wsd_setup_local_asset_details()

        remote_asset_details = workspace.wml_client.data_assets.create(self._asset_name, self._local_asset_path)
        self.href = remote_asset_details['metadata']['href']

    def _wsd_get_file_size(self) -> 'int':
        if not self._wsd:
            raise NotWSDEnvironment()

        if not self._local_asset_path or self._asset_name or self._asset_id:
            self._wsd_setup_local_asset_details()
        return os.stat(path=self._local_asset_path).st_size if os.path.isfile(path=self._local_asset_path) else 0

    @classmethod
    def list_wsd_assets(cls):
        if not cls._is_wsd():
            raise NotWSDEnvironment

        project = get_project()
        return project.get_files()

    def to_dict(self) -> dict:
        """Return a json dictionary representing this model."""
        _dict = vars(self).copy()

        if _dict.get('id', False) is None and _dict.get('href'):
            items = self.href.split('/')
            _dict['id'] = items[-1].split('?')[0]

        del _dict['_wsd']
        del _dict[f"_{self.__class__.__name__}__wml_client"]

        if self._wsd:
            del _dict['_asset_name']
            del _dict['_asset_id']
            del _dict['_local_asset_path']

        del _dict['_initial_asset_id']

        return _dict

    @property
    def wml_client(self):
        return self.__wml_client

    @wml_client.setter
    def wml_client(self, var):
        self.__wml_client = var

        if self.__wml_client:
            self.href = self.__wml_client.service_instance._href_definitions.get_base_asset_href(self._initial_asset_id)
        else:
            self.href = f'/v2/assets/{self._initial_asset_id}'

        if not self._wsd:
            if self.__wml_client:
                if self.__wml_client.default_space_id:
                    self.href = f'{self.href}?space_id={self.__wml_client.default_space_id}'
                else:
                    self.href = f'{self.href}?project_id={self.__wml_client.default_project_id}'


class ConnectionAssetLocation(BaseLocation):
    """Connection class to COS data storage.

    :param bucket: COS bucket name
    :type bucket: str

    :param file_name: COS data path in the bucket
    :type file_name: str

    :param model_location: path to the pipeline model in the COS
    :type model_location: str, optional

    :param training_status: path to the training status json in COS
    :type training_status: str, optional
    """

    def __init__(self, bucket: str, file_name: str, **kwargs) -> None:
        self.bucket = bucket
        self.file_name = file_name
        self.path = file_name

        if kwargs.get('model_location') is not None:
            self._model_location = kwargs['model_location']

        if kwargs.get('training_status') is not None:
            self._training_status = kwargs['training_status']

    def _get_file_size(self, cos_resource_client: 'resource') -> 'int':
        try:
            size = cos_resource_client.Object(self.bucket, self.path).content_length
        except ClientError:
            size = 0
        return size

    def to_dict(self) -> dict:
        """Return a json dictionary representing this model."""
        return vars(self)


class ConnectionAsset(BaseConnection):
    """Connection class for Connection Asset.

    :param connection_id: connection asset ID
    :type connection_id: str
    """

    def __init__(self, connection_id: str):
        self.id = connection_id


class NFSConnection(BaseConnection):
    """Connection class to file storage in CP4D of NFS format.

    :param asset_id: asset ID from the project on CP4D
    :type asset_id: str
    """

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.id = asset_id


class NFSLocation(BaseLocation):
    """Location class to file storage in CP4D of NFS format.

    :param path: data path form the project on CP4D
    :type path: str
    """

    def __init__(self, path: str):
        self.path = path
        self.id = None
        self.file_name = None

    def _get_file_size(self, workspace: 'Workspace', *args) -> 'int':
        params = workspace.wml_client._params().copy()
        params['path'] = self.path
        params['detail'] = 'true'

        href = workspace.wml_client.connections._href_definitions.get_connection_by_id_href(self.id) + '/assets'
        asset_info_response = requests.get(href,
                                           params=params, headers=workspace.wml_client._get_headers(None))
        if asset_info_response.status_code != 200:
            raise Exception(u"Failure during getting asset details", asset_info_response.json())
        return asset_info_response.json()['details']['file_size']

    def get_location(self) -> str:
        if hasattr(self, "file_name"):
            return self.file_name
        else:
            return self.path


class CP4DAssetLocation(AssetLocation):
    """Connection class to data assets in CP4D.

    :param asset_id: asset ID from the project on CP4D
    :type asset_id: str
    """

    def __init__(self, asset_id: str) -> None:
        super().__init__(asset_id)
        warning_msg = ("Depreciation Warning: Class CP4DAssetLocation is no longer supported and will be removed."
                       "Use AssetLocation instead.")
        print(warning_msg)

    def _get_file_size(self, workspace: 'WorkSpace', *args) -> 'int':
        return super()._get_file_size(workspace)


class WMLSAssetLocation(AssetLocation):
    """Connection class to data assets in WML Server.

    :param asset_id: asset ID of the file loaded on space in WML Server
    :type asset_id: str
    """

    def __init__(self, asset_id: str) -> None:
        super().__init__(asset_id)
        warning_msg = ("Depreciation Warning: Class WMLSAssetLocation is no longer supported and will be removed."
                       "Use AssetLocation instead.")
        print(warning_msg)

    def _get_file_size(self, workspace: 'WorkSpace', *args) -> 'int':
        return super()._get_file_size(workspace)


class CloudAssetLocation(AssetLocation):
    """Connection class to data assets as input data references to batch deployment job on Cloud.

    :param asset_id: asset ID of the file loaded on space on Cloud
    :type asset_id: str
    """

    def __init__(self, asset_id: str) -> None:
        super().__init__(asset_id)
        self.href = self.href
        warning_msg = ("Depreciation Warning: Class CloudAssetLocation is no longer supported and will be removed."
                       "Use AssetLocation instead.")
        print(warning_msg)

    def _get_file_size(self, workspace: 'WorkSpace', *args) -> 'int':
        return super()._get_file_size(workspace)


class WSDAssetLocation(BaseLocation):
    """Connection class to data assets in WS Desktop.

    :param asset_id: asset ID from the project on WS Desktop
    :type asset_id: str
    """

    def __init__(self, asset_id: str) -> None:
        self.href = None
        self._asset_name = None
        self._asset_id = None
        self._local_asset_path = None
        self._initial_asset_id = asset_id
        self.__wml_client = None

        warning_msg = ("Depreciation Warning: Class WSDAssetLocation is no longer supported and will be removed."
                       "Use AssetLocation instead.")
        print(warning_msg)

    @classmethod
    def list_assets(cls):
        project = get_project()
        return project.get_files()

    def _setup_local_asset_details(self) -> None:
        # note: set local asset file from asset_id
        project = get_project()
        project_id = project.get_metadata()["metadata"]["guid"]

        local_assets = project.get_files()

        # note: reuse local asset_id when object is reused more times
        if self._asset_id is None:
            local_asset_id = self.href.split('/')[3].split('?space_id')[0]

        else:
            local_asset_id = self._asset_id
        # --- end note

        if local_asset_id not in str(local_assets):
            raise MissingLocalAsset(local_asset_id, reason="Provided asset_id cannot be found on WS Desktop.")

        else:
            for asset in local_assets:
                if asset['asset_id'] == local_asset_id:
                    asset_name = asset['name']
                    self._asset_name = asset_name
                    self._asset_id = local_asset_id

            local_asset_path = f"{os.path.abspath('.')}/{project_id}/assets/data_asset/{asset_name}"
            self._local_asset_path = local_asset_path

    def _move_asset_to_server(self, workspace: 'WorkSpace') -> None:
        if not self._local_asset_path or self._asset_name or self._asset_id:
            self._setup_local_asset_details()

        remote_asset_details = workspace.wml_client.data_assets.create(self._asset_name, self._local_asset_path)
        self.href = remote_asset_details['metadata']['href']

    @classmethod
    def _set_path(cls, href: str) -> 'WSDAssetLocation':
        location = cls('.')
        location.href = href
        return location

    @property
    def wml_client(self):
        return self.__wml_client

    @wml_client.setter
    def wml_client(self, var):
        self.__wml_client = var

        if self.__wml_client:
            self.href = self.__wml_client.service_instance._href_definitions.get_base_asset_href(self._initial_asset_id)
        else:
            self.href = f'/v2/assets/{self._initial_asset_id}'

    def to_dict(self) -> dict:
        """Return a json dictionary representing this model."""
        _dict = vars(self).copy()
        del _dict['_asset_name']
        del _dict['_asset_id']
        del _dict['_local_asset_path']
        del _dict[f"_{self.__class__.__name__}__wml_client"]
        del _dict['_initial_asset_id']

        return _dict

    def _get_file_size(self) -> 'int':
        if not self._local_asset_path or self._asset_name or self._asset_id:
            self._setup_local_asset_details()
        return os.stat(path=self._local_asset_path).st_size if os.path.isfile(path=self._local_asset_path) else 0


class DeploymentOutputAssetLocation(BaseLocation):
    """Connection class to data assets where output of batch deployment will be stored.

    :param name: name of .csv file which will be saved as data asset
    :type name: str
    :param description: description of the data asset
    :type description: str, optional
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description


class DatabaseLocation(BaseLocation):
    """Location class to Database.

    :param schema_name: database schema name
    :type schema_name: str
    :param table_name: database table name
    :type table_name: str
    """

    def __init__(self, schema_name: str, table_name: str, **kwargs) -> None:
        self.schema_name = schema_name
        self.table_name = table_name

    def _get_file_size(self) -> None:
        raise NotImplementedError()
