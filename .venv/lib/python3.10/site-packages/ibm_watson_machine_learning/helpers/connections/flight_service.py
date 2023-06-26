#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


import os
import logging
import json
import sys
import time
import threading
import queue
import pandas as pd

from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Iterable, Generator

from ibm_watson_machine_learning.utils.autoai.errors import InvalidSamplingType, CorruptedData
from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, SamplingTypes
from ibm_watson_machine_learning.wml_client_error import (
    DataStreamError, WrongLocationProperty, WrongFileLocation, SpaceIDandProjectIDCannotBeNone, EmptyDataSource)
from ibm_watson_machine_learning.utils.utils import is_lib_installed, prepare_interaction_props_for_cos

is_lib_installed(lib_name='pyarrow', install=True)
import pyarrow as pa
from pyarrow import flight
from pyarrow.lib import ArrowException
from warnings import warn
from math import ceil


logger = logging.getLogger("automl-pod")

DEFAULT_PARTITIONS_NUM = 4
DEFAULT_BATCH_SIZE_FLIGHT_COMMAND = 10000
DEFAULT_BATCH_SIZE_FLIGHT_COMMAND_BINARY_READ = 1000


class FakeCallback:
    def __init__(self):
        self.logger = logger

    def status_message(self, msg: str):
        self.logger.debug(msg)


class FlightConnection:
    """FlightConnection object unify the work for data reading from different types of data sources,
    including databases. It uses a Flight Service and `pyarrow` library to connect and transfer the data.

    .. note::
        All available and supported connection types could be found on:
        https://connectivity-matrix.us-south.cf.test.appdomain.cloud

    :param headers: WML authorization headers to connect with Flight Service
    :type headers: dict

    :param project_id: ID of project
    :type project_id: str

    :param space_id: ID of space
    :type space_id: str

    :param label: Y column name, it is required for subsampling
    :type label: str

    :param sampling_type: a sampling strategy required of choice
    :type sampling_type: str

    :param learning_type: type of the dataset: 'classification', 'multiclass', 'regression', needed for resampling,
        if value is equal to None 'first_n_records' strategy will be used no mather what is specified in
        'sampling_type'
    :type learning_type: str

    :param data_location: data location information passed by user
    :type data_location: dict, optional

    :param enable_subsampling: tells to activate sampling mode for large data
    :type enable_subsampling: bool, optional

    :param callback: required for sending messages
    :type callback: StatusCallback, optional

    :param data_batch_size_limit: upper limit for data in one batch of data that should be downloaded in Bytes,
        default: 1GB
    :type data_batch_size_limit: int, optional

    :param logical_batch_size_limit: upper limit for logical batch when subsampling is turned on (in Bytes),
        default 2GB, the logical batch is the batch that is merged to the subsampled batch (eg. 2GB + 1GB) and then
        subsampling is performed on top of that 3GBs and 1GB batch (subsampled one) is rewritten again
    :type logical_batch_size_limit: int, optional

    :param flight_parameters: pure unchanged flight service parameters that need to be passed to the service
    :type flight_parameters: dict, optional

    :param fallback_to_one_connection: indicates if in case of failure we should switch to the one connection
        and try again, default `True`
    :type fallback_to_one_connection: bool, optional

    :param return_subsampling_stats: indicates whether return batch data stats: dataset size, no. of batches,
        applicable only if subsampling is enabled, default `False`
    :type return_subsampling_stats: bool, optional

    :param total_size_limit: upper limit for overall data that should be downloaded in Bytes, default: 1GB,
        if more than one of: `total_size_limit`, `total_nrows_limit`, `total_percentage_limit` are set,
        then data are limited to the lower threshold, if None, then all data are downloaded in batches
        in `iterable_read` method
    :type total_size_limit: int, optional

    :param total_nrows_limit: upper limit for overall data that should be downloaded in number of rows,
        if more than one of: `total_size_limit`, `total_nrows_limit`, `total_percentage_limit` are set,
        then data are limited to the lower threshold
    :type total_nrows_limit: int, optional

    :param total_percentage_limit: upper limit for overall data that should be downloaded in percent of all dataset,
        must be a float number between 0 and 1, if more than one of: `total_size_limit`, `total_nrows_limit`,
        `total_percentage_limit` are set, then data are limited to the lower threshold
    :type total_percentage_limit: float, optional
    """

    def __init__(self,
                 headers: dict,
                 sampling_type: str,
                 label: str,
                 learning_type: str,
                 params: dict,
                 project_id: Optional[str] = None,
                 space_id: Optional[str] = None,
                 asset_id: Optional[str] = None,
                 connection_id: Optional[str] = None,
                 data_location: Optional[dict] = None,
                 enable_subsampling: Optional[bool] = False,
                 callback: Optional['Callback'] = None,
                 data_batch_size_limit: Optional[int] = 1073741824,  # 1GB in Bytes 
                 logical_batch_size_limit: Optional[int] = None,
                 flight_parameters: dict = None,
                 extra_interaction_properties: dict = None,
                 fallback_to_one_connection: Optional[bool] = True,
                 number_of_batch_rows: int = None,
                 stop_after_first_batch: bool = False,
                 return_subsampling_stats: bool = False,
                 total_size_limit=1073741824,  # 1GB in Bytes
                 total_nrows_limit=None,
                 total_percentage_limit=1.0,
                 **kwargs
                 ) -> None:

        if project_id is None and space_id is None:
            raise SpaceIDandProjectIDCannotBeNone(
                reason="'space_id' and 'project_id' are None. Please set one of them.")

        self.headers = headers  # WML authorization headers

        self.number_of_batch_rows = number_of_batch_rows
        self.stop_after_first_batch = stop_after_first_batch

        # backward compatibility: 
        if kwargs.get('data_size_limit'):
            warn("The parameters data_size_limit in FlightConnection is deprecated. Use total_size_limit instead.")
            self.total_size_limit = kwargs.get('data_size_limit')

        # Note: Upper bound limitation for data in memory
        self.data_batch_size_limit = data_batch_size_limit  # size of normal or subsampled batch in RAM (Bytes)

        self.total_size_limit = total_size_limit
        self.total_nrows_limit = total_nrows_limit
        self.yielded_nrows = 0
        if self.total_size_limit:
            self.data_batch_size_limit = min(self.total_size_limit, self.data_batch_size_limit)

        if self.total_nrows_limit:
            if self.number_of_batch_rows:
                self.number_of_batch_rows = min(self.total_nrows_limit, self.number_of_batch_rows)
            elif self.total_nrows_limit < DEFAULT_BATCH_SIZE_FLIGHT_COMMAND:  # 10000 is default_batch_size in Flight Service command
                self.number_of_batch_rows = self.total_nrows_limit

        if not isinstance(total_percentage_limit, float) or \
                (isinstance(total_percentage_limit, float) and (
                        total_percentage_limit <= 0.0 or total_percentage_limit > 1.0)):
            raise ValueError("Invalid `total_percentage_limit` parameter's value. "
                             "The `total_percentage_limit` need to be float between 0.0 and 1.0.")
        else:
            self.total_percentage_limit = total_percentage_limit

        # --- end note

        # callback is used in the backend to send status messages
        self.callback = callback if callback is not None else FakeCallback()

        # Handle logical_batch_size_limit parameter set up
        if logical_batch_size_limit is None:
            # Set the logical batch size regarding sample size limit on given hardware (tshirt size)
            self.logical_batch_size_limit = 2 * self.data_batch_size_limit  # two times the size of larger not subsampled batch in RAM (Bytes)
        else:
            self.logical_batch_size_limit = logical_batch_size_limit

        # Note: Variables from AutoAI training
        self.sampling_type = sampling_type
        self.label = label
        self.learning_type = learning_type
        self.params = params
        self.project_id = project_id
        self.space_id = space_id
        self.asset_id = asset_id
        self.connection_id = connection_id
        self.data_source_type = None
        # --- end note

        self.data_location = data_location

        # Note: control and store variables of flight reading mechanism
        self.lock_read = threading.Lock()
        self.stop_reading = False
        self.row_size = 0
        self.threads_exceptions: List['str'] = []
        self.q = queue.Queue()
        # a threading.Condition() to notify of q or
        # stop_reading changes
        self.read_status_change = threading.Condition()

        self.subsampled_data: 'pd.DataFrame' = pd.DataFrame()
        self.data: 'pd.DataFrame' = pd.DataFrame()
        self.enable_subsampling = enable_subsampling
        self.return_subsampling_stats = return_subsampling_stats
        self.total_size = 0  # total size of downloaded data in Bytes (only in single thread)
        self.downloaded_data_size = 0  # total size of downloaded data in Bytes (every case)
        self.downloaded_data_nrows = 0  # total number  of downloaded data in rows (every case)

        self.batch_queue = []

        self.flight_parameters = flight_parameters if flight_parameters is not None else {}
        self._wml_client = kwargs.get('_wml_client')

        # user can define how many parallel connections initiate to database
        self.max_flight_batch_number = self.params.get('n_parallel_data_connections', DEFAULT_PARTITIONS_NUM)
        if 'num_partitions' in self.flight_parameters:
            self.max_flight_batch_number = self.flight_parameters['num_partitions']
        # --- end note

        self.fallback_to_one_connection = fallback_to_one_connection

        self.read_binary = False
        self.write_binary = False

        self._infer_as_varchar = 'false'  # by default set infer_as_varchar to false in flight command. If None - the infer_as_varchar parameter won't be send.

        additional_connection_args = {}
        if os.environ.get('TLS_ROOT_CERTS_PATH'):
            additional_connection_args['tls_root_certs'] = os.environ.get('TLS_ROOT_CERTS_PATH')

        self.extra_interaction_properties = extra_interaction_properties

        self.flight_location = None
        self.flight_port = None

        self._set_default_flight_location()

        self.flight_client = flight.FlightClient(
            location=f"grpc+tls://{self.flight_location}:{self.flight_port}",
            disable_server_verification=True,
            override_hostname=self.flight_location,
            **additional_connection_args
        )

        self.empty_data_threads = set()
        # note: client as property and setter for dynamic href creation for AssetLocation

    @property
    def infer_as_varchar(self):
        return self._infer_as_varchar

    @infer_as_varchar.setter
    def infer_as_varchar(self, var):
        if var is None:
            self._infer_as_varchar = var
        elif var in ('true', True, 'false', False):
            self._infer_as_varchar = str(var).lower()
        else:
            raise ValueError("FlightConnection.infer_as_varchar property received invalid value."
                             "A valid value is one of (None, 'true', 'false')")

    def _q_put_nowait(self, item):
        # we are not interested in q size increase, so no Condition waiting
        self.q.put_nowait(item)

    def _q_get(self, **kwargs):
        # return item in q, and notify interest threads that q is changing
        item = None
        if self.q.qsize() == 0:
            item = self.q.get(**kwargs)
        else:
            with self.read_status_change:
                item = self.q.get(**kwargs)
                self.read_status_change.notify_all()
        return item

    def _q_reset(self):
        with self.read_status_change:
            self.q = queue.Queue()
            self.read_status_change.notify_all()

    def _set_stop_reading(self, value):
        # we don't need to change value while holding the lock, we just want
        # to notify waiting threads
        with self.read_status_change:
            self.stop_reading = value
            self.read_status_change.notify_all()

    def _set_default_flight_location(self) -> None:
        """Try to set default flight location and port from WS."""
        if not os.environ.get(
                'FLIGHT_SERVICE_LOCATION') and self._wml_client and self._wml_client.CLOUD_PLATFORM_SPACES:
            try:
                flight_location = self._wml_client.PLATFORM_URLS_MAP[self._wml_client.wml_credentials['url']].replace(
                    'https://', '')
            except Exception as e:
                if self._wml_client.wml_credentials['url'] in self._wml_client.PLATFORM_URLS_MAP.values():
                    flight_location = self._wml_client.wml_credentials['url'].replace('https://', '')
                else:
                    raise e
            flight_port = 443
        else:
            host = os.environ.get('ASSET_API_SERVICE_HOST', os.environ.get('CATALOG_API_SERVICE_HOST'))

            if host is None or 'api.' not in host:
                default_service_url = os.environ.get('RUNTIME_FLIGHT_SERVICE_URL', 'grpc+tls://wdp-connect-flight:443')
                default_service_url = default_service_url.split('//')[-1]
                flight_location = os.environ.get('FLIGHT_SERVICE_LOCATION')
                flight_port = os.environ.get('FLIGHT_SERVICE_PORT')

                if flight_location is None or flight_location == '':
                    flight_location = default_service_url.split(':')[0]

                if flight_port is None or flight_port == '':
                    flight_port = default_service_url.split(':')[-1]

            else:
                flight_location = host
                flight_port = '443'

        self.flight_location = flight_location
        self.flight_port = flight_port

        logger.debug(f"Flight location: {self.flight_location}")
        logger.debug(f"Flight port: {self.flight_port}")

    def authenticate(self) -> 'flight.ClientAuthHandler':
        """Create an authenticator object for Flight Service."""

        class TokenClientAuthHandler(flight.ClientAuthHandler):
            """Authenticator implementation from pyarrow flight."""

            def __init__(self, token, _type: str, impersonate: bool = False):
                super().__init__()
                if impersonate:
                    self.token = bytes(f'{token}', 'utf-8')
                else:
                    self.token = bytes(f'{_type} ' + token, 'utf-8')

            def authenticate(self, outgoing, incoming):
                outgoing.write(self.token)
                self.token = incoming.read()

            def get_token(self):
                logger.debug(f"Flight service get_token() {self.token}")
                return self.token

        if 'Bearer' in self.headers.get('Authorization', ''):
            if "impersonate" in self.headers:
                authorization_header = self.headers.get('Authorization', 'Bearer  ')
                impersonate_header = self.headers.get('impersonate')
                auth_json_str = json.dumps(dict(authorization=authorization_header, impersonate=impersonate_header))
                return TokenClientAuthHandler(token=auth_json_str, _type='json_string', impersonate=True)
            else:
                return TokenClientAuthHandler(token=self.headers.get('Authorization', 'Bearer  ').split('Bearer ')[-1],
                                              _type='Bearer')

        elif 'Basic' in self.headers.get('Authorization', ''):
            if "impersonate" in self.headers:
                authorization_header = self.headers.get('Authorization', 'Basic  ')
                impersonate_header = self.headers.get('impersonate')
                auth_json_str = json.dumps(dict(authorization=authorization_header, impersonate=impersonate_header))
                return TokenClientAuthHandler(token=auth_json_str, _type='json_string', impersonate=True)
            else:
                return TokenClientAuthHandler(token=self.headers.get('Authorization', 'Basic  ').split('Basic ')[-1],
                                              _type='Basic')

        else:
            return TokenClientAuthHandler(token=self.headers.get('Authorization'), _type='Bearer')

    def get_endpoints(self) -> Iterable[List['flight.FlightEndpoint']]:
        """Listing all available Flight Service endpoints (one endpoint corresponds to one batch)"""

        max_auth_waiting_time = 180
        real_waiting_time = 0
        count_auth_retries_power = 4
        retry_authentication = True

        while retry_authentication:
            try:
                retry_authentication = False
                self.flight_client.authenticate(self.authenticate())
            except Exception as e:
                if "failed to connect to all addresses" in str(e):
                    retry_authentication = True  # wait up to 180 seconds - Flight Service can be restarting issue #30564
                else:
                    retry_authentication = False

                if retry_authentication and real_waiting_time < max_auth_waiting_time:
                    logger.debug(f"Cannot connect to Flight Service in {real_waiting_time}s,"
                                 f" attempting to retry the authentication. ")
                    waiting_time = 2**count_auth_retries_power
                    real_waiting_time += waiting_time
                    time.sleep(waiting_time)
                    count_auth_retries_power += 1
                else:
                    # suggest CPD users to check the Flight variables
                    if hasattr(self._wml_client, 'ICP'):
                        if self._wml_client.ICP:
                            raise ConnectionError(
                                f"Cannot connect to the Flight service. Please make sure you set correct "
                                f"FLIGHT_SERVICE_LOCATION and FLIGHT_SERVICE_PORT environmental variables.\n"
                                f"If you are trying to connect to FIPS-enabled cluster, "
                                f"please set the following as environment variable and try again:\n"
                                f"GRPC_SSL_CIPHER_SUITES="
                                f"ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384 "
                                f"Error: {e}")
                    else:
                        raise ConnectionError(f"Cannot connect to the Flight service. Error: {e}")

        count = 0
        retry = True
        while retry:
            try:
                retry = False
                count += 1
                for source_command in self._select_source_command():
                    info = self.flight_client.get_flight_info(
                        flight.FlightDescriptor.for_command(source_command)
                    )

                    yield info.endpoints

            except flight.FlightInternalError as e:
                logger.debug(f"Caught FlightInternalError in get_endpoints: {str(e)}")
                if 'CDICO2034E' in str(e):
                    if 'The property [infer_as_varchar] is not supported.' in str(e):
                        # Don't send infer_as_varchar in flight command and try again.
                        self.infer_as_varchar = None
                        retry = True
                    elif 'The property [quote_character]' in str(e):
                        # Don't send quote_character in flight command if it is not yet supported and try again.
                        self.params['quote_character'] = None
                        retry = True
                    else:
                        raise WrongLocationProperty(reason=str(e))

                    if count > 1:
                        logger.debug(f"Reached the max retry times ${count}")
                        raise e
                elif any(err_code in str(e)
                         for err_code in ['CDICO2016E', 'CDICO2026E', 'CDICO2027E']):
                    retry = True
                    if count < 3:
                        logger.debug(f"Retry to get the endpoints of flight service at the ${count} time")
                        time.sleep(3)
                    else:
                        logger.debug(f"Reached the max retry times ${count}")
                        raise e
                elif 'CDICO2015E' in str(e):
                    raise WrongFileLocation(reason=str(e))
                elif 'CDICO9999E' in str(e):
                    raise WrongLocationProperty(reason=str(e))
                else:
                    raise e
            except pa.lib.ArrowInvalid as e:
                logger.debug(f"Caught ArrowInvalid in get_endpoints: {str(e)}")
                if any(err_msg in str(e)
                         for err_msg in ['No asset found with the id', 'Asset with ID']):
                    retry = True
                    if count < 3:
                        logger.debug(f"Retry to get the endpoints of flight service at the ${count} time")
                        time.sleep(3)
                    else:
                        logger.debug(f"Reached the max retry times ${count}")
                        raise e
                else:
                    raise e

    def _get_data(self,
                  thread_number: int,
                  endpoint: 'flight.FlightEndpoint') -> None:
        """Read data from Flight Service (only one batch).

        :param thread_number: specific number of the downloading thread
        :type thread_number: int

        :param endpoint: flight endpoint
        :type endpoint: flight.FlightEndpoint

        :return: batch data
        :rtype: pandas.DataFrame
        """
        try:
            reader = self.flight_client.do_get(endpoint.ticket)
            mb_counter = 0

            while True:
                mb_counter += 1
                data, row_size = self._read_chunk(thread_number=thread_number, reader=reader, mb_counter=mb_counter)

                if row_size != 0:  # append batches only when we have data
                    if not self.stop_reading:
                        self._chunk_queue_check(data=data, thread_number=thread_number)

                        if self.enable_subsampling:
                            # put all data into further subsampling
                            logger.debug(f"GD {thread_number}: putting mini batch to the queue...")
                            self._q_put_nowait((thread_number, data))
                            logger.debug(f"GD {thread_number}: mini batch already put.")

                        else:
                            with self.lock_read:
                                self.total_size = self.total_size + row_size * len(data)

                                # note: what to do when we have total size nearly under the limit
                                if self.total_size <= self.data_batch_size_limit:
                                    upper_row_limit = (self.data_batch_size_limit - self.total_size) // row_size
                                    data = data.iloc[:upper_row_limit]
                                    self._q_put_nowait((thread_number, data))
                                # --- end note
                                else:
                                    self._q_put_nowait((thread_number, 0))  # finish this thread
                                    self._set_stop_reading(True)

                    else:
                        break

            logger.debug(f"GD {thread_number}: Finishing thread work...")
            self._q_put_nowait((thread_number, 0))  # finish this thread

        except StopIteration:
            # Reading of batch finished, result commited."
            logger.debug(f"GD {thread_number}: StopIteration occurred.")
            self._q_put_nowait((thread_number, 0))

        except Exception as e:
            logger.debug(f"GD {thread_number}: Some error occurred. Error: {e}")
            self._q_put_nowait((thread_number, 0))
            self.threads_exceptions.append(str(e))

    @staticmethod
    def _cast_columns_to_float64_and_bool(data: 'pd.DataFrame') -> 'pd.DataFrame':
        def check_bool_value(value: str):
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            else:
                return value

        """Flight Service will cast decfloat types to strings, we need to cast them to correct types."""
        for i, (col_name, col_type) in enumerate(zip(data.dtypes.index, data.dtypes)):
            if col_type == 'object':
                try:
                    data[col_name] = data[col_name].astype('float64')
                except ValueError:  # ignore when column cannot be cast to other type than string
                    # logger.debug(f"Column '{col_name}' cannot be cast to float64 as it has normal strings inside")
                    try:
                        data[col_name] = data[col_name].apply(check_bool_value)
                    except ValueError:  # ignore when column cannot be cast to bool
                        logger.debug(f"Column '{col_name}' cannot be cast to bool as it has normal strings inside")
                    except Exception as e:
                        logger.debug(f"Casting data column '{col_name}' error: {e}")
                except Exception as e:
                    logger.debug(f"Casting data column '{col_name}' error: {e}")

        return data

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data columns and log data size"""
        logger.debug(f"Process data: Casting string data to numerical columns")
        data = self._cast_columns_to_float64_and_bool(data)

        logger.debug(f"BATCH SIZE: {sys.getsizeof(data)} Bytes")
        return data

    def _read_chunk(self, thread_number: int, reader: 'flight.FlightStreamReader', mb_counter: int):
        """Provides unified reading method for flight chunks."""
        logger.debug(f"RC {thread_number}: Waiting for next mini batch from Flight Service...")
        try:
            # Flight Service could split one batch into several chunks to have better performance
            mini_batch, metadata = reader.read_chunk()
            logger.debug(f"RC {thread_number}: Mini batch received from Flight Service.")
            data = pa.Table.from_batches(batches=[mini_batch]).to_pandas()
        except StopIteration as stop_iteration_error:
            if self.row_size == 0:
                self.empty_data_threads.add(thread_number)
            raise stop_iteration_error

        if len(data) == 1:
            row_size = sys.getsizeof(data.iloc[0].values)

        elif len(data) > 1:
            # Cast data to numerical types to reduce row size if possible. Needed especially when infer_schema is false
            processed_data = self._cast_columns_to_float64_and_bool(data)
            row_size = sys.getsizeof(processed_data) // len(processed_data)
        else:  # len(data) == 0
            row_size = 0

        if self.row_size == 0:
            self.row_size = row_size

        mini_batch_size = sys.getsizeof(data)
        logger.debug(f"RC {thread_number}: downloading mini_batch: {mb_counter} / shape: {data.shape} "
                     f"/ size: {mini_batch_size}")

        with self.lock_read:
            self.downloaded_data_size += mini_batch_size
            self.downloaded_data_nrows += len(data)

        if mb_counter % 10 == 0:
            logger.debug(f"RC {thread_number}: Total downloaded data size: "
                         f"{self.downloaded_data_size / 1024 / 1024}MB, rows: {self.downloaded_data_nrows}")

        return data, row_size

    def _chunk_queue_check(self, data: 'pd.DataFrame', thread_number: int):
        # note: check if queue is not too large, if it is, wait 3 sec and check again
        while True:
            q_valid_size = self.q.qsize()-len(self.empty_data_threads)
            if self.number_of_batch_rows is not None:
                data_rows = len(data) * q_valid_size
                if data_rows > self.number_of_batch_rows and not self.stop_reading:
                    logger.debug(f"QC {thread_number}: Waiting 3 sec as data queue is too large.")
                    logger.debug(
                        f"QC {thread_number}: Data queue size: {data_rows} rows, max rows per batch: "
                        f"{self.number_of_batch_rows}")
                    with self.read_status_change:
                        self.read_status_change.wait(timeout=3)
                    continue
                else:
                    break
            else:
                data_size = sys.getsizeof(data) * q_valid_size
                if data_size >= self.data_batch_size_limit and not self.stop_reading:
                    logger.debug(f"QC {thread_number}: Waiting 3 sec as data queue is too large.")
                    logger.debug(
                        f"QC {thread_number}: Data queue size: {data_size / 1024 / 1024}MB, available memory: "
                        f"{self.data_batch_size_limit / 1024 / 1024}MB")
                    with self.read_status_change:
                        self.read_status_change.wait(timeout=3)
                    continue
                else:
                    break
        # --- end note

    def _get_all_data_in_batches(self,
                                 thread_number: int,
                                 endpoint: 'flight.FlightEndpoint') -> None:
        """Read data from Flight Service.

        :param thread_number: specific number of the downloading thread
        :type thread_number: int

        :param endpoint: flight endpoint
        :type endpoint: flight.FlightEndpoint

        :return: batch data
        :rtype: pandas.DataFrame
        """
        try:
            reader = self.flight_client.do_get(endpoint.ticket)
            mb_counter = 0

            while True:
                mb_counter += 1
                data, row_size = self._read_chunk(thread_number=thread_number, reader=reader, mb_counter=mb_counter)

                if row_size != 0 and not self.stop_reading:  # append batches only when we have data
                    self._chunk_queue_check(data=data, thread_number=thread_number)
                    logger.debug(f"RT {thread_number}: putting mini batch to the queue...")
                    self._q_put_nowait((thread_number, data))
                    logger.debug(f"RT {thread_number}: mini batch already put.")

                else:
                    break

            logger.debug(f"RT {thread_number}: Finishing thread work...")
            self._q_put_nowait((thread_number, 0))  # finish this thread

        except StopIteration:
            # Reading of batch finished, result committed."
            self._q_put_nowait((thread_number, 0))

        except Exception as e:
            logger.debug(f"RT {thread_number}: Some error occurred. Error: {e}")
            self._q_put_nowait((thread_number, 0))
            self.threads_exceptions.append(str(e))

    def iterable_read(self) -> Generator:
        """Iterate over batches of data from Flight Service.

        How does it work?

        0. Read, create and yield a subsampled batch.
        1. Start separate threads per Flight partition to read mini batches.
        2. Eg. we have 5 separate threads that read the data (batch by batch and updates the queue).
        3. Start creating the logical batch by create_logical_batch() method. It will consume mini batches
            from the queue and try to create a logical bigger batch.
        4. We have defined a 'batch_queue' list variable that will be storing maximum of 2 logical batches at once.
            For each generated logical batch we perform the following actions:
            a) If 'batch_queue' is empty, append it with first logical batch
            b) Continue logical batch creation...
            c) If 'batch_queue' has one element (batch), append a new batch to it,
                indicates that all Flight data reading threads need to stop downloading the data,
            d) Yield batch from the beginning of the list 'batch_queue'
            e) If we have control back, delete the first batch
            f) Unblock all Flight reading threads
            g) Yield second batch from 'batch_queue'
        5. When we do not have a control flow right now and something other is processing our batch,
            all Flight Threads are working and downloading next data... (we need to ensure that overall queue
            will not overwhelm RAM)
        6. When we get a control flow back, there will be a really fast logical bach creation as
            all of the mini batches needed are already stored in the memory.
        """
        self.callback.status_message("Starting reading training data in batches...")
        if self.enable_subsampling:
            for data in self.read():
                self.yielded_nrows += len(data)
                if self.return_subsampling_stats:
                    data_stats = {"data_size": self.downloaded_data_size,
                                  "data_nrows": self.downloaded_data_nrows,
                                  "data_batch_size": sys.getsizeof(data),
                                  "data_batch_nrows": len(data),
                                  "no_batches": ceil(self.downloaded_data_nrows / len(data)) if len(data) > 0 else 0}
                    yield data, data_stats
                else:
                    yield data
                # rerurn stats here

            self.enable_subsampling = False

        sequences = []
        for n, endpoints in enumerate(self.get_endpoints()):
            self.max_flight_batch_number = len(endpoints)  # when flight does not want to open more endpoints
            threads = []
            sequences.append(threads)

            for i, endpoint in enumerate(endpoints):
                reading_thread = threading.Thread(target=self._get_all_data_in_batches, args=(i, endpoint))
                threads.append(reading_thread)
                logger.debug(f"IR: Starting batch reading thread: {i}, sequence: {n}...")
                reading_thread.start()

        for n, batch in enumerate(self.create_logical_batch(timeout=60 * 60, _type='data')):
            logger.debug(f"IR: Logical batch number {n} received.")
            logger.debug(f"IR: Passing batch to upper layer.")
            self._check_for_breaking_exceptions()
            self.yielded_nrows += len(batch)
            if (n == 0 and self.stop_after_first_batch) or (
                    self.total_nrows_limit and self.total_nrows_limit - self.yielded_nrows <= 0):
                self._set_stop_reading(True)

            if len(batch) == 0:
                break

            if self.return_subsampling_stats:
                yield batch, {"data_batch_size": sys.getsizeof(batch),
                              "data_batch_nrows": len(batch)}
            else:
                yield batch

            logger.debug(f"IR: We have control back, creating next batch...")
            if (n == 0 and self.stop_after_first_batch) or (
                    self.total_nrows_limit and self.total_nrows_limit - self.yielded_nrows <= 0):
                break

    def read(self) -> 'pd.DataFrame':
        """Fetch the data from Flight Service. Fetching is done in batches.
        There is an upper top limit of data size to be fetched configured to 1 GB.

        :return: fetched data
        :rtype: pandas.DataFrame
        """
        self.callback.status_message("Starting reading training data...")
        sequences = []

        def start_reading_threads():
            # Note: endpoints are created by Flight Service based on number of partitions configured
            # one endpoint serves multiple mini batches of the data
            try:
                if self.enable_subsampling:
                    subsampling_thread = threading.Thread(target=self._get_sampling_strategy(), args=(self.label,))
                    subsampling_thread.start()

                else:
                    normal_read_thread = threading.Thread(target=self.normal_read)
                    normal_read_thread.start()

                for n, endpoints in enumerate(self.get_endpoints()):
                    self.max_flight_batch_number = len(endpoints)  # when flight does not want to open more endpoints
                    threads = []
                    sequences.append(threads)

                    for i, endpoint in enumerate(endpoints):
                        reading_thread = threading.Thread(target=self._get_data, args=(i, endpoint))
                        threads.append(reading_thread)
                        logger.debug(f"R: Starting batch reading thread: {i}, sequence: {n}...")

                        reading_thread.start()

            except Exception as e:
                self.row_size = -1  # further we raise an error to finish thread
                self._set_stop_reading(True)
                raise e
            finally:

                for n, sequence in enumerate(sequences):
                    for i, thread in enumerate(sequence):
                        logger.debug(f"R: Joining batch reading thread {i}, sequence: {n}...")

                        thread.join()

                self._q_put_nowait((-1, -1))  # send close queue message (stop reading logical batches)
                try:
                    if self.enable_subsampling:
                        subsampling_thread.join()

                    else:
                        normal_read_thread.join()
                except:
                    pass

        try:
            start_reading_threads()

            self._check_for_breaking_exceptions()

            if self.enable_subsampling:
                if self.row_size > 0:
                    self.subsampled_data = self._sample_data_to_percentage(self.total_percentage_limit,
                                                                           self.subsampled_data,
                                                                           self.downloaded_data_nrows)
                self.subsampled_data = self._process_data(self.subsampled_data)
                self.yielded_nrows += len(self.subsampled_data)
                yield self.subsampled_data

            else:
                self.data = self._process_data(self.data)
                self.yielded_nrows += len(self.data)
                yield self.data

        except ValueError as e:
            if not self.fallback_to_one_connection:
                raise e

            timeout = 60 * 10
            self.callback.status_message(f"Parallel data connection problem, "
                                         f"fallback to 1 connection with {timeout} timeout.")
            logger.debug(f"Data merging error: {e}")
            logger.debug(f"Fallback to 1 connection. Timeout: {timeout}s")
            time.sleep(timeout)

            self.max_flight_batch_number = 1
            sequences = []
            self._q_reset()
            start_reading_threads()

            # Log every partition thread error (should be the flight service info included)
            if self.threads_exceptions:
                logger.debug(f"Partitions Thread Errors: {self.threads_exceptions}")

            try:
                if self.enable_subsampling:
                    self.subsampled_data = self._process_data(self.subsampled_data)
                    self.yielded_nrows += len(self.subsampled_data)
                    yield self.subsampled_data

                else:
                    self.data = self._process_data(self.data)
                    self.yielded_nrows += len(self.data)
                    yield self.data

            except ValueError:
                raise DataStreamError(reason=str(self.threads_exceptions))

    def _get_max_rows(self, _type: str) -> int:
        if self.number_of_batch_rows is not None:
            return self.number_of_batch_rows

        while True:
            if self.row_size == -1:
                raise TypeError("Data could not be read. Try to use a binary read mode.")

            if self.row_size != 0:
                if _type == 'data':
                    max_rows = self.data_batch_size_limit // self.row_size

                    if self.total_size_limit is not None:
                        # Note: convert total_size_limit to total_nrows_limit, later only total_nrows_limit will be used.
                        max_rows_total = self.total_size_limit // self.row_size
                        if self.total_nrows_limit:
                            if max_rows_total < self.total_nrows_limit:
                                self.total_nrows_limit = max_rows_total
                        else:
                            self.total_nrows_limit = max_rows_total

                    if self.total_nrows_limit and self.total_nrows_limit < max_rows:
                        return self.total_nrows_limit
                    else:
                        return max_rows

                else:
                    return self.logical_batch_size_limit // self.row_size

            else:
                self._check_for_breaking_exceptions()
                if self.threads_exceptions:
                    logger.debug(f"R: Partitions Thread Errors: {self.threads_exceptions}")
                    raise DataStreamError(reason=str(self.threads_exceptions))
                time.sleep(1)

    def regression_random_sampling(self, label_column: str) -> None:
        """Start collecting sampled data (random sample for regression problem)"""
        logger.debug(f"Starting regression random sampling.")
        max_rows = self._get_max_rows(_type='data')
        downloaded_data_size = 0
        data_batch_size = []

        for data in self.create_logical_batch():
            data = self.data_dropna(data, label_column)
            data_size = sys.getsizeof(data)
            downloaded_data_size += data_size
            data_batch_size.append(data_size)
            self.callback.status_message(f"Downloaded data size: {downloaded_data_size // 1024 // 1024} MB")
            logger.debug(f"Logical batch size: {data_size}")

            # join previous sampled batch with new one
            if self.subsampled_data is not None:
                data = pd.concat([self.subsampled_data, data])

            if len(data) <= max_rows:
                self.subsampled_data = data

            else:
                self.subsampled_data = data.sample(n=max_rows, random_state=0, axis=0)

            self.data_batch_size = data_batch_size[0]
            logger.debug(f"Subsampled batch size: {sys.getsizeof(self.subsampled_data)}")

    def stratified_subsampling(self, label_column: str) -> None:
        """Start collecting sampled data (stratified sample for classification problem)"""
        logger.debug(f"Starting classification stratified sampling.")
        max_rows = self._get_max_rows(_type='data')
        downloaded_data_size = 0
        data_batch_size = []

        for data in self.create_logical_batch():
            data = self.data_dropna(data, label_column)

            data_size = sys.getsizeof(data)
            downloaded_data_size += data_size
            data_batch_size.append(data_size)
            self.callback.status_message(f"Downloaded data size: {downloaded_data_size // 1024 // 1024} MB")
            logger.debug(f"Logical batch size: {data_size}")

            stats = data[label_column].value_counts()
            indexes = stats[stats == 1].index.values
            for i in indexes:
                logger.debug(f"Unique value in label column: {i}")
                data = data[data[label_column] != i]

            # join previous sampled batch with new one
            if self.subsampled_data is not None:
                data = pd.concat([self.subsampled_data, data])

            if len(data) <= max_rows:
                self.subsampled_data = data

            else:
                self.subsampled_data = data.groupby(label_column, group_keys=False).apply(lambda x: x.sample(frac=max_rows / len(data)))

            logger.debug(f"Subsampled batch size: {sys.getsizeof(self.subsampled_data)}")
        self.data_batch_size = data_batch_size[0]

    def truncate_sampling(self, label_column: str) -> None:
        """Start collecting sampled data (truncate sample for forecasting problem)"""
        logger.debug(f"Starting forecasting truncate sampling.")
        max_rows = self._get_max_rows(_type='data')
        downloaded_data_size = 0
        data_batch_size = []

        for data in self.create_logical_batch():
            data_size = sys.getsizeof(data)
            downloaded_data_size += data_size
            data_batch_size.append(data_size)
            self.callback.status_message(f"Downloaded data size: {downloaded_data_size // 1024 // 1024} MB")
            logger.debug(f"Logical batch size: {data_size}")

            # join previous sampled batch with new one
            if self.subsampled_data is not None:
                data = pd.concat([self.subsampled_data, data])

            if len(data) <= max_rows:
                self.subsampled_data = data

            else:
                timestamp_column = self.params.get('timestamp_column') or self.params.get('timestamp_column_name')
                if timestamp_column is not None:
                    data = data.sort_values(timestamp_column)
                self.subsampled_data = data.tail(max_rows)

            logger.debug(f"Subsampled batch size: {sys.getsizeof(self.subsampled_data)}")
        self.data_batch_size = data_batch_size[0]

    def normal_read(self):
        """Start collecting all the data when user do not want to subsample.
        This should be limited to the max data size 1GB, see limitation implementation in self._read_data().
        """
        for data in self.create_logical_batch():
            self.data = data
            break

    def create_logical_batch(self, timeout: int = 60 * 10, _type: str = 'logical') -> 'pd.DataFrame':
        """Create a logical batch for sampling, logical batch is larger ~2GB.
        If used with normal read, it will return all of the collected data, max 1GB based on a limitation.
        """
        max_rows = self._get_max_rows(_type=_type)
        logical_batch = []
        mini_batch_counter = 0
        threads_finished = 0
        data = None

        while True:
            try:
                if threads_finished == self.max_flight_batch_number:
                    if logical_batch:
                        data = pd.concat(logical_batch)  # flush last data
                        yield self._process_data(data)
                    break

                logger.debug("LB: Waiting for mini batch to appear in the queue...")
                thread_number, data = self._q_get(timeout=timeout)  # wait max 10 min

                self.q.task_done()
                logger.debug(f"LB: Mini batch received and taken from the queue. Batch from thread: {thread_number}")

                # when the last thread finish reading (finish sequence (-1, -1))
                if isinstance(data, int) and thread_number == -1 and data == -1:
                    logger.debug(
                        f"LB: Received the final batch. Batch from thread: {thread_number}")
                    self._set_stop_reading(True)
                    if logical_batch:
                        data = pd.concat(logical_batch)  # flush last data
                        yield self._process_data(data)
                    break

                # when not the last thread finished reading, continue
                elif isinstance(data, int):
                    threads_finished += 1
                    continue

                mini_batch_counter += 1

                logger.debug(f"LB: Mini batch number: {mini_batch_counter}")

            except queue.Empty:
                if data is None or isinstance(data,
                                              int) or data.empty:  # raise the error only if any data were not downloaded.
                    threads_finished += 1
                    raise DataStreamError(
                        reason=f'LB: No data were downloaded in the thread due to timeout on '
                               f'waiting for data batch: max {timeout / 60} minutes, data: {data}')

            logical_batch.append(data.iloc[:max_rows])
            not_used_data = data.iloc[max_rows:]
            max_rows = max_rows - len(logical_batch[-1])

            if max_rows > 0:
                continue

            else:
                logger.debug("LB: Yielding logical batch!")
                data = pd.concat(logical_batch)  # flush data
                yield self._process_data(data)

                # note: check if the not_used_data contains enough data to produce next logical batch:
                while len(not_used_data) > self._get_max_rows(_type=_type):
                    max_rows = self._get_max_rows(_type=_type)
                    logger.debug("LB: Yielding logical batch!")
                    data = not_used_data.iloc[:max_rows] # flush data
                    not_used_data = not_used_data.iloc[max_rows:]
                    yield self._process_data(data)
                # end note
                logical_batch = [not_used_data]

                if self.total_nrows_limit and self.yielded_nrows:
                    max_rows = min(self._get_max_rows(_type=_type) - len(logical_batch[-1]),
                                   self.total_nrows_limit - self.yielded_nrows)
                    if max_rows < 0:  # finish if limit has been reached
                        self._set_stop_reading(True)
                        break
                else:
                    max_rows = self._get_max_rows(_type=_type) - len(logical_batch[-1])


    def _select_source_command(self, infer_schema: bool = False) -> List[str]:
        """Based on a data source type, select appropriate commands for flight service configuration."""

        infer_schema = infer_schema
        if self.read_binary:
            infer_schema = False

        if self.write_binary:
            command = {"interaction_properties": {'write_mode': "write_raw"}}

        else:
            # limitation for number of rows passed within one mini batch
            # from flight service (can be between 100 and 10000)
            # need to test performance with big data
            default_batch_size = DEFAULT_BATCH_SIZE_FLIGHT_COMMAND
            # when reading binary batches, default row is one value of 32k.
            # If we keep a large default batch size like 10k, we end up with
            # batches of 300MB being read and kept in memory as 10k chunks
            # of 32k. This clutters the memory and is more work for GC.
            # Defaulting to 1000 when reading binary to decrease footprint.
            if self.read_binary:
                default_batch_size = DEFAULT_BATCH_SIZE_FLIGHT_COMMAND_BINARY_READ
            # when dataset is small we received empty dfs, it is ok (4 is optimum for larger than 1 GB)
            command = {
                "num_partitions": self.max_flight_batch_number if self.max_flight_batch_number is not None else DEFAULT_PARTITIONS_NUM,
                "batch_size": default_batch_size,
                "interaction_properties": {}
            }

        if self.extra_interaction_properties is not None:
            command["interaction_properties"].update(self.extra_interaction_properties)

        # note: if number_of_batch_rows is bigger that 10 000 then set the 10 000 as batch_size in command. #
        # to specify bigger batch_size use flight_parameters parameter - Flight limitation is 100 000(but sometimes it's to much).
        if self.number_of_batch_rows is not None and self.number_of_batch_rows < default_batch_size:
            command['batch_size'] = self.number_of_batch_rows

        command.update(self.flight_parameters)

        if self.space_id is not None:
            command['space_id'] = self.space_id

        elif self.project_id is not None:
            command['project_id'] = self.project_id

        if self.read_binary:
            command['interaction_properties'].update({'read_mode': "read_raw"})

        if self.asset_id:
            command['asset_id'] = self.asset_id
            ## TODO: Remove commented code below - issue https://github.ibm.com/NGP-TWC/ml-planning/issues/28663
            # if infer_schema:
            #     command['interaction_properties'].update({'infer_schema': "true"})

        elif self.connection_id:
            command['asset_id'] = self.connection_id

        if 'bucket' in self.data_location['location']:
            if self.data_location.get('location', {}).get('path', None) is None \
                    and 'file_name' in self.data_location.get('location', ''):
                self.data_location['location']['path'] = self.data_location['location']['file_name']
            command['interaction_properties'].update(prepare_interaction_props_for_cos(
                self.params, self.data_location['location']['path']))

            if 'file_format' in command['interaction_properties'] and ('write_mode' in command[
                'interaction_properties'] or self.read_binary):
                del command['interaction_properties']['file_format']

            if 'sheet_name' in command['interaction_properties'] and ('write_mode' in command[
                'interaction_properties'] or self.read_binary):
                del command['interaction_properties']['sheet_name']

            if 'encoding' in command['interaction_properties'] and ('write_mode' in command[
                'interaction_properties'] or self.read_binary):
                del command['interaction_properties']['encoding']

            if 'quote_character' in command['interaction_properties'] and ('write_mode' in command[
                'interaction_properties'] or self.read_binary):
                del command['interaction_properties']['quote_character']

            if infer_schema:
                command['interaction_properties']['infer_schema'] = "true"
            command['interaction_properties']['file_name'] = self.data_location['location']['path']
            command['interaction_properties']['bucket'] = self.data_location['location']['bucket']

        elif 'schema_name' in self.data_location['location']:
            command['interaction_properties']['schema_name'] = self.data_location['location']['schema_name']
            command['interaction_properties']['table_name'] = self.data_location['location']['table_name']

        elif not self.asset_id:
            command['interaction_properties'].update(self.data_location['location'])
            if infer_schema:
                command['interaction_properties']['infer_schema'] = "true"

        # Property 'infer_as_varchar` needs to be false, when 'infer_schema' is true
        if command['interaction_properties'].get('infer_schema',
                                                 "false") == "true" and self.infer_as_varchar is not None:
            command['interaction_properties'].update({'infer_as_varchar': self.infer_as_varchar})

        # Git based project assets property
        if self.data_location is not None and str(self.data_location['location'].get('userfs', 'false')).lower() == 'true':
            command.update({'userfs': True})

        if 'path' in command['interaction_properties']:
            command['interaction_properties']['file_name'] = command['interaction_properties']['path']
            del command['interaction_properties']['path']

        if 'connection_properties' not in command:
            logger.debug(f"Command: {command}")

        return [json.dumps(command)]

    def read_binary_data(self, read_to_file=None) -> None:
        """Try to read data from flight service using the 'read_raw' parameter. This will allow to fetch binary data.
        Binary read should be used for small data, like json files, zip files etc. not for the big datasets as
        each data batch is joined to the previous one in-memory.
        """
        self.read_binary = True

        if self.flight_parameters.get('num_partitions') is None:
            self.flight_parameters['num_partitions'] = 1
            self.max_flight_batch_number = 1

        cm = open(read_to_file, 'wb') if read_to_file else nullcontext()
        with cm as sink:
            binary_data_array = []
            for n, endpoints in enumerate(self.get_endpoints()):
                for i, endpoint in enumerate(endpoints):
                    reader = self.flight_client.do_get(endpoint.ticket)
                    try:
                        while True:
                            mini_batch, metadata = reader.read_chunk()
                            if read_to_file:
                                sink.write(b''.join(mini_batch.columns[0].tolist()))
                            else:
                                binary_data_array.extend(mini_batch.columns[0].tolist())
                    except StopIteration:
                        pass
        if read_to_file:
            return [read_to_file]
        else:
            binary_data_container = b''.join(binary_data_array)
            yield binary_data_container

    def write_binary_data(self, file_path: str) -> None:
        """Write data in 16MB binary data blocks. 16MB upper limit is set by the Flight Service.
        The writer will open the source local file and will stream one batch of 16MB to the Flight.
        Only 16MB of data is loaded into the memory at a time.

        :param file_path: path to the source file
        :type file_path: str
        """
        self.write_binary = True
        schema = pa.schema([
            ('content', pa.binary())
        ])
        commands = self._select_source_command(infer_schema=False)

        self.flight_client.authenticate(self.authenticate())
        writer, reader = self.flight_client.do_put(flight.FlightDescriptor.for_command(commands[0]), schema)

        with writer:
            batch_max_size = 16770000  # almost 16MB

            with open(file_path, 'rb') as file:
                for block in iter(partial(file.read, batch_max_size), b''):
                    writer.write_batch(pa.record_batch([pa.array([block], type=pa.binary())], schema=schema))
                    self.flight_client.wait_for_available()

    def write_data(self, data: 'pd.DataFrame'):
        """Write data from pandas DataFrame. The limit is 16MB dataframe as this is the upper batch size limit.
        Upper layer should fallback to use binary write.
        """
        schema = pa.Schema.from_pandas(data)
        commands = self._select_source_command(infer_schema=False)

        self.flight_client.authenticate(self.authenticate())
        writer, reader = self.flight_client.do_put(flight.FlightDescriptor.for_command(commands[0]), schema)

        with writer:
            writer.write_table(pa.Table.from_pandas(data))
            self.flight_client.wait_for_available()

        return writer, reader

    def get_batch_writer(self, schema: 'pa.Schema') -> 'FlightStreamWriter':
        """Prepare FlightStreamWriter and return it."""
        commands = self._select_source_command(infer_schema=False)

        self.flight_client.authenticate(self.authenticate())
        writer, reader = self.flight_client.do_put(flight.FlightDescriptor.for_command(commands[0]), schema)
        return writer

    def _check_for_breaking_exceptions(self):
        # Log every partition thread error (should be the flight service info included)
        if self.threads_exceptions:
            logger.debug(f"IR/R: Partitions Thread Errors: {self.threads_exceptions}")

            for msg in self.threads_exceptions:
                if "Data could not be read" in msg:
                    raise TypeError(msg)
                if 'Internal error occurred' in msg:
                    raise TypeError(msg)
                if "Unknown error: Wrapping" in msg:
                    raise CorruptedData(msg)

        # Note: check if any data were downloaded:
        if len(self.empty_data_threads) == self.max_flight_batch_number:
            raise EmptyDataSource()

    def _get_sampling_strategy(self):
        """Return sampling strategy for given sampling and learning type."""
        random_sampling_pred_types = (
            PredictionType.REGRESSION, PredictionType.CLASSIFICATION, PredictionType.BINARY, PredictionType.MULTICLASS
        )
        stratified_sampling_pred_types = (
            PredictionType.CLASSIFICATION, PredictionType.BINARY, PredictionType.MULTICLASS
        )
        truncate_sampling_pred_types = (
            PredictionType.FORECASTING, PredictionType.TIMESERIES_ANOMALY_PREDICTION
        )

        if self.sampling_type == SamplingTypes.RANDOM and self.learning_type in random_sampling_pred_types:
            return self.regression_random_sampling
        elif self.sampling_type == SamplingTypes.STRATIFIED and self.learning_type in stratified_sampling_pred_types:
            return self.stratified_subsampling
        elif self.sampling_type == SamplingTypes.LAST_VALUES and self.learning_type in truncate_sampling_pred_types:
            return self.truncate_sampling
        else:
            raise InvalidSamplingType(self.sampling_type, self.learning_type)

    def _sample_data_to_percentage(self, percentage, data, full_data_nrows):
        if self.enable_subsampling:
            if percentage < 1.0 and percentage > 0.0:
                nrows = int(percentage * full_data_nrows)
                logger.debug(f"Running sampling to prercentage of data: "
                             f"{percentage} from {full_data_nrows} rows gives {nrows} of data rows")
                if len(data) < nrows:
                    return data  # no sampling needed

                if self.sampling_type == SamplingTypes.RANDOM:
                    data = data.sample(n=nrows, random_state=0, axis=0)
                elif self.sampling_type == SamplingTypes.STRATIFIED:
                    data = data.groupby(self.label, group_keys=False).apply(lambda x: x.sample(frac=nrows/len(data)))

                elif self.sampling_type == SamplingTypes.LAST_VALUES:
                    timestamp_column = self.params.get('timestamp_column') or self.params.get('timestamp_column_name')
                    if timestamp_column is not None:
                        data = data.sort_values(timestamp_column)
                    data = data.tail(nrows)
                else:
                    raise NotImplementedError(
                        f"Sampling type: {self.sampling_type} does not supports additional sampling to percentage of data.")
                return data
            elif percentage == 1.0:
                return data
            else:
                raise ValueError('The `percentage` need to be float between 0.0 and 1.0.')
        else:
            return data


    def data_dropna(self, data, label_column):
        # simple preprocess before sampling
        data_with_nans_len = len(data)
        data.dropna(inplace=True, subset=[label_column])
        removed_rows = data_with_nans_len - len(data)
        if removed_rows > 0:
            self.downloaded_data_nrows = self.downloaded_data_nrows - removed_rows  # fix for no_batches calculation #31307

        return data
