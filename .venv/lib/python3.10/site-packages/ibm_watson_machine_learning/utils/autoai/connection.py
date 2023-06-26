#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import TYPE_CHECKING, List, Union

from ibm_watson_machine_learning.helpers.connections import (
    DataConnection, S3Location, FSLocation, AssetLocation, CP4DAssetLocation, WMLSAssetLocation, CloudAssetLocation,
    WSDAssetLocation, DeploymentOutputAssetLocation, ContainerLocation, S3Connection)
from ibm_watson_machine_learning.utils.autoai.enums import (DataConnectionTypes)
from ibm_watson_machine_learning.utils.autoai.errors import ContainerTypeNotSupported


from os import environ
from re import findall


if TYPE_CHECKING:
    from ibm_watson_machine_learning.workspace import WorkSpace


__all__ = [
    "validate_source_data_connections",
    "create_results_data_connection",
    "validate_results_data_connection",
    "create_deployment_output_data_connection",
    "validate_deployment_output_connection",
    "get_max_sample_size_limit"
]


def validate_source_data_connections(source_data_connections: List['DataConnection'],
                                     workspace: 'WorkSpace',
                                     deployment=False) -> List['DataConnection']:
    for data_connection in source_data_connections:

        if isinstance(data_connection.location, FSLocation):
            # note: save data as an data asset
            if workspace.wml_client.ICP:
                asset_id = data_connection.location._save_file_as_data_asset(workspace=workspace)
                data_connection.location = AssetLocation(asset_id)
                data_connection.type = DataConnectionTypes.DS
            # --- end note

        elif isinstance(data_connection.location, WSDAssetLocation):
            # note: try to move local asset to server
            data_connection.location._move_asset_to_server(workspace=workspace)
            # --- end note
        elif isinstance(data_connection.location, AssetLocation) and data_connection.location._wsd:
            # note: try to move local asset to server
            data_connection.location._wsd_move_asset_to_server(workspace=workspace)
            # --- end note
        elif isinstance(data_connection.location, ContainerLocation):
            if workspace.wml_client.ICP:
                raise ContainerTypeNotSupported() # block Container type on CPD
            elif isinstance(data_connection.connection, S3Connection):
                # note: remove S3 inline credential from data asset before training
                data_connection.connection = None
                if hasattr(data_connection.location, 'bucket'):
                    delattr(data_connection.location, 'bucket')
                # --- end note

        elif isinstance(data_connection.connection, S3Connection) and isinstance(data_connection.location,
                                                                                 AssetLocation):
            # note: remove S3 inline credential from data asset before training
            data_connection.connection = None

            for s3_attr in ['bucket', 'path']:
                if hasattr(data_connection.location, s3_attr):
                    delattr(data_connection.location, s3_attr)
            # --- end note

    return source_data_connections


def create_results_data_connection(source_data_connections: List['DataConnection'],
                                   workspace: 'WorkSpace') -> 'DataConnection':
    if isinstance(source_data_connections[0].location, S3Location):
        results_data_connection = DataConnection(
            connection=source_data_connections[0].connection,
            location=S3Location(bucket=source_data_connections[0].location.bucket,
                                path='.')
        )
    else:
        location = FSLocation()
        if workspace.wml_client.default_space_id:
            location.path = location.path.format(option='spaces',
                                                 id=workspace.wml_client.default_space_id)
        else:
            location.path = location.path.format(option='projects',
                                                 id=workspace.wml_client.default_project_id)
        results_data_connection = DataConnection(
            connection=None,
            location=location
        )
    return results_data_connection


def validate_results_data_connection(results_data_connection: Union['DataConnection', None],
                                     workspace: 'WorkSpace',
                                     source_data_connections: List['DataConnection'] = None) -> 'DataConnection':
    # note: if user did not provide results storage information, use default ones
    if results_data_connection is None and source_data_connections is not None:
        results_data_connection = create_results_data_connection(source_data_connections=source_data_connections,
                                                                 workspace=workspace)
    # -- end note
    # note: results can be stored only on FS or COS
    if not isinstance(results_data_connection.location, (S3Location, FSLocation)):
        raise TypeError('Unsupported results location type. Results reference can be stored'
                        ' only on S3Location or FSLocation.')
    # -- end
    return results_data_connection


def create_deployment_output_data_connection(source_data_connections: List['DataConnection'],
                                             output_filename = 'deployment_output.csv') -> 'DataConnection':
    if isinstance(source_data_connections[0].location, S3Location):
        results_data_connection = DataConnection(
            connection=source_data_connections[0].connection,
            location=S3Location(bucket=source_data_connections[0].location.bucket,
                                path=output_filename)
        )
    else:
        location = DeploymentOutputAssetLocation(name=output_filename)
        results_data_connection = DataConnection(
            connection=None,
            location=location
        )
    return results_data_connection


def validate_deployment_output_connection(results_data_connection: Union['DataConnection', None],
                                          workspace: 'WorkSpace',
                                          source_data_connections: List['DataConnection'] = None) -> 'DataConnection':
    # note: if user did not provide results storage information, use default ones
    if results_data_connection is None and source_data_connections is not None:
        results_data_connection = create_results_data_connection(source_data_connections=source_data_connections,
                                                                 workspace=workspace)
    # -- end note

    if isinstance(results_data_connection.location, (AssetLocation, CP4DAssetLocation, WMLSAssetLocation, CloudAssetLocation)):
        if workspace.wml_client.default_space_id:
            results_data_connection.location.path = results_data_connection.location.path.format(option='spaces',
                                                                                                 id=workspace.wml_client.default_space_id)
        else:
            results_data_connection.location.path = results_data_connection.location.path.format(option='projects',
                                                                                                 id=workspace.wml_client.default_project_id)

    # note: results can be stored only on COS or CPD/WMLS DataAsset or DeploymentOutputAssetLocation
    if not isinstance(results_data_connection.location, (S3Location, AssetLocation, CP4DAssetLocation, WMLSAssetLocation, DeploymentOutputAssetLocation)):
        raise TypeError('Unsupported results location type. Results reference can be stored only in '
                        'one of [S3Location, AssetLocation, DeploymentOutputAssetLocation].')
    # -- end
    return results_data_connection


def get_max_sample_size_limit() -> int:
    """Get Sample Size Limit based on T-Shirt size if code is run on training pod:
    For memory < 16 (T-Shirts: XS,S) default is 10MB,
    For memory < 32 & >= 16 (T-Shirts: M) default is 100MB,
    For memory = 32 (T-Shirt L) default is 0.7GB,
    For memory > 32 (T-Shirt XL) or runs outside pod default is 1GB.
    Returns sample size limit in bytes.
    """
    from ibm_watson_machine_learning.data_loaders.datasets.experiment import DEFAULT_SAMPLE_SIZE_LIMIT, \
        DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT
    size_mem = int(findall(r'[0-9]+', environ.get('MEM', '32'))[0])
    if size_mem > 32:
        size_limit = DEFAULT_SAMPLE_SIZE_LIMIT             # 1GB in bytes
    elif size_mem == 32:
        size_limit = int(0.7 * DEFAULT_SAMPLE_SIZE_LIMIT)  # 0.7GB in bytes
    elif 32 > size_mem >= 16:
        size_limit = DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT     # 100MB in bytes
    else:
        size_limit = int(0.1 * DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT)  # 10MB in bytes

    return size_limit

