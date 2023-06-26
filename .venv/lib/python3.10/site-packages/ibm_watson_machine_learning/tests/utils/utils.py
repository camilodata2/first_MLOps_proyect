#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json
import os
import time
import datetime
from configparser import ConfigParser
from functools import wraps
from time import sleep
import ibm_boto3

from ibm_watson_machine_learning.tests.utils.cleanup import bucket_cleanup

__all__ = [
    "get_wml_credentials",
    "get_cos_credentials",
    "get_db_credentials",
    "get_env",
    "is_cp4d",
    "bucket_exists",
    "bucket_name_gen",
    "create_bucket",
    'print_test_separators',
    "get_space_id",
    "resources_discovery",
    "find_not_freed_resources",
    "setup_nfs_env",
    "save_data_to_container",
    "save_data_to_cos_bucket",
    "create_connection_to_cos"
]

if "ENV" in os.environ:
    environment = os.environ['ENV']
else:
    environment = "YP_QA"

timeouts = "TIMEOUTS"
credentials = "CREDENTIALS"
training_data = "TRAINING_DATA"
configDir = "./config.ini"

config = ConfigParser()
config.read(configDir)


def get_env():
    return environment


def get_wml_credentials(env=environment):
    wml_credentials = json.loads(config.get(env, 'wml_credentials'))
    if 'flight_service_location' in wml_credentials and not os.environ.get('FLIGHT_SERVICE_LOCATION'):
        os.environ['FLIGHT_SERVICE_LOCATION'] = wml_credentials['flight_service_location']
    if 'flight_service_port' in wml_credentials and not os.environ.get('FLIGHT_SERVICE_PORT'):
        os.environ['FLIGHT_SERVICE_PORT'] = str(wml_credentials['flight_service_port'])
    return wml_credentials


def get_cos_credentials(env=environment):
    return json.loads(config.get(env, 'cos_credentials'))


def get_db_credentials(db_name, env=environment):
    """
    Returns credentials for a given database.
    Credentials should be added to config.ini
    with the following template: `<database_name>_credentials: {...}`
    """
    return json.loads(config.get(env, f"{db_name}_credentials"))


def is_cp4d():
    if "CP4D" in get_env():
        return True
    elif "ICP" in get_env():
        return True
    elif "OPEN_SHIFT" in get_env():
        return True
    elif "CPD" in get_env():
        return True

    return False


def bucket_exists(cos_resource, bucket_name):
    """
    Return True if bucket with `bucket_name` exists. Else False.
    """
    buckets = cos_resource.buckets.all()
    for bucket in buckets:
        if bucket.name == bucket_name:
            return True
    print("Bucket {0} not found".format(bucket_name))
    return False


def bucket_name_gen(prefix='bucket-tests', id_size=8):
    import random
    import string

    return prefix + "-" + ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(id_size))


def create_bucket(cos_resource, bucket_name):
    """
    Create bucket of with a given name and timestamp suffix.
    """
    try:
        bucket_cleanup(cos_resource, prefix=f"{bucket_name}-")
    except Exception as e:
        print(f"Bucket cleanup with prefix {bucket_name}- failed due to:\n{e}\n skipped")

    bucket_name = bucket_name_gen(prefix=f"{bucket_name}-{str(datetime.date.today())}")
    print(f"Creating COS bucket: {bucket_name}")
    cos_resource.Bucket(bucket_name).create()

    return bucket_name


def print_test_separators(method):
    """Printing visual separators for tests."""

    @wraps(method)
    def _method(*method_args, **method_kwargs):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        method_output = method(*method_args, **method_kwargs)
        print("________________________________________________________________________")

        return method_output

    return _method


def get_space_id(wml_client, space_name, cos_resource_instance_id=None):
    """
    Return space id of existed or just created space named as `space_name`.
    """
    spaces_details = wml_client.spaces.get_details(limit=200).get('resources')
    space_id = None
    for space_d in spaces_details:
        if space_d['entity']['name'] == space_name:
            space_id = space_d['metadata']['id']

    if not space_id:
        # create new space for tests
        if wml_client.CLOUD_PLATFORM_SPACES:
            metadata = {
                wml_client.spaces.ConfigurationMetaNames.NAME: space_name,
                wml_client.spaces.ConfigurationMetaNames.STORAGE: {
                    'type': 'bmcos_object_storage',
                    "resource_crn": cos_resource_instance_id},
                wml_client.spaces.ConfigurationMetaNames.COMPUTE: {
                    "name": get_wml_credentials()['name'],
                    "crn": get_wml_credentials()['iam_serviceid_crn']}
            }
        else:
            metadata = {
                wml_client.spaces.ConfigurationMetaNames.NAME: space_name
            }
        details = wml_client.spaces.store(meta_props=metadata, background_mode=False)
        space_id = details['metadata'].get('id')
        print(f"New space `{space_name}` has been created, space_id={space_id}")
        sleep(5)  # wait for space preparing

    return space_id


def resources_discovery(wml_client):
    resources_map = {}

    def check(parent_el, parent_name):
        for r in parent_el.__dict__:
            try:
                if parent_el.__dict__[r].__class__.__module__.startswith('ibm_watson_machine_learning.'):
                    details = parent_el.__dict__[r].get_details()['resources']
                    if type(details) is list:
                        try:
                            uids = [x['metadata']['id'] for x in details]
                        except:
                            uids = [x['metadata']['guid'] for x in details]
                        print(f"{parent_name}.{r}: {len(uids)}")
                        resources_map[f"{parent_name}.{r}"] = uids

                        check(parent_el.__dict__[r], f"{parent_name}.{r}")
            except Exception as e:
                pass

    check(wml_client, "wml_client")

    return resources_map


def find_not_freed_resources(initial_list, final_list):
    not_freed_list = {}
    for k in initial_list:
        initial_uids = initial_list[k]
        final_uids = final_list[k]
        not_freed_uids = [uid for uid in final_uids if uid not in initial_uids]
        if len(not_freed_uids) > 0:
            not_freed_list[k] = not_freed_uids

    if len(not_freed_list) > 0:
        raise Exception(f"Resources not freed: {not_freed_list}")


def setup_nfs_env(wml_credentials, dataset_file_name, remote_directory='datasets'):
    import paramiko
    from scp import SCPClient

    ssh_host = wml_credentials.get('ssh_host')
    ssh_username = wml_credentials.get('ssh_username')
    ssh_password = wml_credentials.get('ssh_password')

    kube_username = wml_credentials.get('kube_username')
    kube_password = None

    if not (ssh_host and ssh_username and ssh_password):
        print("NFS setup omitted. Required credentials missing: ssh_host, ssh_username, ssh_password")
        return

    host = wml_credentials['url']
    username = wml_credentials['username']
    password = wml_credentials['password']

    storage_config_filepath = "autoai-sdk-test-storage-class-config.yaml"
    pvc_config_filepath = "autoai-sdk-test-pvc-config.yaml"
    storage_config_file_name = storage_config_filepath.split('/')[-1]
    pvc_config_file_name = pvc_config_filepath.split('/')[-1]
    storage_class_name = "managed-nfs-storage"
    pvc_name = "autoai-sdk-test-pvc"
    volume_name = "autoai-sdk-test-volume"

    project_id = wml_credentials['project_id']
    connection_id = None
    namespace = host.replace('https://', '').split('-')[0]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ssh_host, username=ssh_username, password=ssh_password)

    def run_ssh(cmd, format='plain'):
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd)
        out = ssh_stdout.read().decode('utf-8')
        err = ssh_stderr.read().decode('utf-8')

        if err and not ('200 OK' in err or '409 Conflict' in err or '201 Created' in err):
            raise Exception(err)

        if format == 'json':
            return json.loads(out.strip().split('\n')[-1])
        elif format == 'plain':
            return out
        else:
            raise Exception(f'Unsupported return value format: {format}')

    def put_file(filepath, directory='.'):
        scp = SCPClient(ssh.get_transport())
        scp.put(filepath, recursive=True, remote_path=f'~/{directory}')
        scp.close()

    kube_password = run_ssh('cat /root/auth/kubeadmin-password')

    res = run_ssh(f'oc login -u {kube_username} -p {kube_password} --kubeconfig /root/auth/kubeconfig')

    try:
        res = run_ssh(f'oc project {namespace} --kubeconfig /root/auth/kubeconfig')
        print(res)
    except:
        raise Exception(
            "Cannot change project to correct one for this cluster. Probably cluster changed. Add to credentials kube_username and make sure ssh credentials match the cluster.")

    # Check if there is already an existing NFS storage class
    res = run_ssh('oc get storageclass --kubeconfig /root/auth/kubeconfig')

    # Create a PVC volume
    if storage_class_name in res:
        print(f"Storage class exists: {storage_class_name}")
    else:
        res = run_ssh('ls')
        if storage_config_file_name not in res:
            put_file(storage_config_filepath)

        res = run_ssh(f'oc create -f ./{storage_config_file_name} --kubeconfig /root/auth/kubeconfig')

        if not ' created' in res:
            raise Exception(f'Failure during creating storage class: {res}')
        else:
            print(f"Storage class created: {res.replace(' created', '')}")

    res = run_ssh('oc get pvc --kubeconfig /root/auth/kubeconfig')

    if pvc_name in res:
        print(f"PVC exists: {pvc_name}")
    else:
        res = run_ssh('ls')
        if pvc_config_file_name not in res:
            put_file(pvc_config_filepath)

        res = run_ssh(f'oc create -f ./{pvc_config_file_name} --kubeconfig /root/auth/kubeconfig')

        if not ' created' in res:
            raise Exception(f'Failure during creating PVC: {res}')
        else:
            print(f"PVC created: {res.replace(' created', '')}")

    # Create a Volume on your existing PVC

    res = run_ssh(f"""
        curl -kiv -X POST \
            {host}/icp4d-api/v1/authorize \
            -H 'Content-Type: application/json' \
            -d '{{
                    "username":"{username}",
                    "password":"{password}"
            }}'
    """, format='json')

    token = res['token']

    print(f"Successfully got token")

    res = run_ssh(f"""
            curl -kiv -X GET \
                -H 'Authorization: Bearer {token}' \
                {host}/zen-data/v3/service_instances?addon_type=volumes
    """, format='json')

    instance = None
    for e in res['service_instances']:
        if e['display_name'] == volume_name and e['metadata']['zenControlPlaneNamespace'] == namespace \
                and e['metadata']['existing_pvc_name'] == pvc_name:
            instance = e
            break

    if instance:
        print(f"Service instance already exists: {instance['id']}")
    else:
        print("No service instance found, creating one.")

        res = run_ssh(f"""
            curl -kiv -X POST \
                -H 'Authorization: Bearer {token}' \
                -H 'Content-Type: application/json' \
                -d '{{
                    "addon_type": "volumes",
                    "addon_version": "-",
                    "create_arguments": {{
                        "metadata": {{
                            "existing_pvc_name": "{pvc_name}"
                        }}
                    }},
                    "namespace": "{namespace}",
                    "display_name":"{volume_name}"
                }}' \
                {host}/zen-data/v3/service_instances
        """, format='json')

        instance = res
        print(f"Service instance created: {instance['id']}")

    # Start the volume

    res = run_ssh(f"""
        curl -kiv --location --request POST '{host}/zen-data/v1/volumes/volume_services/{volume_name}' \
            --header 'Authorization: Bearer {token}' \
            --header 'Content-Type: application/json' \
            --data-raw '{{}}'
    """, format='json')

    started_earlier = False

    if '_statusCode_' in res and res['_statusCode_'] == 409:
        print(f'The volume `{volume_name}` is already started')
        started_earlier = True
    else:
        print(f'Successfully started volume: {volume_name}')

    for i in range(100):
        res = run_ssh(f"""
            curl -kiv --location --request GET '{host}/zen-data/v3/service_instances/{instance['id']}/?include_service_status=true' \
                --header 'Authorization: Bearer {token}'
        """, format='json')

        time.sleep(2)

        if res['services_status'] == 'RUNNING':
            print(f'Volume `{volume_name}` is started and running.')
            break

    if res['services_status'] == 'FAILED':
        raise Exception(
            f'Volume `{volume_name}` failed to run. Probably wrong cluster is set in login and as project. The project when `oc project` should be `{namespace}`')

    if res['services_status'] != 'RUNNING':
        raise Exception(f'Volume `{volume_name}` not running.')

    # Upload a file into the volume
    directory_exists = True

    try:
        res = run_ssh(f"""
            curl -kiv --location --request GET '{host}/zen-volumes/{volume_name}/v1/volumes/directories/{remote_directory}' \
                --header 'Authorization: Bearer {token}'
        """, format='json')
    except Exception as e:
        if '404 Not Found' in str(e):
            directory_exists = False
        else:
            raise e

    if directory_exists and dataset_file_name in res['responseObject']['directoryContents']:
        print('File is already uploaded.')
    else:
        res = run_ssh(f'ls')

        if remote_directory not in res.split():
            res = run_ssh(f'mkdir {remote_directory}')

        res = run_ssh(f'ls {remote_directory}')
        if not dataset_file_name in res:
            put_file(f"autoai/data/{dataset_file_name}", remote_directory)

        res = run_ssh(f"""
            curl -kiv --location --request PUT '{host}/zen-volumes/{volume_name}/v1/volumes/files/{remote_directory}%2F{dataset_file_name}' \
                --header 'Authorization: Bearer {token}' \
                --form 'upFile=@./{remote_directory}/{dataset_file_name}'
        """, format='json')

        print(res)

    # Create the connection

    res = run_ssh(f"""
        curl -kiv --location --request GET '{host}/v2/connections?project_id={project_id}' \
            --header 'Authorization: Bearer {token}'
    """, format='json')

    for e in res['resources']:
        if e['entity']['properties'].get('volume') == volume_name:
            connection_id = e['metadata']['asset_id']
            break

    if connection_id:
        print(f'Connection already exists: {connection_id}')
    else:
        res = run_ssh(f"""
            curl -kiv --location --request POST '{host}/v2/connections?project_id={project_id}' \
                --header 'Authorization: Bearer {token}' \
                --header 'Content-Type: application/json' \
                --data-raw '{{ "datasource_type": "9f30e3c3-b854-4144-b5c3-98b7a835dc79", "owner_id": "1000330999", "name": "{remote_directory}", "origin_country" : "us", "properties": {{ "volume": "{volume_name}", "trust_all_ssl_cert": "false", "password": "{password}", "gateway_url": "https://internal-nginx-svc:12443", "username": "{username}" }} }}'
        """, format='json')

        connection_id = res['metadata']['asset_id']

        print(f'Connection created: {connection_id}')

    # Get asset id
    # TODO

    asset_id = None

    ssh.close()

    return connection_id, f"/{remote_directory}/{dataset_file_name}", asset_id


def save_data_to_container(data_path, data_cos_path, wml_client, data_filenames=None):
    import ibm_watson_machine_learning._wrappers.requests as requests

    if wml_client.default_space_id:
        details = wml_client.spaces.get_details(wml_client.default_space_id)

    elif wml_client.default_project_id:
        project_url = wml_client.service_instance._href_definitions.get_project_href(
            wml_client.default_project_id)
        response = requests.get(
            project_url,
            headers=wml_client._get_headers()
        )

        details = response.json()
    else:
        raise ValueError("WML client should have set default project or space id.")

    properties = details['entity']['storage']['properties']
    properties.update(details['entity']['storage']['properties']['credentials']['editor'])
    properties.update(details['entity']['storage']['properties']['credentials']['editor'])

    save_data_to_cos_bucket(data_path=data_path,
                            data_cos_path=data_cos_path,
                            access_key_id=properties.get('access_key_id'),
                            secret_access_key=properties.get('secret_access_key'),
                            cos_endpoint=properties['endpoint_url'],
                            bucket_name=properties['bucket_name'],
                            data_filenames=data_filenames,
                            is_container=True)


def save_data_to_cos_bucket(data_path, data_cos_path, access_key_id, secret_access_key, cos_endpoint, bucket_name,
                            data_filenames=None, is_container=False):
    cos_resource = ibm_boto3.resource(
        service_name="s3",
        endpoint_url=cos_endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key
    )

    # Prepare bucket
    if not is_container:
        if not bucket_exists(cos_resource, bucket_name):
            bucket_name = create_bucket(cos_resource, bucket_name)

            assert bucket_exists(cos_resource, bucket_name)

    # Write data to cos bucket

    if os.path.isfile(data_path):
        cos_resource.Bucket(bucket_name).upload_file(data_path, data_cos_path)
    elif os.path.isdir(data_path):
        # upload multiple files
        for file in data_filenames:
            data_file_path = os.path.join(data_path, file) if data_path else file
            cos_file_path = os.path.join(data_cos_path, file) if data_cos_path else file

            cos_resource.Bucket(bucket_name).upload_file(data_file_path, cos_file_path)

    else:
        raise ValueError("data_path is not valid directory or file.")

    return bucket_name


def create_connection_to_cos(wml_client, cos_credentials, cos_endpoint, bucket_name, save_data=False, data_path=None,
                             data_cos_path=None, data_filenames=None):
    if save_data and data_path and data_cos_path:
        bucket_name = save_data_to_cos_bucket(data_path, data_cos_path,
                                              cos_credentials['cos_hmac_keys']['access_key_id'],
                                              cos_credentials['cos_hmac_keys']['secret_access_key'], cos_endpoint,
                                              bucket_name,
                                              data_filenames=data_filenames,
                                              is_container=False)

    auth_endpoint = cos_credentials.get('auth_endpoint')
    if auth_endpoint is None:
        auth_endpoint = wml_client.service_instance._href_definitions.get_iam_token_url()

    connection_details = wml_client.connections.create({
        'datasource_type': wml_client.connections.get_datasource_type_uid_by_name('bluemixcloudobjectstorage'),
        'name': f'Connection to COS for tests {datetime.datetime.utcnow().isoformat()}',
        'properties': {
            'bucket': bucket_name,
            'access_key': cos_credentials['cos_hmac_keys']['access_key_id'],
            'secret_key': cos_credentials['cos_hmac_keys']['secret_access_key'],
            'iam_url': auth_endpoint,
            'url': cos_endpoint
        }
    })

    connection_id = wml_client.connections.get_uid(connection_details)

    return connection_id, bucket_name
