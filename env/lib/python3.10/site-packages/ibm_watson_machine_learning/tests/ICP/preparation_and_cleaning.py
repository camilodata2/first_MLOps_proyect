#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from configparser import ConfigParser
import json
import os
import io
import sys
import ibm_boto3
from ibm_botocore.client import Config
import wget
import uuid


if "ENV" in os.environ:
    environment = os.environ['ENV']
else:
    environment = "SVT"


config = ConfigParser()
config.read('../config.ini')

MNIST = 'mnist'
MNIST_LMDB = 'mnist_lmdb'
BOSTON = 'boston'
BAIR_BVLC = 'bair_bvlc_caffenet'


def get_env():
    return environment


def get_wml_credentials():
    return json.loads(config.get(environment, 'wml_credentials'))


def get_cos_credentials():
    return json.loads(config.get(environment, 'cos_credentials'))


def get_feedback_data_reference():
    return json.loads(config.get(environment, 'feedback_data_reference'))


def get_spark_reference():
    return json.loads(config.get(environment, 'spark_reference'))


def get_cos_auth_endpoint():
    return config.get(environment, 'cos_auth_endpoint')


def get_cos_service_endpoint():
    return config.get(environment, 'cos_service_endpoint')


def get_client():
    wml_lib = __import__('ibm_watson_machine_learning', globals(), locals())
    return wml_lib.APIClient(get_wml_credentials())


def get_cos_resource():
    cos_credentials = get_cos_credentials()
    api_key = cos_credentials['apikey']
    service_instance_id = cos_credentials['resource_instance_id']
    auth_endpoint = get_cos_auth_endpoint()
    service_endpoint = get_cos_service_endpoint()

    cos = ibm_boto3.resource(
        's3',
        ibm_api_key_id = api_key,
        ibm_service_instance_id = service_instance_id,
        ibm_auth_endpoint = auth_endpoint,
        config = Config(signature_version='oauth'),
        endpoint_url = service_endpoint
    )

    return cos


def prepare_cos(cos_resource, bucket_prefix='wml-test', data_code = MNIST):
    client = get_client()
    clean_env(client, cos_resource)
    import datetime

    postfix = datetime.datetime.now().isoformat().replace(":", "-").split(".")[0].replace("T", "-")

    bucket_names = {
        'data': '{}-{}-data-{}'.format(bucket_prefix, environment.lower(), postfix),
        'results': '{}-{}-results-{}'.format(bucket_prefix, environment.lower(), postfix)
    }

    cos_resource.create_bucket(Bucket=bucket_names['data'])
    upload_data(cos_resource, bucket_names['data'], data_code)

    cos_resource.create_bucket(Bucket=bucket_names['results'])

    return bucket_names


def download_data(data_code, data_dir):
    if data_code == MNIST:
        data_links = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            'https://s3.amazonaws.com/img-datasets/mnist.npz'
        ]
    elif data_code == BOSTON:
        raise Exception('Unsupported.')
    elif data_code == BAIR_BVLC:
        data_links = [
            'http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel'
        ]
    else:
        raise Exception('Unrecognized data_code.')

    if data_dir is not None:
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

    for link in data_links:
        if not os.path.isfile(os.path.join(data_dir, os.path.join(link.split('/')[-1]))):
            wget.download(link, out=data_dir)


def upload_data(cos_resource, bucket_name, data_code=MNIST):
    if data_code == MNIST:
        data_dir = os.path.join('datasets', 'MNIST_DATA')
    elif data_code == BAIR_BVLC:
        data_dir = os.path.join('datasets', 'BAIR_BVLC')
    elif data_code == BOSTON:
        data_dir = os.path.join('datasets', 'boston', 'BOSTON_DATA')
    elif data_code == MNIST_LMDB:
        data_dir = os.path.join('datasets', 'mnist_data_lmdb', 'mnist_train_lmdb')
    else:
        raise Exception('Unrecognized data_code.')

    try:
        download_data(data_code, data_dir)
    except:
        pass

    bucket_obj = cos_resource.Bucket(bucket_name)

    if data_code == MNIST or data_code == BAIR_BVLC or data_code == BOSTON:
        for filename in os.listdir(data_dir):
            with open(os.path.join(data_dir, filename), 'rb') as data:
                bucket_obj.upload_file(os.path.join(data_dir, filename), filename)
                print('{} is uploaded.'.format(filename))
    elif data_code == MNIST_LMDB:
        bucket_obj.upload_file(os.path.join('datasets', 'mnist_data_lmdb', 'mnist_test_lmdb', 'data.mdb'), 'mnist_test_lmdb/data.mdb')
        bucket_obj.upload_file(os.path.join('datasets', 'mnist_data_lmdb', 'mnist_test_lmdb', 'lock.mdb'), 'mnist_test_lmdb/lock.mdb')
        bucket_obj.upload_file(os.path.join('datasets', 'mnist_data_lmdb', 'mnist_train_lmdb', 'data.mdb'), 'mnist_train_lmdb/data.mdb')
        bucket_obj.upload_file(os.path.join('datasets', 'mnist_data_lmdb', 'mnist_train_lmdb', 'lock.mdb'), 'mnist_train_lmdb/lock.mdb')
    else:
        raise Exception('No such dataset available')


    for obj in bucket_obj.objects.all():
        print('Object key: {}'.format(obj.key))
        print('Object size (kb): {}'.format(obj.size/1024))


def get_cos_training_data_reference(bucket_names):
    cos_credentials = get_cos_credentials()
    service_endpoint = get_cos_service_endpoint()

    return {
        "connection": {
            "endpoint_url": service_endpoint,
            "access_key_id": cos_credentials['cos_hmac_keys']['access_key_id'],
            "secret_access_key": cos_credentials['cos_hmac_keys']['secret_access_key']
        },
        "source": {
            "bucket": bucket_names['data'],
        },
        "type": "s3"
    }


def get_cos_training_results_reference(bucket_names):
    cos_credentials = get_cos_credentials()
    service_endpoint = get_cos_service_endpoint()

    return {
        "connection": {
            "endpoint_url": service_endpoint,
            "access_key_id": cos_credentials['cos_hmac_keys']['access_key_id'],
            "secret_access_key": cos_credentials['cos_hmac_keys']['secret_access_key']
        },
        "target": {
            "bucket": bucket_names['results'],
        },
        "type": "s3"
    }

def clean_cos_bucket(cos_resource, bucket_name):
    bucket_obj = cos_resource.Bucket(bucket_name)
    for upload in bucket_obj.multipart_uploads.all():
        upload.abort()
    for o in bucket_obj.objects.all():
        o.delete()
    bucket_obj.delete()


def clean_cos(cos_resource, bucket_names):
    clean_cos_bucket(cos_resource, bucket_names['data'])
    clean_cos_bucket(cos_resource, bucket_names['results'])


from datetime import datetime, timedelta
from pytz import timezone
yesterday = datetime.now(timezone('UTC')) - timedelta(days=1)


def clean_env(client, cos_resource, threashold_date=yesterday):
    clean_experiments(client, threashold_date)
    clean_training_runs(client, threashold_date)
    clean_definitions(client, threashold_date)
    clean_models(client, threashold_date)
    clean_deployments(client, threashold_date)
    try:
        clean_ai_functions(client, threashold_date)
        clean_runtimes(client, threashold_date)
        clean_custom_libraries(client, threashold_date)
    except:
        pass

    if cos_resource is not None:
        for bucket in cos_resource.buckets.all():
            if 'wml-test-' in bucket.name and bucket.creation_date < threashold_date:
                print('Deleting \'{}\' bucket.'.format(bucket.name))
                try:
                    for upload in bucket.multipart_uploads.all():
                        upload.abort()
                    for o in bucket.objects.all():
                        o.delete()
                    bucket.delete()
                except Exception as e:
                    print("Exception during bucket deletion occured: " + str(e))


def clean_models(client, threashold_date=yesterday):
    details = client.repository.get_model_details()

    for model_details in details['resources']:
        if model_details['metadata']['created_at'] < threashold_date.isoformat():
            print('Deleting \'{}\' model.'.format(model_details['metadata']['guid']))
            try:
                client.repository.delete(model_details['metadata']['guid'])
            except Exception as e:
                print('Deletion of model failed: ' + str(e))


def clean_definitions(client, threashold_date=yesterday):
    details = client.repository.get_definition_details()

    for definition_details in details['resources']:
        if definition_details['metadata']['created_at'] < threashold_date.isoformat():
            print('Deleting \'{}\' definition.'.format(definition_details['metadata']['guid']))
            try:
                client.repository.delete(definition_details['metadata']['guid'])
            except Exception as e:
                print('Deletion of definition failed: ' + str(e))


def clean_deployments(client, threashold_date=yesterday):
    details = client.deployments.get_details()

    for deployment_details in details['resources']:
        if deployment_details['metadata']['created_at'] < threashold_date.isoformat():
            print('Deleting \'{}\' deployment.'.format(deployment_details['metadata']['guid']))
            try:
                client.deployments.delete(deployment_details['metadata']['guid'])
            except Exception as e:
                print('Deletion of deployment failed: ' + str(e))


def clean_experiments(client, threashold_date=yesterday):
    details = client.repository.get_experiment_details()

    for experiment_details in details['resources']:
        if experiment_details['metadata']['created_at'] < threashold_date.isoformat():
            print('Deleting \'{}\' experiment.'.format(experiment_details['metadata']['guid']))
            try:
                client.repository.delete(experiment_details['metadata']['guid'])
            except Exception as e:
                print('Deletion of experiment failed: ' + str(e))


def clean_training_runs(client, threashold_date=yesterday):
    details = client.training.get_details()

    for run_details in details['resources']:
        if run_details['metadata']['created_at'] < threashold_date.isoformat():
            print('Deleting \'{}\' training run.'.format(run_details['metadata']['guid']))
            try:
                client.training.delete(run_details['metadata']['guid'])
            except Exception as e:
                print('Deletion of training run failed: ' + str(e))


def clean_runtimes(client, threashold_date=yesterday):
    details = client.runtimes.get_details()

    for runtime_details in details['resources']:
        if runtime_details['metadata']['created_at'] < threashold_date.isoformat():
            print('Deleting \'{}\' runtime.'.format(runtime_details['metadata']['guid']))
            try:
                client.runtimes.delete(runtime_details['metadata']['guid'])
            except Exception as e:
                print('Deletion of runtime failed: ' + str(e))


def clean_custom_libraries(client, threashold_date=yesterday):
    details = client.runtimes.get_library_details()

    for lib_details in details['resources']:
        if lib_details['metadata']['created_at'] < threashold_date.isoformat():
            print('Deleting \'{}\' custom library.'.format(lib_details['metadata']['guid']))
            try:
                client.runtimes.delete_library(lib_details['metadata']['guid'])
            except Exception as e:
                print('Deletion of custom library failed: ' + str(e))


def clean_ai_functions(client, threashold_date=yesterday):
    details = client.repository.get_function_details()

    for func_details in details['resources']:
        if func_details['metadata']['created_at'] < threashold_date.isoformat():
            print('Deleting \'{}\' AI function.'.format(func_details['metadata']['guid']))
            try:
                client.repository.delete(func_details['metadata']['guid'])
            except Exception as e:
                print('Deletion of AI function failed: ' + str(e))


def run_monitor(client, experiment_run_uid, queue):
    stdout_ = sys.stdout
    captured_output = io.StringIO()  # Create StringIO object
    sys.stdout = captured_output  # and redirect stdout.
    client.experiments.monitor_logs(experiment_run_uid)
    sys.stdout = stdout_  # Reset redirect.

    print(captured_output.getvalue())

    queue.put(captured_output.getvalue())


def run_monitor_metrics(client, experiment_run_uid, queue):
    stdout_ = sys.stdout
    captured_output = io.StringIO()  # Create StringIO object
    sys.stdout = captured_output  # and redirect stdout.
    client.experiments.monitor_metrics(experiment_run_uid)
    sys.stdout = stdout_  # Reset redirect.

    print(captured_output.getvalue())

    queue.put(captured_output.getvalue())


def run_monitor_training(client, training_run_uid, queue):
    stdout_ = sys.stdout
    captured_output = io.StringIO()  # Create StringIO object
    sys.stdout = captured_output  # and redirect stdout.
    client.training.monitor_logs(training_run_uid)
    sys.stdout = stdout_  # Reset redirect.

    print(captured_output.getvalue())

    queue.put(captured_output.getvalue())
