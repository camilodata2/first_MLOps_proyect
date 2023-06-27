#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from shutil import rmtree, copyfileobj
import logging
import numpy as np
import nose.tools as nt
import inspect
import os
import pprint

import ibm_watson_machine_learning._wrappers.requests as requests
import base64
import json
import tarfile, gzip, shutil


# WML Repo python client
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaNames
from ibm_watson_machine_learning.libs.repo.mlrepository import MetaProps
from ibm_watson_machine_learning.libs.repo.mlrepositoryclient import MLRepositoryClient
from ibm_watson_machine_learning.libs.repo.mlrepositoryartifact import MLRepositoryArtifact


def download_content(request_id, dep_id, details_json):
    request_reference = {'requestid': request_id, 'deploymentid': dep_id}
    headers = {'Authorization': details_json['mltoken'],
               'Content-Type': 'application/octet-stream-stream',
               'Accept': 'application/octet-stream'}
    try:
        model_content = requests.get(details_json['href'] + '/content', headers=headers).content

    except Exception as ex:
        import traceback
        stack_err = traceback.format_exc()
    return model_content

def decompress_file_gzip(gzip_filepath, filepath):
    with gzip.open(gzip_filepath, 'rb') as f:
        content = f.read()
        output_f = open(filepath, 'wb+')
        output_f.write(content)
        output_f.close()

def extract_tar(archive_path, path):
    tar = tarfile.open(archive_path)
    tar.extractall(path)
    tar.close()

def get_model_archive(request_id, dep_id, details_json):
    request_reference = {'requestid': request_id, 'deploymentid': dep_id}
    DOWNLOAD_BASE_PATH = "./tf_models/"
    download_dir_name = DOWNLOAD_BASE_PATH + "tf_model_" + request_id
    tar_file_name = '{}/artifact_content.tar'.format(download_dir_name)
    gz_file_name = '{}/artifact_content.tar.gz'.format(download_dir_name)

    os.makedirs(download_dir_name)

    model_content = download_content(request_id, dep_id, details_json)

    try:
        modelArcFile = open(gz_file_name, "wb")
        modelArcFile.write(model_content)
        modelArcFile.close()
    except Exception as ex:
        import traceback
        stack_err = traceback.format_exc()


    try:
        decompress_file_gzip(gz_file_name, tar_file_name)
        extract_tar(tar_file_name, download_dir_name)
    except Exception as ex1:
        import traceback
        stack_err = traceback.format_exc()
    return download_dir_name


def is_valid_tf_archive(gz_file):
    expected_file = "./saved_model.pb"

    try:
        tar = tarfile.open(gz_file, 'r:gz')
    except Exception as ex:
        raise IOError('Unable to read the compressed archive file in {0} due to '
                         'error "{1}". '
                         'Ensure a valid tar archive is compressed in gzip format.'
                         .format(gz_file, ex))
    if expected_file in [file.name for file in tar.getmembers()]:
        tar.close()
        return True
    else:
        tar.close()
        return False


service_path = "http://ibm-watson-ml-fvt.stage1.mybluemix.net"
user = "0847e2f0-de37-4ef7-b32b-cc28ef00c009"
password = "c9b1be92-7eed-40c1-9d5b-387725436c83"

# Generate mltoken and store it in MLRepositoryClient > MLRepositoryApi > MLApiClient
ml_repository_client = MLRepositoryClient(service_path)
ml_repository_client.authorize(user, password)


tf_tar_gz = "/Users/krishna/skl/samples/tensorflow/tf_serve_10261406.tar.gz"
tf_tar_gz_err = "/Users/krishna/skl/samples/tensorflow/somefile.py.gz"
tf_tar_gz_err1 = "/Users/krishna/skl/samples/tensorflow/somefile.err.gz"
tf_tar_gz_err2 = "/Users/krishna/ngp/dl/p1/repository/library/python_v3/test/resources/tf_fvt_invalid.tar.gz"
tf_model_name = 'k_tf_local_tar2'
tf_model_metadata = {
    MetaNames.DESCRIPTION: "Tensorflow model for predicting Hand-written digits",
    MetaNames.AUTHOR_NAME: "Krishna",
    MetaNames.FRAMEWORK_NAME: "tensorflow",
    MetaNames.FRAMEWORK_VERSION: "1.2"
}


#### Validate the tar.gz file
# print(is_valid_tf_archive(tf_tar_gz))




##################################################################################################
#### Save the model to WML Repo
tf_model_tar_artifact = MLRepositoryArtifact(tf_tar_gz_err2,
                                         name=tf_model_name,
                                         meta_props=MetaProps(tf_model_metadata.copy()))
print(type(tf_model_tar_artifact))
saved_model = ml_repository_client.models.save(tf_model_tar_artifact)
pprint.pprint(saved_model.meta.get())
#### ENd of save to WML Repo
##################################################################################################

##################################################################################################
#### Load check using REST
# mlUrl = service_path
# mlUser = user
# mlPassword = password
#
# auth64 = mlUser + ":" + mlPassword
# wml_url = mlUrl + "/v2/identity/token"
# token_response = requests.get(wml_url, auth=(mlUser, mlPassword))
# wml_token = json.loads(token_response.text).get('token')
# print(wml_token)
# bearer = "Bearer "+ wml_token
#
# details_json = {'href': 'https://ibm-watson-ml-fvt.stage1.mybluemix.net/v3/ml_assets/models/f81ad6bd-26f6-4c6f-aae9-3fe82be5d318/versions/4cdc20a2-f1a1-445f-b43a-f12fc88b93b7',
#                 'mltoken': bearer }
#
# saved_dir = get_model_archive("reqid_2", "depid_2", details_json)
# print(saved_dir)
## End of Save
##################################################################################################











