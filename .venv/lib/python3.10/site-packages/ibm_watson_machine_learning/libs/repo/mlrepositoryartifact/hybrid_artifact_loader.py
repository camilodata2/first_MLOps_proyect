#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import shutil
import zipfile

from ibm_watson_machine_learning.utils.autoai.utils import try_import_joblib
from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate


class HybridArtifactLoader(object):
    def load(self, artifact_queryparam):

        if artifact_queryparam is None:
           return self.extract_content(artifact_queryparam, HybridArtifactLoader._load_content)

        if artifact_queryparam is not None and artifact_queryparam == "full":
            return self.extract_content(artifact_queryparam, HybridArtifactLoader._load_content)

        if artifact_queryparam is not None and artifact_queryparam == "pipeline_model":
            return self.extract_content_json()

    def extract_content(self, queryparam_val, callback):
        directory_name = 'artifact'
        try:
            shutil.rmtree(directory_name)
        except:
            pass

        try:
            id_length = 20
            dir_id = uid_generate(id_length)
            model_dir_name = directory_name + dir_id

            gz_file_name = '{}/artifact_content.tar.gz'.format(model_dir_name)

            input_stream = None
            os.makedirs(model_dir_name)
            if queryparam_val is None:
                input_stream = self.reader().read()
            if queryparam_val is not None and queryparam_val == 'full':
                input_stream = self._content_reader_gzip()
            file_content = input_stream.read()
            gz_f = open(gz_file_name, 'wb+')
            gz_f.write(file_content)
            gz_f.close()
            if queryparam_val is None:
                self.reader().close()

            with zipfile.ZipFile(gz_file_name) as zip_ref:
                zip_ref.extractall(model_dir_name)
            artifact_instance = callback(model_dir_name)

            shutil.rmtree(model_dir_name)
            return artifact_instance
        except Exception as ex:
            shutil.rmtree(model_dir_name)
            raise ex

    def extract_content_json(self):
        directory_name = 'artifact'
        try:
            shutil.rmtree(directory_name)
        except:
            pass

        try:
            id_length = 20
            dir_id = uid_generate(id_length)
            model_dir_name = directory_name + dir_id

            output_file_name = '{}/pipeline_model.json'.format(model_dir_name)

            os.makedirs(model_dir_name)
            input_stream = self._content_reader_json()
            file_content = input_stream.read()
            gz_f = open(output_file_name, 'wb+')
            gz_f.write(file_content)
            gz_f.close()
            return output_file_name
        except Exception as ex:
            shutil.rmtree(model_dir_name)
            raise ex

    def _content_reader_gzip(self):
        if self._download_href is not None:
            if self._download_href.__contains__("models"):
                if self.meta.prop('space_id'):
                    download_url = self._download_href + f"&space_id={self.meta.prop('space_id')}"\
                                                         "&content_format=pipeline-node&pipeline_node_id=automl"
                elif self.meta.prop('project_id'):
                    download_url = self._download_href + f"&project_id={self.meta.prop('project_id')}" \
                                                         "&content_format=pipeline-node&pipeline_node_id=automl"
                return self.client.repository_api.download_artifact_content_v4_cloud(download_url, 'true')

    def _content_reader_json(self):
        if self._download_href is not None:
            if self._download_href.__contains__("models"):
                download_url = self._download_href + f"&space_id={self.meta.prop('space_id')}&content_format=native"
                return self.client.repository_api.download_artifact_content_v4_cloud(download_url, 'true')

    @staticmethod
    def _load_content(content_dir):
        """
        Load AutoAI pipeline into a local runtime.
        """
        joblib = try_import_joblib()
        extracted_file = [file for file in os.listdir(content_dir) if file.endswith(".pickle")][0]
        return joblib.load(os.path.join(content_dir, extracted_file))
