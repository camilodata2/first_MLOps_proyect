#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.util.unique_id_gen import uid_generate
import os, shutil

class FunctionArtifactLoader(object):

    def load(self,path=None):
        return self.extract_content(path)

    def extract_content(self, file_path=None):
        if file_path is None:
            file_path = 'function_artifact'
            try:
                shutil.rmtree(file_path)
            except:
                pass
            os.makedirs(file_path)

        try:
            id_length = 20
            lib_generated_id = uid_generate(id_length)

            file_name = '{}/{}_{}.zip'.format(file_path, "function_artifact", lib_generated_id)
            input_stream = self.reader().read()
            file_content = input_stream.read()
            gz_f = open(file_name, 'wb+')
            gz_f.write(file_content)
            gz_f.close()
            return os.path.abspath(file_name)
        except Exception as ex:
            raise ex
