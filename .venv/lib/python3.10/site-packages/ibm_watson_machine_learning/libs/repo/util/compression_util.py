#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import tarfile, gzip, shutil


class CompressionUtil(object):
    @staticmethod
    def create_tar(path, arcname, archive_path):
        tar = tarfile.open(archive_path, "w")
        tar.add(path, arcname)
        tar.close()

    @staticmethod
    def extract_tar(archive_path, path):
        tar = tarfile.open(archive_path)
        tar.extractall(path)
        tar.close()

    @staticmethod
    def compress_file_gzip(filepath, gzip_filepath):
        with open(filepath, 'rb') as f_in, gzip.open(gzip_filepath, 'wb+') as f_out:
            shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def decompress_file_gzip(gzip_filepath, filepath):
        with gzip.open(gzip_filepath, 'rb') as f:
            content = f.read()
            output_f = open(filepath, 'wb+')
            output_f.write(content)
            output_f.close()
