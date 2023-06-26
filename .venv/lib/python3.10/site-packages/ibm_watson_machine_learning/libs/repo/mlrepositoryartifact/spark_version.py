#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watson_machine_learning.libs.repo.util.library_imports import LibraryChecker
from ibm_watson_machine_learning.libs.repo.base_constants import *

lib_checker = LibraryChecker()
if lib_checker.installed_libs[PYSPARK]:
    from pyspark import SparkContext, SparkConf

class SparkVersion(object):
    @staticmethod
    def significant():
        lib_checker.check_lib(PYSPARK)
        conf = SparkConf()
        sc = SparkContext.getOrCreate(conf=conf)
        version_parts = sc.version.split('.')
        spark_version = version_parts[0]+'.' + version_parts[1]
        return format(spark_version)