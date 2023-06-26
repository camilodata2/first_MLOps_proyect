#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import re

class GenericArchiveFrameworkCheck(object):
    @staticmethod
    def is_archive_framework(name):
        if (re.match('spss-modeler', name) is not None) or (re.match('pmml', name) is not None) \
          or (re.match('caffe', name) is not None) or (re.match('caffe2', name) is not None) \
          or (re.match('torch',name) is not None) \
          or (re.match('pytorch', name) is not None) or (re.match('blueconnect', name) is not None) \
          or (re.match('mxnet', name) is not None) or (re.match('theano', name) is not None) \
          or (re.match('darknet', name) is not None):
          return True
        else:
         return False
