#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from random import choice
import string

def uid_generate(id_length):
        return ''.join(choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(id_length))