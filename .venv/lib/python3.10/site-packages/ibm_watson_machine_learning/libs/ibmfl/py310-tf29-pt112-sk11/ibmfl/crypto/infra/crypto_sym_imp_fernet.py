#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from cryptography.fernet import Fernet

from ibmfl.crypto.infra.crypto_sym_int import CryptoSym


class CryptoSymFernet(CryptoSym):
    """
    This class implements the interface for symmetric encryption functions using Fernet.
    """

    def __init__(self, key: bytes = None, **kwargs):
        super(CryptoSymFernet, self).__init__(key)
        if key is None:
            self.generate_key()
        else:
            self.key = key
            self.cipher = Fernet(self.key)

    def generate_key(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        return
