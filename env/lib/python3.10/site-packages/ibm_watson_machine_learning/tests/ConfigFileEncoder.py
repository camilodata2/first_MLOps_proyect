#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os, random, struct
from Crypto.Cipher import AES
import sys

def encrypt_file(key, in_filename, out_filename=None, chunksize=64 * 1024):
    """ Encrypts a file using AES (CBC mode) with the
        given key.

        key:
            The encryption key - a string that must be
            either 16, 24 or 32 bytes long. Longer keys
            are more secure.

        in_filename:
            Name of the input file

        out_filename:
            If None, '<in_filename>.enc' will be used.

        chunksize:
            Sets the size of the chunk which the function
            uses to read and encrypt the file. Larger chunk
            sizes can be faster for some files and machines.
            chunksize must be divisible by 16.
    """
    if not out_filename:
        out_filename = in_filename + '.enc'
    iv = os.urandom(16)

    encryptor = AES.new(key, AES.MODE_CBC, iv)
    filesize = os.path.getsize(in_filename)

    with open(in_filename, 'rb') as infile:
        with open(out_filename, 'wb') as outfile:
            outfile.write(struct.pack('<Q', filesize))
            outfile.write(iv)

            while True:
                chunk = infile.read(chunksize)
                n = len(chunk)
                if len(chunk) == 0:
                    break
                elif len(chunk) % 16 != 0:
                    chunk += bytes(' ', 'utf-8') * (16 - n % 16)

                outfile.write(encryptor.encrypt(chunk))


try:
    key = sys.argv[1]
except:
    key = None
    print("Usage : ConfigFileEncoder.py  <16 bit key> <input config file> <output file>")
    sys.exit(1)

try:
    input_filename = sys.argv[2]
except:
    input_filename = None
    print("Usage : ConfigFileEncoder.py  <16 bit key> <input config file> <output file>")
    sys.exit(1)

try:
    output_filename = sys.argv[3]
except:
    output_filename = input_filename + '.enc'

encrypt_file(key, input_filename, output_filename)
