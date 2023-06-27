#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2019- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os, random, struct
from Crypto.Cipher import AES
import sys

def decrypt_file(key, in_filename, out_filename=None, chunksize=24*1024):
    """ Decrypts a file using AES (CBC mode) with the
        given key. Parameters are similar to encrypt_file,
        with one difference: out_filename, if not supplied
        will be in_filename without its last extension
        (i.e. if in_filename is 'aaa.zip.enc' then
        out_filename will be 'aaa.zip')
    """
    if not out_filename:
        out_filename = os.path.splitext(in_filename)[0]

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        decryptor = AES.new(key, AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)


try:
    key = sys.argv[1]
except:
    key = None
    print("Usage : ConfigFileGenerator.py  <16 bit key> <input config file> <output file>")
    sys.exit(1)

try:
    input_filename = sys.argv[2]
except:
    input_filename = None
    print("Usage : ConfigFileGenerator.py  <16 bit key> <input config file> <output file>")
    sys.exit(1)

try:
    output_filename = sys.argv[3]
except:
    output_filename = os.path.splitext(input_filename)[0]

decrypt_file(key, input_filename, output_filename)
