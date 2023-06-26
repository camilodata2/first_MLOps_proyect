#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2020- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import subprocess
import os


def get_openshift_token(username, password, server):
    try:
        cmd = subprocess.run(["./oc login -u {} -p {} --server={} --insecure-skip-tls-verify=true > /dev/null && ./oc whoami --show-token=true".format(
            username,
            password,
            server
        )],
            cwd=os.path.dirname(os.path.realpath(__file__)),
            shell=True, capture_output=True, check=True)
        token = str(cmd.stdout, 'utf-8').strip()
        print("Token: {}".format(token))
    except subprocess.CalledProcessError as ex:
        print("Command execution failed with code: {}, reason:\n{}".format(ex.returncode, ex.stderr))
        token = None

    return token

