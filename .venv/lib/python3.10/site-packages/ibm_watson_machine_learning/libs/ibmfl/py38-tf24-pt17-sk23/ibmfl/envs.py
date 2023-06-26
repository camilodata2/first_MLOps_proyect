#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import pathlib
from environs import Env

env = Env()
env.read_env()

# export FL_WORKING_DIR=/home/username/
# export FL_MODEL_DIR=/home/username/model


def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


with env.prefixed("FL_"):
    working_directory = env("WORKING_DIR", os.getcwd())
    create_dir(working_directory)

    model_directory = env("MODEL_DIR", working_directory)
    create_dir(model_directory)
