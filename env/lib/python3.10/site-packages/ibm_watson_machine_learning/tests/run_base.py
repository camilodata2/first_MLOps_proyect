"""
Script which collects and executes tests for given environment using `pytest`.
Python Client tests operates on environmental variables, this script using command line
arguments is setting all the required envs and then execute tests compatible with provided configuration.

--------
Examples:
! Execute bellow commands in `~/python-client/python/ibm_watson_machine_learning/ibm_watson_machine_learning/tests dir.
- Run all tests for a given env:
```
    python run_base.py --env CLOUD --space_name sdk_base_space --bucket_name sdk_base_bucket
```
- Run specified test suite: [!Please specify suite by using names of existing subdirectories].
```
    python run_base.py --env CLOUD --space_name sdk_base_space --bucket_name sdk_base_bucket --test_suite deployments
```
- Run single test:
```
    python run_base.py --env CLOUD --space_name sdk_base_space --bucket_name sdk_base_bucket \
                       --test_path base/deployments/test_xgboost_from_object.py
```
"""
#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import glob
import argparse
import itertools
import contextlib

import pytest

# Dictionaries containing lists of supported frameworks for a give test suite.
cp4d_frameworks = {
    'deployments': [
        'ai_function',
        'pmml',
        'pytorch',
        'scikit',
        'spark',
        'tensorflow',
        'xgboost',
        'rshiny'
    ]
}

cloud_frameworks = {
    'deployments': [
        'ai_function',
        'do',
        'spss',
        'pmml',
        'pytorch',
        'scikit',
        'spark',
        'tensorflow',
        'xgboost',
    ]
}


def get_test_paths(suite_name=None):
    """
    Return path of all the test cases supported for a selected env.
    If suite_name is provided the tests will be limited for a given suite.
    """
    framework_dict = cp4d_frameworks if "CPD" in os.environ.get("ENV") else cloud_frameworks
    if suite_name is not None:
        framework_list = framework_dict[suite_name]
        path_template = f"base/{suite_name}/test_*{{}}*.py"
        tests_paths = [glob.glob(path_template.format(framework), recursive=True) for framework in framework_list]
    else:
        framework_list = [framework for key in framework_dict.keys() for framework in framework_dict[key]]
        path_template = "base/{}/test_*{}*.py"
        tests_paths = [glob.glob(path_template.format(suite_name, framework), recursive=True)
                       for suite_name in framework_dict.keys() for framework in framework_list]

    # Flatten the paths list.
    return list(itertools.chain.from_iterable(tests_paths))


def set_env_kwargs(args):
    """
    Creates a dictionary with required env variables for further testing.
    """
    env_kwargs = {"ENV": args.env}
    if args.space_name is not None:
        env_kwargs.update({"SPACE_NAME": args.space_name})
    if args.bucket_name is not None:
        env_kwargs.update({"BUCKET_NAME": args.bucket_name})
    return env_kwargs


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collects and executes tests for a given environment.")
    parser.add_argument("--env", type=str, dest="env",
                        help="Environment for which tests will be executed [for example `CLOUD`]")
    parser.add_argument("--space-name", type=str, dest="space_name", default=None,
                        help="Name of the space which is going to be used during the tests.")
    parser.add_argument("--bucket-name", type=str, dest="bucket_name", default=None,
                        help="Bucket name used in tests.")
    parser.add_argument("--test-suite", type=str, dest="test_suite", default=None,
                        help="Provide this argument if only selected suite is relevant for testing "
                             "[Please use names of existing subdirectories].")
    parser.add_argument("--test-path", type=str, dest="test_path", default=None,
                        help="If this argument is set only one test from a given path will be executed.")
    parser.add_argument("--junitxml", type=str, dest="junitxml", default=None,
                        help="Path for saved tests report in xml format (used in jenkins pipelines).")
    args = parser.parse_args()

    with set_env(**set_env_kwargs(args)):
        try:
            test_paths = get_test_paths(args.test_suite)
        except KeyError:
            raise NotADirectoryError("Provided `test_suite` isn't an existing directory!")

        # If test-path is provided execute only one test
        if args.test_path is not None:
            if os.path.exists(args.test_path):
                test_paths = [args.test_path]
            else:
                raise FileNotFoundError("Provided `test_path` does not exists!")

        # Execute collected tests using pytest.
        pytest_args = test_paths if args.junitxml is None else test_paths + ["--junitxml", args.junitxml]
        pytest.main(args=pytest_args)
