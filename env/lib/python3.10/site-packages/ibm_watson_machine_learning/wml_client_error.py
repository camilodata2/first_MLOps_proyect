#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2017- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging
import sys

__all__ = [
    "WMLClientError",
    "MissingValue",
    "MissingMetaProp",
    "NotUrlNorUID",
    "NoWMLCredentialsProvided",
    "ApiRequestFailure",
    "UnexpectedType",
    "ForbiddenActionForPlan",
    "NoVirtualDeploymentSupportedForICP",
    "MissingArgument",
    "WrongEnvironmentVersion",
    "CannotAutogenerateBedrockUrl",
    "WrongMetaProps",
    "CannotSetProjectOrSpace",
    "ForbiddenActionForGitBasedProject",
    "CannotInstallLibrary",
    "DataStreamError",
    "WrongLocationProperty",
    "WrongFileLocation",
    "EmptyDataSource",
    "SpaceIDandProjectIDCannotBeNone"
]


class WMLClientError(Exception):
    def __init__(self, error_msg, reason=None, logg_messages=True):
        self.error_msg = error_msg
        self.reason = reason
        if logg_messages:
            logging.getLogger(__name__).warning(self.__str__())
            logging.getLogger(__name__).debug(
                str(self.error_msg) + ('\nReason: ' + str(self.reason) if sys.exc_info()[0] is not None else ''))

    def __str__(self):
        return str(self.error_msg) + ('\nReason: ' + str(self.reason) if self.reason is not None else '')


class MissingValue(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, 'No \"' + value_name + '\" provided.', reason)


class MissingMetaProp(MissingValue):
    def __init__(self, name, reason=None):
        WMLClientError.__init__(self, 'Missing meta_prop with name: \'{}\'.'.format(name), reason)


class NotUrlNorUID(WMLClientError, ValueError):
    def __init__(self, value_name, value, reason=None):
        WMLClientError.__init__(self,
                                'Invalid value of \'{}\' - it is not url nor uid: \'{}\''.format(value_name, value),
                                reason)


class NoWMLCredentialsProvided(MissingValue):
    def __init__(self, reason=None):
        MissingValue.__init__(self, 'WML credentials', reason)


class ApiRequestFailure(WMLClientError):
    def __init__(self, error_msg, response, reason = None):
        self.response = response
        if str(response.status_code) == '404' and 'DOCTYPE' in str(response.content):
            raise MissingWMLComponent()

        elif (str(response.status_code) == '400' and
              'Invalid content. You cannot include any tags in the HTTP request.' in str(response.content)):
            WMLClientError.__init__(self, f"Please check if any parameter that you provided include HTTP tag. "
                                          f"If yes, please remove it and try again.", reason=str(response.content))

        else:
            WMLClientError.__init__(self,
                                    '{} ({} {})\nStatus code: {}, body: {}'.format(
                                        error_msg,
                                        response.request.method,
                                        response.request.url,
                                        response.status_code,
                                        response.text if response.apparent_encoding is not None else '[binary content, ' + str(
                                            len(response.content)) + ' bytes]'), reason)


class UnexpectedType(WMLClientError, ValueError):
    def __init__(self, el_name, expected_type, actual_type):
        WMLClientError.__init__(self, 'Unexpected type of \'{}\', expected: {}, actual: \'{}\'.'.format(el_name,
                                                                                                        '\'{}\''.format(
                                                                                                            expected_type) if type(
                                                                                                            expected_type) == type else expected_type,
                                                                                                        actual_type))


class ForbiddenActionForPlan(WMLClientError):
    def __init__(self, operation_name, expected_plans, actual_plan):
        WMLClientError.__init__(self,
                                'Operation \'{}\' is available only for {} plan, while this instance has \'{}\' plan.'.format(
                                    operation_name, ('one of {} as'.format(expected_plans) if len(
                                        expected_plans) > 1 else '\'{}\''.format(expected_plans[0])) if type(
                                        expected_plans) is list else '\'{}\''.format(expected_plans), actual_plan))


class NoVirtualDeploymentSupportedForICP(MissingValue):
    def __init__(self, reason=None):
        MissingValue.__init__(self, 'No Virtual deployment supported for ICP', reason)


class MissingArgument(WMLClientError, ValueError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Argument: {value_name} missing.", reason)


class WrongEnvironmentVersion(WMLClientError, ValueError):
    def __init__(self, used_version, environment_name, supported_versions):
        WMLClientError.__init__(self, "Version used in credentials not supported in this environment",
                                reason=f"Version {used_version} isn't supported in "
                                       f"{environment_name} environment, "
                                       f"select from {supported_versions}")


class CannotAutogenerateBedrockUrl(WMLClientError, ValueError):
    def __init__(self, e1, e2):
        WMLClientError.__init__(self, "Attempt of generating `bedrock_url` automatically failed. "
                                      "If iamintegration=True, please provide `bedrock_url` in wml_credentials. "
                                      "If iamintegration=False, please validate your credentials.", reason=[e1, e2])


class WrongMetaProps(MissingValue):
    def __init__(self, reason=None):
        WMLClientError.__init__(self, "Wrong metaprops.", reason)


class MissingWMLComponent(WMLClientError):
    def __init__(self):
        WMLClientError.__init__(self, f"Missing WML Component",
                                reason=f"It appears that WML component is not installed on your environment. "
                                       f"Contact your cluster administrator.")


class CannotSetProjectOrSpace(WMLClientError):
    def __init__(self, reason):
        WMLClientError.__init__(self, f"Cannot set Project or Space",
                                reason=reason)


class ForbiddenActionForGitBasedProject(WMLClientError):
    def __init__(self, reason):
        WMLClientError.__init__(self, f"This action is not supported for git based project.",
                                reason=reason)


class CannotInstallLibrary(WMLClientError, ValueError):
    def __init__(self, lib_name: str, reason: str):
        WMLClientError.__init__(self, f"Library '{lib_name}' cannot be installed! Please install it manually.", reason)


class DataStreamError(WMLClientError, ConnectionError):
    def __init__(self, reason):
        WMLClientError.__init__(self, "Cannot fetch data via Flight Service. Try again.", reason)


class WrongLocationProperty(WMLClientError, ConnectionError):
    def __init__(self, reason):
        WMLClientError.__init__(self, "Cannot fetch data via Flight Service. Try again.", reason)


class WrongFileLocation(WMLClientError, ValueError):
    def __init__(self, reason):
        WMLClientError.__init__(self, "Cannot fetch data via Flight Service. Try again.", reason)

class EmptyDataSource(WMLClientError, ValueError):
    def __init__(self ):
        WMLClientError.__init__(self, "Cannot fetch data via Flight Service. "
                                      "Verify if data were saved under data connection and try again.")

class SpaceIDandProjectIDCannotBeNone(WMLClientError, ValueError):
    def __init__(self, reason: str):
        WMLClientError.__init__(self, f"Missing 'space_id' or 'project_id'.", reason)
