import logging
import warnings
from typing import Type

logger = logging.getLogger(__name__)


def log_and_raise(msg: str, error: Type[BaseException]):
    """
    Add an error to logging then raise the corresponding error.

    :param: msg (string) the message to print to log.
    :param: error the error to raise.
    ...
    :raises: error of type `error` param.
    """
    logger.error(msg)
    raise error(msg)


def log_and_warn(msg: str):
    """
    Add an error to logging then raise the corresponding error.

    :param: msg (string) the message to print to log.
    :param: error the error to raise.
    ...
    :raises: error of type `error` param.
    """
    logger.warning(msg)
    warnings.warn(msg)


def assert_log_raise(condition: bool, msg: str, error: Type[BaseException]):
    """
    If condition is False, log then raise corresponding error.

    :param: condition (bool) condition to check true.
    :param: msg (string) the message to print to log.
    :param: error the error to raise.
    ...
    :raises: error of type `error` param.
    """
    if not condition:
        logger.error(msg)
        raise error(msg)


def assert_log_warn(condition: bool, msg: str):
    """
    If condition is False, log then raise warning.

    :param: condition (bool) condition to check true.
    :param: msg (string) the message to print to log.
    ...
    :raises: warning if condition false.
    """
    if not condition:
        logger.warning(msg)
