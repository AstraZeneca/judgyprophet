import warnings

import pytest

from judgyprophet import JudgyProphet, utils


def pytest_configure(config):
    """
    Ensure STAN model is compiled before tests are run.
    """
    jp = JudgyProphet()
    if jp.stan_model is None:
        utils.log_and_warn(
            (
                "Model has not been compiled yet."
                " Attempting to compile STAN code now..."
            )
        )
        jp.compile()
        # Check compile worked properly
        if jp.stan_model is None:
            utils.log_and_raise(
                (
                    "STAN model failed to compile."
                    " Please check your pystan installation is working correctly."
                    " Also check your operating system has a valid C++ compiler installed."
                ),
                ValueError,
            )
