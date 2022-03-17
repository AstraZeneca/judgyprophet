import copy
import logging
import pickle
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
import pystan

# Used for calculating initial values for optimizer
from scipy.stats import linregress

from judgyprophet import utils

logger = logging.getLogger(__name__)


class JudgyProphet:
    """
    A class to fit an 'judgyprophet' model: a forecasting algorithm which incorporates
    business information about known events using Bayesian informative priors.

    The input is the actuals, as well as information about a number of known
    change in level or change in trend events.
    This information is used to feed data to informative priors, which is used as
    a quantitative judgemental pre-adjustment.

    This model is a wrapper for the stan model 'judgyprophet.stan'.
    When the object is instantiated, the compiled STAN model is looked for in 'stan/build/judgyprophet.pkl'.

    If this is not available then the compile() method will need to be run before fit(). This will
    compile the stan code for future use.

    :param prebuild_path: (optional) a custom path to compile the stan build code to. When .compile()
        is run, this path will be used to save the compiled model. It will also be used to access the
        build path when the object is instantiated. The usecase for this is if you don't have write
        access to your python install path.
    """

    name = "judgyprophet"

    def __init__(self, prebuilt_path: str = None):

        # This will be populated with the compiled model once self.compile() method is run,
        #   or if prebuilt_path supplied
        self._model: Dict[str, Any] = {}
        self.model_output: Dict[str, Any] = {}
        self.fit_data: pd.Series = None
        self.fit_config: Dict[str, Any] = None

        # Load the compiled STAN code. Catch IOError and raise warning that model not compiled.
        self.stan_model = None

        # Ouput of STAN MCMC is stored here
        self.stan_file_path = (
            Path(__file__).resolve().parent / "stan" / f"{self.name}.stan"
        )
        logger.debug("Stan file path: %s", self.stan_file_path)
        if prebuilt_path:
            logger.debug("Using custom build path: %s", prebuilt_path)
            self.stan_build_path: PathLike = Path(prebuilt_path)
        else:
            self.stan_build_path = (
                Path(__file__).resolve().parent / "stan" / "build" / f"{self.name}.pkl"
            )
            logger.debug("Using default build path: %s", self.stan_build_path)

        try:
            logger.debug(
                "Opening build path to fetch model at '%s'", self.stan_build_path
            )
            with open(self.stan_build_path, "rb") as stannin:
                self.stan_model = pickle.load(stannin)
        except IOError:
            utils.log_and_warn(
                "STAN build path not found. Model STAN code needs to be compiled before you can use the package."
                " Try to compile the STAN model using `JudgyProphet().compile()` before fitting model."
                " If that fails, check your pystan installation."
            )

        # Used for rescaling actuals -- in a similar way to Prophet
        self.actuals_loc = 0.0
        self.actuals_scale = 1.0
        self.rescaled_data = pd.Series([], dtype=float)

        # Event lists with added parameter estimates
        self.level_events: List[Dict[str, Any]] = []
        self.trend_events: List[Dict[str, Any]] = []
        self.unspecified_changepoints: List[Dict[str, Any]] = []

        # Dict to store fit config
        self._fit_config: Dict[str, Any] = {}

    def fit(
        self,
        data: pd.Series,
        level_events: List[Dict[str, Any]],
        trend_events: List[Dict[str, Any]],
        unspecified_changepoints: Union[int, Iterable] = 0,
        seasonal_period: Optional[int] = None,
        seasonal_type: str = "add",
        event_start_cutoff: int = 4,
        event_end_cutoff: int = 0,
        fourier_order: Optional[int] = None,
        sigma_unspecified_changepoints: float = 0.1,
        sigma_base_bias: float = 1.0,
        sigma_base_trend: float = 1.0,
        sigma_level: float = 0.1,
        sigma_trend: float = 0.1,
        sigma_seasonal: float = 1.0,
        sigma_actuals: float = 0.5,
        starting_jitter: float = 0.1,
        **kwargs,
    ):
        """
        Fit a judgyprophet model to the actuals provided in 'data'.
        Changes in trend will be added to the dates in trend_events with initial gradient as defined.
        Changes in level will be added to the dates in level_events with initial shift as defined.

        This is a wrapper that samples from the STAN model 'stan/judgyprophet.stan' using the data provided.

        :param actuals: A pandas series of the actual timeseries to forecast.
            It is assumed there are no missing data points,
            i.e. x[1] is the observation directly following x[0], etc.
            The index of the series should indicate the time point of the observation.
            E.g. if observations are timestamps, the index would be pd.DatetimeIndex;
            if observations are indexed by discrete time points, the index would be an integer
            index. The indexes in level_events and trend_events should be the same type as the
            index of actuals.
        :param level_events: A list of dictionaries. Each dict should have the following entries
            - 'index' the start index of the event (i.e. index = i assumes the start of the event
                is at location actuals[i]). The index should be of the same type as the actuals index.
            - 'c0' the estimated level increase following the event
        :param trend_events: A list of dictionaries. Each dict should have the following entries
            - 'index' the start index of the event (i.e. index = i assumes the start of the event
                is at location actuals[i]). The index should be of the same type as the actuals index.
            - 'm0' the estimated gradient increase following the event
            - 'gamma' (Optional) the damping to use for the trend. This is a float between 0 and 1.
                It's not recommended to be below 0.8 and must be 0 > gamma <= 1.
                If gamma is missing from the dict, or gamma = 1, a linear trend is used (i.e. no damping).
        :param unspecified_changepoints: (optional, int or iterable, default 0) The number of 'prophet' like
            changepoints (i.e. learnt from the data as there has been an unanticipated change
            to the timeseries trend or level). Either an int of the number of changepoints which are
            distributed evenly along the data, or an ArrayLike of indexes of the data where
            changepoints are required. If it is an ArrayLike, each element should be the same type
            as the actuals index.
        :param seasonal_period: (optional, int, default None) The seasonal period of the timeseries.
            This should be an integer. If this is set to None or 1,
            no seasonality will be used. Seasonality is modelled using Fourier series.
        :param seasonal_type: (optional, default 'add') the type of seasonality to use:
            - 'add' for additive
            - 'mult' for multiplicative
        :param event_start_cutoff: (optional, int, default 4) the number of data points of history required
            before an event; before a changepoint will be put there (otherwise it is absorbed
            into the base trend and will not be learned).
        :param event_end_cutoff: (optional, int, default 0) the number of data points required after an event
            before a changepoint will be put there and its parameters will be learnt by the model
            (otherwise it is predicted using m0).
        :param fourier_order: (optional, int, default None) the number of fourier cos and sin
            terms to use for modelling seasonality. If None, this will be set to floor(seasonal_period / 2).
            This is useful if you have a long seasonal period and not much data. You can use
            a fourier order that is lower than half the seasonal period. It should be an integer
            number between 1 and floor(seasonal_period / 2).
        :param sigma_unspecified_changepoints: (optional, float, default .1) The standard
            deviation of the prior on the unspecified changepoints. The smaller this is,
            the higher the L1 penalty on the changepoints.
        :param sigma_base_bias: (optional, float, default 1.) The standard deviation of the prior on the base bias
            (i.e. bias before any changepoints or events are observed). The smaller this is,
            the higher the regularisation on the parameter.
        :param sigma_base_trend: (optional, float, default 1.) The standard deviation of the prior
            on the base trend (i.e. trend before any changepoints or events are observed).
            The smaller this is, the higher the regularisation on the parameter.
        :param sigma_level: (optional, float, default .1) The standard deviation of the prior on level events.
            The smaller this is, the more the parameter will be regularised towards the initial estimate 'c0'.
        :param sigma_trend: (optional, float, default .1) The standard deviation of the prior on trend events.
            The smaller this is, the more the parameter will be regularised towards the initial estimate 'm0'.
        :param sigma_seasonal: (optional, float, default 1.) The standard deviation of the prior on
            seasonality. The smaller this is, the more the parameter will be regularised towards 0.
        :param sigma_actuals: (optional, float, default .5) The standard deviation of the prior on actuals noise.
        :param starting_jitter: (optional, float, default .1) jitter in the starting values -- resolves optimisation problems.
        :param **kwargs: Any extra keyword arguments will be passed onto StanModel.sampling (e.g. iter, chains)
        """
        utils.assert_log_raise(
            self.stan_model is not None,
            (
                "STAN build path not found, meaning Model STAN code failed to compile properly at install."
                " Try to compile the STAN model by running JudgyProphet().compile() before fitting model."
                " If that fails, check your pystan installation."
            ),
            ValueError,
        )
        logger.debug(
            "Fitting judgyprophet with the following data:"
            "\n\t- data: %s"
            "\n\t- level_events: %s"
            "\n\t- trend_events: %s",
            data,
            level_events,
            trend_events,
        )

        self.fit_config = {
            "unspecified_changepoints": unspecified_changepoints,
            "seasonal_period": seasonal_period,
            "event_start_cutoff": event_start_cutoff,
            "event_end_cutoff": event_end_cutoff,
            "seasonal_type": seasonal_type,
            "fourier_order": fourier_order,
            "sigma_unspecified_changepoints": sigma_unspecified_changepoints,
            "sigma_base_bias": sigma_base_bias,
            "sigma_base_trend": sigma_base_trend,
            "sigma_level": sigma_level,
            "sigma_trend": sigma_trend,
            "sigma_seasonal": sigma_seasonal,
            "sigma_actuals": sigma_actuals,
            "starting_jitter": starting_jitter,
        }
        logger.debug(
            "Fitting 'judgyprophet' with the following settings: %s", self.fit_config
        )
        self.fit_data = self._prepare_data(data)
        # Move the specified index into a column
        self._fit_data_checks(trend_events=trend_events, level_events=level_events)
        self._prepare_event_data(level_events=level_events, trend_events=trend_events)
        stan_data = self._prepare_stan_data()
        stan_init = self._set_stan_init(stan_data)
        try:
            self._model = self.stan_model.optimizing(
                data=stan_data, init=stan_init, algorithm="LBFGS", **kwargs
            )
        except RuntimeError:
            utils.log_and_warn("Optimisation using LBGFS failed. Trying Newton method.")
            self._model = self.stan_model.optimizing(
                data=stan_data, init=stan_init, algorithm="Newton", **kwargs
            )
        self.model_output = {}

        # level events
        self.model_output["level_events"] = []
        for level_i in self.level_events:
            model_output_level = level_i.copy()
            # Fetch learnt parameters for each active level event
            if level_i["is_active"]:
                if len(self._model["c"].shape) == 0:
                    model_output_level["c"] = self._model["c"].item()
                else:
                    model_output_level["c"] = self._model["c"][
                        level_i["active_event_index"]
                    ]
            self.model_output["level_events"].append(model_output_level)

        # trend events
        self.model_output["trend_events"] = []
        for trend_i in self.trend_events:
            model_output_trend = trend_i.copy()
            # Fetch learnt parameters for each active trend event
            if trend_i["is_active"]:
                if len(self._model["m"].shape) == 0:
                    model_output_trend["m"] = self._model["m"].item()
                else:
                    model_output_trend["m"] = self._model["m"][
                        trend_i["active_event_index"]
                    ]
            self.model_output["trend_events"].append(model_output_trend)

        # changepoints
        self.model_output["changepoint_events"] = []
        for cp_event in self.unspecified_changepoints:
            model_output_cp = cp_event.copy()
            if len(self._model["cp_m"].shape) == 0:
                model_output_cp["cp_m"] = self._model["cp_m"].item()
                model_output_cp["cp_c"] = self._model["cp_c"].item()
            else:
                model_output_cp["cp_m"] = self._model["cp_m"][
                    cp_event["active_event_index"]
                ]
                model_output_cp["cp_c"] = self._model["cp_c"][
                    cp_event["active_event_index"]
                ]
            self.model_output["changepoint_events"].append(model_output_cp)

        # seasonality
        self.model_output["base"] = {
            "beta_0": self._model["beta_0"],
            "beta_1": self._model["beta_1"],
            "fourier_coefficients": self._model["fourier_coefficients"],
        }
        logger.debug("Model output: %s", self.model_output)

    def predict(self, horizon: int = 0) -> pd.DataFrame:
        """
        Predict model up to the supplied horizon. If no horizon is given, the insample fit is returned.

        :param horizon: (Optional, default 0) the horizon to predict upto. Must be a positive integer.
        ...
        :returns: a dataframe of prediction results, including a column indicating if the point is insample or not.
        """
        logger.debug("Predicting for horizon: %s", horizon)
        try:
            if len(self.fit_data) >= horizon:
                predict_ds = self.fit_data.index.union(
                    self.fit_data.index.shift(periods=horizon)
                )
            # Handle case when number of training points < prediction horizon
            else:
                utils.log_and_warn("Number of training points < horizon")
                # Because we will do a union at the end, and the predict horizon should not be super large,
                #  do a lazy loop-through
                predict_ds = self.fit_data.index.copy()
                for i in range(0, horizon + 1):
                    extra_indexes = self.fit_data.index.shift(periods=i)
                    predict_ds = predict_ds.union(extra_indexes)

        # .shift not implemented for integer style indexes. So handle this case.
        except NotImplementedError:
            logger.debug("Predicting for integer style index")
            predict_from_index = self.fit_data.index.max() + 1
            predict_ds = self.fit_data.index.union(
                range(predict_from_index, predict_from_index + horizon)
            )
        t = np.arange(len(predict_ds))
        predictions = (
            self.model_output["base"]["beta_0"]
            + self.model_output["base"]["beta_1"] * t
        )

        # Level events
        for level_i in self.model_output["level_events"]:
            # Skip inactive events
            if not level_i["is_predicted"]:
                continue
            delta_level = predict_ds >= self._get_data_index(
                level_i["index"], full_index=predict_ds
            )
            predictions += delta_level * level_i["c"]

        # Trend events
        for trend_i in self.model_output["trend_events"]:
            # Skip inactive events
            if not trend_i["is_predicted"]:
                continue
            if trend_i["is_damping"]:
                delta_trend = predict_ds > self._get_data_index(
                    trend_i["index"], full_index=predict_ds
                )
                trend_t = np.cumsum(delta_trend)
                predictions += (
                    delta_trend
                    * trend_i["m"]
                    * (1 - trend_i["gamma"] ** trend_t)
                    / (1 - trend_i["gamma"])
                )
            else:
                delta_trend = predict_ds >= self._get_data_index(
                    trend_i["index"], full_index=predict_ds
                )
                trend_t = np.cumsum(delta_trend) - (delta_trend > 0)
                predictions += delta_trend * trend_i["m"] * trend_t

        # Unspecified changepoints
        for cp_i in self.model_output["changepoint_events"]:
            delta_cp = predict_ds >= self._get_data_index(
                cp_i["index"], full_index=predict_ds
            )
            cp_t = np.cumsum(delta_cp) - (delta_cp > 0)
            predictions += delta_cp * cp_i["cp_c"] + delta_cp * cp_i["cp_m"] * cp_t

        # Seasonality
        seasonal_level = self._get_seasonal_level(horizon=horizon)
        if seasonal_level is not None:
            if self.fit_config["seasonal_type"] == "mult":
                predictions *= seasonal_level + np.ones(len(seasonal_level))
            else:
                predictions += seasonal_level
        predictions_rescaled = predictions * self.actuals_scale + self.actuals_loc
        output = pd.DataFrame(
            {
                "forecast": predictions_rescaled,
            },
            index=predict_ds,
        )
        output["y"] = pd.NA
        output.loc[self.fit_data.index, "y"] = self.fit_data.values
        output["insample"] = True
        output.loc[output.index > self.fit_data.index.max(), "insample"] = False
        logger.debug("Prediction output: %s", output)
        return output

    def compile(self):
        """
        Compile model from the STAN code and save as pickle in self.stan_build_path.

        This is only required if compilation was unsuccessful during installation.
        """
        with open(self.stan_file_path) as stannin:
            model_code = stannin.read()
        self.stan_model = pystan.StanModel(model_name=self.name, model_code=model_code)
        with open(self.stan_build_path, "wb") as stanbuildin:
            pickle.dump(self.stan_model, stanbuildin)

    def get_model_output(self, rescale: bool = True) -> Dict[str, Dict]:
        """
        Get model output, if 'rescale' is True, it will be rescaled back onto the original scale.
        The data is rescaled to have mean-0, var-1 so priors are relatively stable.
        But this means the learnt parameters will be for the rescaled data.
        This function rescales those parameters back onto the original scale (if rescale set)
        and returns them.

        If 'seasonal_type' was set to 'mult', then we are unable to rescale the parameters back
        as they are too tightly coupled. If this is the case, set 'rescale' to False to get
        unscaled parameters.
        ...
        :returns: model output dict that's on the original scale.
        ...
        :raises: NotImplementedError if 'seasonal_type' is 'mult' and 'rescale' is set to True.
        """
        model_output = copy.deepcopy(self.model_output)
        # Fetch the seasonal level while we have access to the fourier series
        model_output["base"]["seasonal_level"] = self._get_seasonal_level()
        if rescale is False:
            return model_output
        if self.fit_config["seasonal_type"] == "mult":
            raise NotImplementedError(
                "The 'seasonal_type' was set to 'mult' during fit."
                " Parameters too tightly coupled to rescale."
                " Set 'rescale' to False to return parameters on mean-0, var-1 scale."
            )
        # To rescale the model output, we absorb the constant into the bias beta_0.
        #   Everything else we multiply by scale.

        # Base level parameters
        model_output["base"]["beta_0"] = (
            model_output["base"]["beta_0"] * self.actuals_scale + self.actuals_loc
        )
        model_output["base"]["beta_1"] *= self.actuals_scale

        # Level events
        for level_i in model_output["level_events"]:
            level_i["c0"] *= self.actuals_scale
            level_i["c"] *= self.actuals_scale
        for trend_i in model_output["trend_events"]:
            trend_i["m0"] *= self.actuals_scale
            trend_i["m"] *= self.actuals_scale

        # Unspecified changepoints
        for cp_i in model_output["changepoint_events"]:
            cp_i["cp_c"] *= self.actuals_scale
            cp_i["cp_m"] *= self.actuals_scale

        # Seasonality
        if self.fit_config["seasonal_type"] == "add":
            if len(model_output["base"]["fourier_coefficients"]) > 0:
                model_output["base"]["fourier_coefficients"] *= self.actuals_scale
                model_output["base"]["seasonal_level"] *= self.actuals_scale
        return model_output

    def _prepare_event_data(
        self, level_events: List[Dict[str, Any]], trend_events: List[Dict[str, Any]]
    ):
        """
        Parse events, setting which should be learned, predicted and deactivated.

        This is all based on how much history there is.
            E.g.
            - if an event comes before any history, it should be deactivated.
            - if an event comes after any history, it should be predicted using initial estimate.

        :param level_events: the level_events input to fit
        :param trend_events: the trend_events input to fit
        """
        self._prepare_level_events(level_events)
        self._prepare_trend_events(trend_events)
        self._prepare_unspecified_changepoints()
        n_active_events = sum(
            [
                event_i["is_active"]
                for event_i in (self.level_events + self.trend_events)
            ]
        )
        utils.assert_log_warn(
            n_active_events > 0,
            (
                "No active trend or level events (i.e. no event indexes overlap with data)."
                " The model will just fit a base trend to the data."
            ),
        )
        logger.debug(
            "Finalised event settings:"
            "\n\t- level_events: %s"
            "\n\t- trend_events: %s"
            "\n\t- unspecified_changepoints: %s",
            self.level_events,
            self.trend_events,
            self.unspecified_changepoints,
        )

    def _prepare_stan_data(
        self,
    ):
        """
        Parse data and event data to prepare input to STAN

        ...
        :returns: dict of data for STAN model
        """
        stan_data: Dict[str, Any] = {}
        stan_data["T"] = len(self.fit_data)
        stan_data["y"] = self.rescaled_data
        stan_data["t"] = np.arange(stan_data["T"])

        # Level event parameters
        stan_data["L"] = len(
            [level_i for level_i in self.level_events if level_i["is_active"]]
        )
        stan_data["c0"] = [
            level_i["c0"] for level_i in self.level_events if level_i["is_active"]
        ]
        if stan_data["L"] == 0:
            stan_data["delta_level"] = [[]]
        else:
            stan_data["delta_level"] = [
                (self.fit_data.index >= self._get_data_index(level_i["index"])).astype(
                    int
                )
                for level_i in self.level_events
                if level_i["is_active"]
            ]

        # Trend event parameters
        stan_data["M"] = len(
            [trend_i for trend_i in self.trend_events if trend_i["is_active"]]
        )
        stan_data["m0"] = [
            trend_i["m0"] for trend_i in self.trend_events if trend_i["is_active"]
        ]

        # Need to force numpy type as int as otherwise causes error when passing empty list
        stan_data["is_damping"] = np.array(
            [
                int(trend_i["is_damping"])
                for trend_i in self.trend_events
                if trend_i["is_active"]
            ]
        ).astype(int)
        stan_data["gamma"] = [
            trend_i["gamma"] or 1.0
            for trend_i in self.trend_events
            if trend_i["is_active"]
        ]
        if stan_data["M"] == 0:
            stan_data["delta_trend"] = [[]]
            stan_data["trend_t"] = [[]]
        else:
            stan_data["delta_trend"] = []
            stan_data["trend_t"] = []
            for trend_i in self.trend_events:
                if not trend_i["is_active"]:
                    continue
                # If there's damping we need to handle edgecase when t=0 due to STAN formula
                delta_trend = self.fit_data.index > self._get_data_index(
                    trend_i["index"]
                )
                trend_t = np.cumsum(delta_trend)
                stan_data["delta_trend"].append(delta_trend.astype(int))
                stan_data["trend_t"].append(trend_t)

        # Unspecified changepoints
        stan_data["C"] = len(self.unspecified_changepoints)
        if stan_data["C"] == 0:
            stan_data["delta_cp"] = [[]]
            stan_data["cp_t"] = [[]]
        else:
            stan_data["delta_cp"] = [
                (self.fit_data.index >= changepoint_i["index"]).astype(int)
                for changepoint_i in self.unspecified_changepoints
            ]
            stan_data["cp_t"] = [
                np.cumsum(delta_trend) - (delta_trend > 0)
                for delta_trend in stan_data["delta_cp"]
            ]

        # Seasonality
        seasonal_period = self.fit_config["seasonal_period"] or 1
        if seasonal_period == 1:
            stan_data["is_seasonal"] = int(False)
            stan_data["seasonal_period"] = 1
            stan_data["fourier_order"] = 0
            stan_data["fourier_series"] = [[]]
        else:
            stan_data["is_seasonal"] = int(True)
            stan_data["seasonal_period"] = self.fit_config["seasonal_period"]
            fourier_order = self.fit_config["fourier_order"] or int(
                np.floor(seasonal_period / 2)
            )
            # STAN not aware we have the same number of sins as cos's, so multiply fourier order by 2.
            stan_data["fourier_order"] = fourier_order * 2
            fourier_series_to_validate = self._get_fourier_series()
            utils.assert_log_raise(
                isinstance(fourier_series_to_validate, list),
                "Problems generating fourier series.",
                ValueError,
            )
            stan_data["fourier_series"] = fourier_series_to_validate
        stan_data["is_multiplicative_seasonality"] = int(
            self.fit_config["seasonal_type"] == "mult"
        )

        # Scale hyperparameters
        stan_data["sigma_cp"] = self.fit_config["sigma_unspecified_changepoints"]
        stan_data["sigma_beta_0"] = self.fit_config["sigma_base_bias"]
        stan_data["sigma_beta_1"] = self.fit_config["sigma_base_trend"]
        stan_data["sigma_level"] = self.fit_config["sigma_level"]
        stan_data["sigma_trend"] = self.fit_config["sigma_trend"]
        stan_data["sigma_seasonal"] = self.fit_config["sigma_seasonal"]
        stan_data["sigma_actuals"] = self.fit_config["sigma_actuals"]
        logger.debug("Finalised STAN data: %s", stan_data)
        return stan_data

    def _set_stan_init(
        self,
        stan_data: Dict[str, Any],
    ):
        """
        Use stan_data set to set initial parameter values for optimizer

        :param stan_data: the stan data prepared by _prepare_stan_data
        ...
        :returns: dict of initial values for STAN parameters
        """
        stan_init = {}
        # Jitter has to be added to the starting values or LBGFS fails
        jitter_scale: float = self.fit_config["starting_jitter"]
        active_events = [
            self.rescaled_data[event_i["index"] :].index.min()
            for event_i in (self.level_events + self.trend_events)
            if event_i["is_active"]
        ]
        if active_events:
            first_event_index = np.min(active_events)
        # If there are no active_events, just use all the data!
        else:
            first_event_index = self.rescaled_data.index.max()

        pre_event_mask = (
            self.rescaled_data.index
            < self.rescaled_data.loc[first_event_index:].index.min()
        )
        pre_event_data = self.rescaled_data.loc[pre_event_mask]
        # Handle edge case where an event happens very soon
        if len(pre_event_data) < 4:
            stan_init["beta_0"] = pre_event_data.min() + np.random.normal(
                scale=jitter_scale
            )
            stan_init["beta_1"] = 0 + np.random.normal(scale=jitter_scale)
        else:
            pre_event_t = stan_data["t"][pre_event_mask]
            pre_event_estimates = linregress(pre_event_t, y=pre_event_data)
            stan_init["beta_1"] = pre_event_estimates.slope + np.random.normal(
                scale=jitter_scale
            )
            stan_init["beta_0"] = pre_event_estimates.intercept + np.random.normal(
                scale=jitter_scale
            )

        stan_init["c"] = stan_data["c0"] + np.random.normal(
            scale=jitter_scale, size=stan_data["L"]
        )
        stan_init["m"] = stan_data["m0"] + np.random.normal(
            scale=jitter_scale, size=stan_data["M"]
        )
        stan_init["cp_c"] = np.random.normal(scale=jitter_scale, size=stan_data["C"])
        stan_init["cp_m"] = np.random.normal(scale=jitter_scale, size=stan_data["C"])
        logger.debug("Starting points: %s", stan_init)
        return stan_init

    def _fit_data_checks(
        self, level_events: List[Dict[str, Any]], trend_events: List[Dict[str, Any]]
    ):
        """
        Check data input to fit

        :param level_events: the level events input to fit
        :param trend_events: the trend events input to fit
        ...
        :raises: ValueError if there are problems with the data
        """
        utils.assert_log_raise(
            isinstance(level_events, list),
            "'level_events' arg should be a list of dicts",
            ValueError,
        )
        utils.assert_log_raise(
            isinstance(trend_events, list),
            "'trend_events' arg should be a list of dicts",
            ValueError,
        )
        for level_i in level_events:
            utils.assert_log_raise(
                isinstance(level_i, dict),
                "'level_events' arg should be a list of dicts (this list does not have dict elements)",
                ValueError,
            )
            utils.assert_log_raise(
                {"index", "c0"}.issubset(set(level_i.keys())),
                "Each dict entry of 'level_events' requires keys 'index', 'c0'",
                ValueError,
            )
            utils.assert_log_raise(
                isinstance(level_i["c0"], (int, float)),
                "Level event initial level 'c0' must be a int or float",
                ValueError,
            )
        for trend_i in trend_events:
            utils.assert_log_raise(
                isinstance(trend_i, dict),
                "The 'trend_events' arg should be a list of dicts (this list does not have dict elements)",
                ValueError,
            )
            utils.assert_log_raise(
                {"index", "m0"}.issubset(set(trend_i.keys())),
                "Each dict entry of 'trend_events' requires keys 'index', 'm0' (and optionally 'gamma')",
                ValueError,
            )
            utils.assert_log_raise(
                isinstance(trend_i["m0"], (int, float)),
                "Trend event gradient 'm0' must be a int or float",
                ValueError,
            )
            if "gamma" in trend_i.keys():
                utils.assert_log_raise(
                    isinstance(trend_i["gamma"], (int, float)),
                    "Trend event damping 'gamma' must be float > 0 and <= 1",
                    ValueError,
                )
                # Replace gamma 1. with None
                utils.assert_log_raise(
                    trend_i["gamma"] <= 1 and trend_i["gamma"] > 0,
                    "Trend event damping 'gamma' must be float between 0 and 1 (non-inclusive)",
                    ValueError,
                )
        level_event_dates = [level_i["index"] for level_i in level_events]
        utils.assert_log_raise(
            len(set(level_event_dates)) == len(level_event_dates),
            "Level event indexes are not unique!",
            ValueError,
        )
        trend_event_dates = [trend_i["index"] for trend_i in trend_events]
        utils.assert_log_raise(
            len(set(trend_event_dates)) == len(trend_event_dates),
            "Trend event indexes are not unique!",
            ValueError,
        )
        event_dates = level_event_dates + trend_event_dates
        for event_date in event_dates:
            try:
                self.fit_data.loc[event_date:]
            except TypeError as error:
                raise TypeError(
                    "The 'index' entry in each level event must index the data pandas Series"
                    " (though they can be outside the index range).\n"
                    "Example 1:\n\tIf index of 'data' is range(5), i.e. standard range index;"
                    " valid 'date' entries are: 1, 3, 10, -3. Invalid are: '2021-10', dt.datetime(2020, 5, 1), 'str'."
                    "\nExample 2:\n\tIf index of 'data' is"
                    " pd.date_range(start='2020-01-31', end='2020-05-31', freq='M'), i.e. DatetimeIndex,"
                    " valid 'date' entries are pd.Timestamp objects and their shortened string accessors."
                    " E.g.: pd.Timestamp('2020-01-31'), '2020-01-31', etc.\n"
                    "For more information on pandas timestamp indexing, see\n"
                    "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html"
                ) from error
        # Check fit config params
        utils.assert_log_raise(
            self.fit_config["event_start_cutoff"] >= 0,
            "Arg event_start_cutoff must be an int greater than 0.",
            ValueError,
        )
        utils.assert_log_raise(
            isinstance(self.fit_config["event_start_cutoff"], int),
            "Arg event_start_cutoff must be an int greater than 0.",
            ValueError,
        )
        utils.assert_log_raise(
            self.fit_config["event_end_cutoff"] >= 0,
            "Arg event_end_cutoff must be an int greater than 0.",
            ValueError,
        )
        utils.assert_log_raise(
            isinstance(self.fit_config["event_end_cutoff"], int),
            "Arg event_end_cutoff must be an int greater than 0.",
            ValueError,
        )
        utils.assert_log_warn(
            self.fit_config["event_start_cutoff"] <= 6,
            "Arg event_start_cutoff greater than 6 indexes.",
        )
        utils.assert_log_warn(
            self.fit_config["event_end_cutoff"] <= 6,
            "Arg event_end_cutoff greater than 6 indexes.",
        )
        utils.assert_log_raise(
            (
                self.fit_config["seasonal_type"] == "add"
                or self.fit_config["seasonal_type"] == "mult"
            ),
            (
                "Arg 'seasonal_type' must be either 'add' (for additive seasonality)"
                "or 'mult' (for multiplicative seasonality)."
            ),
            ValueError,
        )
        if isinstance(self.fit_config["unspecified_changepoints"], int):
            utils.assert_log_raise(
                self.fit_config["unspecified_changepoints"] >= 0,
                (
                    "Arg 'unspecified_changepoints' must be an int greater than or equal to 0,"
                    " or an arraylike of indexes of data."
                ),
                ValueError,
            )
        else:
            try:
                list(self.fit_config["unspecified_changepoints"])
            except TypeError:
                utils.log_and_raise(
                    (
                        "Arg 'unspecified_changepoints' must be an int greater than or equal to 0,"
                        " or an arraylike of indexes of data."
                    ),
                    ValueError,
                )
            try:
                [
                    self.fit_data.loc[i]
                    for i in self.fit_config["unspecified_changepoints"]
                ]
            except KeyError:
                utils.log_and_raise(
                    (
                        "Arg 'unspecified_changepoints' must be an int greater than or equal to 0,"
                        " or an arraylike of indexes of data. Currently unspecified changepoints is an"
                        " arraylike that has an index not accessible by data.loc[.]."
                    ),
                    ValueError,
                )
        if self.fit_config["seasonal_period"] is not None:
            utils.assert_log_raise(
                isinstance(self.fit_config["seasonal_period"], int),
                "Arg seasonal_period must be an int greater than 0.",
                ValueError,
            )
            utils.assert_log_raise(
                self.fit_config["seasonal_period"] > 0,
                "Arg seasonal_period must be an int greater than 0.",
                ValueError,
            )
            if self.fit_config["fourier_order"] is None:
                seasonal_period = self.fit_config["seasonal_period"] or 1
                utils.assert_log_raise(
                    seasonal_period <= len(self.fit_data) / 2,
                    (
                        "Arg 'seasonal_period' is more than half the size of the data"
                        " but arg 'fourier_order' is unset."
                        " Set 'fourier_order' to less than half the size of the data."
                        f" Dataset size: {len(self.fit_data)},"
                        f" seasonal_period: {self.fit_config['seasonal_period']}"
                    ),
                    ValueError,
                )
        if self.fit_config["fourier_order"] is not None:
            utils.assert_log_raise(
                isinstance(self.fit_config["fourier_order"], int),
                "Arg 'fourier_order' must be an int euqal/greater than 1 and equal/less than floor(seasonal_period / 2).",
                ValueError,
            )
            utils.assert_log_raise(
                self.fit_config["seasonal_period"] is not None,
                (
                    "Arg 'seasonal_period' should not be None if 'fourier_order' is not None."
                ),
                ValueError,
            )
            utils.assert_log_raise(
                (
                    self.fit_config["fourier_order"] > 0
                    and self.fit_config["fourier_order"]
                    <= int(np.floor(self.fit_config["seasonal_period"] / 2))
                ),
                "Arg 'fourier_order' must be an int greater than 0 and less than floor(seasonal_period / 2).",
                ValueError,
            )
            seasonal_period = self.fit_config["seasonal_period"] or 1
            utils.assert_log_raise(
                self.fit_config["seasonal_period"] > 1,
                (
                    "If arg 'fourier_order' defined then 'seasonal_period' must not be None"
                    " and be greater than 1."
                ),
                ValueError,
            )
        sd_hyperparams = [k for k in self.fit_config.keys() if "sigma_" in k]
        for param in sd_hyperparams:
            utils.assert_log_raise(
                isinstance(self.fit_config[param], (int, float)),
                f"Arg {param} must be a float greater than 0.",
                ValueError,
            )
            utils.assert_log_raise(
                self.fit_config[param] > 0,
                f"Arg {param} must be a float greater than 0.",
                ValueError,
            )
        utils.assert_log_raise(
            isinstance(self.fit_config["seasonal_type"], str),
            "Arg 'seasonal_type' must be either 'add' or 'mult'.",
            ValueError,
        )
        utils.assert_log_raise(
            (
                (self.fit_config["seasonal_type"] == "add")
                or (self.fit_config["seasonal_type"] == "mult")
            ),
            "Arg 'seasonal_type' must be either 'add' or 'mult'.",
            ValueError,
        )

    def _load_built_model(self):
        """
        Load built model from pickle file if available.

        If the file is not available, the model is built from the STAN code and saved as pickle.
        ...
        :returns: built pystan model object.
        """
        if self.stan_build_path.exists():
            with open(self.stan_build_path, "rb") as stannin:
                return pickle.load(stannin)
        else:
            logger.warning(
                "Built stan model for '%s' not found." " Model building now...",
                self.name,
            )
            return self.compile()

    # Helpers
    def _prepare_level_events(
        self,
        level_events: List[Dict[str, Any]],
    ):
        """
        Prepare level events for creating STAN input

        :param level_events: level_events input to self.fit
        """
        self.level_events = []
        n_active_level_events = 0
        for level_i in level_events:
            event_index = level_i["index"]
            event_data = level_i.copy()
            # Rescale input
            event_data["c0"] = level_i["c0"] / self.actuals_scale
            num_pre_event_actuals = len(self.fit_data.loc[:event_index]) - 1
            num_post_event_actuals = len(self.fit_data.loc[event_index:])
            # Deactivate event if not enough prehistory.
            if num_pre_event_actuals <= self.fit_config["event_start_cutoff"]:
                logger.warning(
                    (
                        "Pre-event data for level event %s less than %s points."
                        " Event deactivated in model. Event index: %s, training data start index: %s"
                    ),
                    level_i["name"],
                    self.fit_config["event_start_cutoff"],
                    event_index,
                    self.fit_data.index.min(),
                )
                event_data["is_active"] = False
                event_data["is_predicted"] = False
                event_data["c"] = 0
                event_data["active_event_index"] = None
            # Setup event for prediction only if not enough posthistory.
            elif num_post_event_actuals <= self.fit_config["event_end_cutoff"]:
                logger.info(
                    (
                        "Post-event data for level event %s less than %s points."
                        " Event deactivated in model. Event index: %s, training data end index: %s"
                    ),
                    level_i["name"],
                    self.fit_config["event_end_cutoff"],
                    event_index,
                    self.fit_data.index.min(),
                )
                event_data["is_active"] = False
                event_data["is_predicted"] = True
                event_data["c"] = event_data["c0"]
                event_data["active_event_index"] = None
            else:
                logger.info(
                    (
                        "Adding level event %s to model."
                        " Event index: %s, training data start index: %s, training data end index: %s."
                        " Initial level: %s."
                    ),
                    level_i["name"],
                    event_index,
                    self.fit_data.index.min(),
                    self.fit_data.index.max(),
                    level_i["c0"],
                )
                event_data["is_active"] = True
                event_data["is_predicted"] = True
                event_data["c"] = None
                event_data["active_event_index"] = n_active_level_events
                n_active_level_events += 1
            self.level_events.append(event_data)

    def _prepare_trend_events(
        self,
        trend_events: List[Dict[str, Any]],
    ):
        """
        Prepare trend events for creating STAN input

        :param trend_events: trend_events input to self.fit
        """
        self.trend_events = []
        n_active_trend_events = 0
        for trend_i in trend_events:
            event_index = trend_i["index"]
            event_data = trend_i.copy()
            # Rescale input
            event_data["m0"] = trend_i["m0"] / self.actuals_scale
            num_pre_event_actuals = len(self.fit_data.loc[:event_index])
            num_post_event_actuals = len(self.fit_data.loc[event_index:])
            # Deactivate event if not enough prehistory.
            if num_pre_event_actuals < self.fit_config["event_start_cutoff"]:
                logger.warning(
                    (
                        "Pre-event data for trend event %s less than %s points."
                        " Event deactivated in model. Event index: %s, training data start index: %s"
                    ),
                    trend_i["index"],
                    self.fit_config["event_start_cutoff"],
                    event_index,
                    self.fit_data.index.min(),
                )
                event_data["is_active"] = False
                event_data["is_predicted"] = False
                event_data["m"] = 0
                event_data["gamma"] = 1.0
                event_data["active_event_index"] = None
            # Setup event for prediction only if not enough posthistory.
            elif num_post_event_actuals <= self.fit_config["event_end_cutoff"]:
                logger.warning(
                    (
                        "Post-event data for trend event %s less than %s points."
                        " Event deactivated in model. Event index: %s, training data end index: %s"
                    ),
                    trend_i["name"],
                    self.fit_config["event_end_cutoff"],
                    event_index,
                    self.fit_data.index.min(),
                )
                event_data["is_active"] = False
                event_data["is_predicted"] = True
                # Figure out if damping required or not.
                if trend_i.get("gamma") is None:
                    event_data["is_damping"] = False
                    event_data["gamma"] = None
                elif np.isclose(trend_i.get("gamma"), 1.0):
                    event_data["is_damping"] = False
                    event_data["gamma"] = None
                else:
                    event_data["is_damping"] = True
                    event_data["gamma"] = trend_i["gamma"]
                event_data["m"] = event_data["m0"]
                event_data["active_event_index"] = None
            else:
                logger.info(
                    (
                        "Adding trend event %s to model."
                        " Event index: %s, training data start index: %s, training data end index: %s."
                        " Initial gradient: %s. Damping: %s."
                    ),
                    trend_i["name"],
                    event_index,
                    self.fit_data.index.min(),
                    self.fit_data.index.max(),
                    trend_i["m0"],
                    trend_i.get("gamma"),
                )
                event_data["is_active"] = True
                event_data["is_predicted"] = True
                event_data["m"] = None
                # Figure out if damping required or not.
                if trend_i.get("gamma") is None:
                    event_data["is_damping"] = False
                    event_data["gamma"] = None
                elif np.isclose(trend_i.get("gamma"), 1.0):
                    event_data["is_damping"] = False
                    event_data["gamma"] = None
                else:
                    event_data["is_damping"] = True
                    event_data["gamma"] = trend_i["gamma"]
                event_data["active_event_index"] = n_active_trend_events
                n_active_trend_events += 1
            self.trend_events.append(event_data)

    def _prepare_unspecified_changepoints(self):
        """
        Prepare unspecified changepoints for creating STAN input
        """
        self.unspecified_changepoints = []
        changepoint_indexes = self._get_unspecified_changepoint_indexes()
        event_indexes = set(
            [
                self._get_data_index(event_i["index"])
                for event_i in (self.level_events + self.trend_events)
            ]
        )
        active_event_index = 0
        for index in changepoint_indexes:
            # Ensure changepoints don't overlap with preexisting events
            if index in event_indexes:
                utils.log_and_warn(
                    f"Unspecified changepoint with index {index} also specified as a level or trend event."
                    " Removing this changepoint."
                )
                continue
            self.unspecified_changepoints.append(
                {
                    "index": index,
                    "cp_c0": 0.0,
                    "cp_m0": 0.0,
                    "active_event_index": active_event_index,
                }
            )
            active_event_index += 1

    def _get_data_index(
        self, index: Any, full_index: Optional[Type[pd.Index]] = None
    ) -> Any:
        """
        Fetch the index in the correct format from fit_data

        Certain pandas index formats accept a variety of types as  e.g. "2020-01"
        for pandas Timestamp index.
        This method takes the index in its raw form and returns the actual pandas
        index it refers to (it assumes we know that index exists in the data).

        :param index: the index to access
        :param full_index: If this is None, index is taken from self.fit_data.
            But for prediction the index will be different from this as it
            will have the added prediction horizon indexes.
        ...
        :return: index in correct pandas format
        """
        if full_index is None:
            return self.fit_data.loc[index:].index.min()
        else:
            # Seems you need to put the index into a pandas series to get the full indexing benefits
            _tmp_series = pd.Series(np.arange(len(full_index)), index=full_index)
            return _tmp_series.loc[index:].index.min()

    def _get_unspecified_changepoint_indexes(self):
        """
        Select locations for the unspecified changepoints, and remove if they collide with
        level or trend events.
        """
        changepoints = self.fit_config["unspecified_changepoints"]
        # If changepoints is not an int, then the checks should have ascertained
        #  it is an iterable which indexes self.fit_data. Therefore we can return as is.
        if not isinstance(changepoints, int):
            return [self._get_data_index(cp) for cp in changepoints]
        # Avoid div by 0 error
        if changepoints == 0:
            return []
        T = len(self.fit_data)
        # Ensure changepoints don't go right out to the edges
        edge_gap = np.ceil(T / changepoints).astype(int)
        # Find equally spaced indexes along data
        cp_start = np.max((edge_gap, self.fit_config["event_start_cutoff"])) - 1
        cp_end = T - 1 - np.max((edge_gap, self.fit_config["event_end_cutoff"]))
        changepoint_indexes_float = np.linspace(
            start=cp_start, stop=cp_end, num=changepoints
        )
        changepoint_indexes_int = np.floor(changepoint_indexes_float).astype(int)
        changepoint_indexes = self.fit_data.index[np.unique(changepoint_indexes_int)]
        return changepoint_indexes

    def _prepare_data(self, data: pd.Series):
        """
        Prepare data and perform input checks

        :param data: data input to fit method
        ...
        :returns: prepared data
        """
        logger.debug("Raw input data: %s", data)
        # Ensure actuals data is a pd.Series of floats
        utils.assert_log_raise(
            isinstance(data, pd.Series),
            "Arg 'data' should be a numeric pandas Series",
            ValueError,
        )
        try:
            self.fit_data = data.astype(np.float64)
        except ValueError:
            utils.log_and_raise(
                "Arg 'data' should be a numeric pandas Series. Unable to coerce current input to float",
                ValueError,
            )
        utils.assert_log_raise(
            len(self.fit_data) > 0,
            "Arg 'data' has no data points (it is empty)",
            ValueError,
        )
        utils.assert_log_warn(
            len(self.fit_data) > 5,
            "Arg 'data' has less than 5 data points. Expect a poor fit.",
        )
        accepted_indexes = [
            pd.RangeIndex,
            pd.Int64Index,
            pd.UInt64Index,
            pd.DatetimeIndex,
            pd.PeriodIndex,
            pd.Timedelta,
        ]
        utils.assert_log_raise(
            isinstance(data.index, tuple(accepted_indexes)),
            f"Arg 'data' must have index of type: {', '.join([str(_index) for _index in accepted_indexes])}.",
            ValueError,
        )
        utils.assert_log_raise(
            not any(pd.isna(data)), "Arg 'data' has NA values.", ValueError
        )
        freq_indexes = [pd.DatetimeIndex, pd.PeriodIndex, pd.Timedelta]
        if isinstance(data.index, tuple(freq_indexes)):
            try:
                data.index.shift(periods=1)
            except pd.errors.NullFrequencyError:
                utils.log_and_raise(
                    (
                        f" Arg 'data' with indexes of type {', '.join([str(_index) for _index in freq_indexes])}"
                        " must have 'freq' set, which determines the regularity of the index."
                        " e.g. data.index.asfreq('M') for monthly. This is used to calculate the"
                        " horizon during prediction."
                    ),
                    ValueError,
                )
        utils.assert_log_raise(
            set(data.index.sort_values()) == set(data.index),
            "Data index must be sorted and have no gaps",
            ValueError,
        )

        # Rescale data so it's on a similar scale and hyperparameter estimates are valid
        if self.fit_config["seasonal_type"] == "mult":
            self.actuals_loc = np.min(data.values)
            logger.debug(
                "Multiplicative data. Rescaling by data min: %s", self.actuals_loc
            )
        else:
            self.actuals_loc = np.mean(data.values)
            logger.debug("Additive data. Rescaling by data mean: %s", self.actuals_loc)
        data_scale = np.std(data.values)
        if data_scale > 1e-6:
            logger.debug("Data scale > 1e-6. Using non-1 scale: %s", data_scale)
            self.actuals_scale = data_scale
        else:
            logger.debug(
                "Data scale '%s' too close to 0. Keeping scale as default: %s.",
                data_scale,
                self.actuals_scale,
            )
        self.rescaled_data = (data - self.actuals_loc) / self.actuals_scale
        logger.debug("Rescaled data: %s", self.rescaled_data)
        return data

    def _get_fourier_series(self, horizon: int = 0) -> Optional[List[np.ndarray]]:
        """
        Get the fourier series for modelling seasonality.

        :param horizon: when making predictions, we will need to extend the fourier
            series beyond the train set. The horizon sets how much further we need to go.
        ...
        :returns: list of size 'fourier_order' with arrays of coefficients.
            If 'fourier_order' unset, set fourier_order = seasonal_period - 1.
        """
        seasonal_period = self.fit_config["seasonal_period"] or 1
        if seasonal_period == 1:
            return None
        fourier_order: int = self.fit_config["fourier_order"] or int(
            np.floor(seasonal_period / 2)
        )
        T = len(self.fit_data) + horizon
        t = np.arange(1, T + 1)
        fourier_coefficients = []
        for j in range(1, fourier_order + 1):
            fourier_coefficients.append(np.cos(2 * j * np.pi * t / seasonal_period))
            fourier_coefficients.append(np.sin(2 * j * np.pi * t / seasonal_period))
        return fourier_coefficients

    def _get_seasonal_level(self, horizon: int = 0):
        """
        Fetch the seasonal levels of the timeseries, for returning model parameters
        """
        seasonal_period = self.fit_config["seasonal_period"] or 1
        if seasonal_period <= 1:
            return None
        fourier_series = self._get_fourier_series(horizon=horizon)
        fourier_coefficients = self.model_output["base"]["fourier_coefficients"]
        seasonal_level = np.zeros(len(self.fit_data) + horizon)
        fourier_order = fourier_coefficients.size
        if fourier_order > 1:
            for j in range(fourier_order):
                seasonal_level += fourier_coefficients[j] * fourier_series[j]
        else:
            seasonal_level += fourier_coefficients * fourier_series[0]
        return seasonal_level
