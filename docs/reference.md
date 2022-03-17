# `judgyprophet` API Reference

# `judgyprophet.JudgyProphet`

```
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
```


## `JudgyProphet.fit`


```
def fit(
    self,
    data: pd.Series,
    level_events: List[Dict[str, Any]],
    trend_events: List[Dict[str, Any]],
    unspecified_changepoints: Union[int, Iterable] = 0,
    seasonal_period: Optional[int] = None,
    seasonal_type: str = 'add',
    event_start_cutoff: int = 4,
    event_end_cutoff: int = 0,
    fourier_order: Optional[int] = None,
    sigma_unspecified_changepoints: float = .1,
    sigma_base_bias: float = 1.,
    sigma_base_trend: float = 1.,
    sigma_level: float = .1,
    sigma_trend: float = .1,
    sigma_seasonal: float = 1.,
    sigma_actuals: float = .5,
    starting_jitter: float = .1,
    **kwargs
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
    :param unspecified_changepoints: (optional, default 0) The number of 'prophet' like
        changepoints (i.e. learnt from the data as there has been an unanticipated change
        to the timeseries trend or level). Either an int of the number of changepoints which are
        distributed evenly along the data, or an ArrayLike of indexes of the data where
        changepoints are required. If it is an ArrayLike, each element should be the same type
        as the actuals index.
    :param seasonal_period: (optional, default None) The seasonal period of the timeseries.
        This should be an integer. If this is set to None or 1,
        no seasonality will be used. Seasonality is modelled using Fourier series.
    :param seasonal_type: (optional, default 'add') the type of seasonality to use:
        - 'add' for additive
        - 'mult' for multiplicative
    :param event_start_cutoff: the number of data points of history required before an event
        before a changepoint will be put there (otherwise it is absorbed into the base trend
        and will not be learned).
    :param event_end_cutoff: the number of data points required after an event before
        a changepoint will be put there and its parameters will be learnt by the model
        (otherwise it is predicted using m0).
    :param fourier_order: (optional, int) the number of fourier terms to use for modelling
        seasonality. If None, this will be set to seasonal_period - 1.
        This is useful if you have a long seasonal period and not much data. You can use
        a fourier order that is lower than the seasonal period.
    :param sigma_unspecified_changepoints: The standard deviation of the prior on the unspecified
        changepoints. The smaller this is, the higher the L1 penalty on the changepoints.
    :param sigma_base_bias: The standard deviation of the prior on the base bias
        (i.e. bias before any changepoints or events are observed). The smaller this is,
        the higher the regularisation on the parameter.
    :param sigma_base_trend: The standard deviation of the prior on the base trend
        (i.e. trend before any changepoints or events are observed). The smaller this is,
        the higher the regularisation on the parameter.
    :param sigma_level: The standard deviation of the prior on level events.
        The smaller this is, the more the parameter will be regularised towards the initial estimate 'c0'.
    :param sigma_trend: The standard deviation of the prior on trend events.
        The smaller this is, the more the parameter will be regularised towards the initial estimate 'm0'.
    :param sigma_seasonal: The standard deviation of the prior on seasonality.
        The smaller this is, the more the parameter will be regularised towards 0.
    :param sigma_actuals: The standard deviation of the prior on actuals noise.
    :param starting_jitter: jitter in the starting values -- resolves optimisation problems.
    :param **kwargs: Any extra keyword arguments will be passed onto StanModel.sampling (e.g. iter, chains)
    """
```


## `JudgyProphet.predict`

```
def predict(self, horizon: int = 0) -> pd.DataFrame:
    """
    Predict model up to the supplied horizon. If no horizon is given, the insample fit is returned.

    :param horizon: (Optional, default 0) the horizon to predict upto. Must be a positive integer.
    ...
    :returns: a dataframe of prediction results, including a column indicating if the point is insample or not.
    """
```


## `JudgyProphet.compile`
```
def compile(self):
    """
    Compile model from the STAN code and save as pickle in self.stan_build_path.

    This is only required if compilation was unsuccessful during installation.
    """
```


## `JudgyProphet.get_model_output`


```
def get_model_output(
        self,
        rescale: bool = True
    ) -> Dict[str, Dict]:
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
```
