# How-to Tutorials

The tutorial section is a more detailed explanation of the JudgyProphet
functionality. It discusses how to model level events, trend events, and
seasonality. It also gives a brief overview of hyperparameters settings.
The tutorial is based on the jupyter notebools in the `tutorial` folder.

## Level Event

In this tutorial we use `judgyprophet` to forecast a time series with
what we call a `level_event`. It is a sudden change in level of the time
series without an underlying trend change. An example would be the
following:

``` python
from judgyprophet.tutorials.resources import get_level_event
example_data = get_level_event()
example_data.plot.line()
```

    <AxesSubplot:>

![png](T1%20-%20Level%20Events_files/T1%20-%20Level%20Events_1_1.png)

We can see a relatively stable constant trajectory, followed by a shift
in that trajectory around April 2020. It is quite stable after that. The
example above also shows the format of the data required by
`judgyprophet`. The data should be a `pandas Series`, with the actuals
as the entries, e.g.:

    2019-01-01    3.287609
    2019-02-01    4.753766
    2019-03-01    3.955497
    2019-04-01    4.451812
    2019-05-01    5.345102
    Freq: MS, dtype: float64

The index should denote the datetime element of the series, it should be
ordered and have no gaps. It can either be a pandas index for
specifically working with time series (e.g. `pd.DatetimeIndex`), or just
an integer based index – this means you don’t have to explicitly list
dates. If it is a pandas time series index, the `freq` should be set.
This allows `judgyprophet` to calculate the horizon during prediction.
In our case the `freq` is set to be ‘MS’, meaning month start.

#### Format the level event expectation for JudgyProphet

Suppose that we are aware in January 2020 that an event is likely to
happen in April 2020 which will change the level by approximately 10. We
would encode this in `judgyprophet` as follows:

``` python
level_events = [
    {'name': 'Expected event 1', 'index': '2020-04-01', 'c0': 10}
]
```

Each level event is encoded as a `dict` with two required entries: the
`'index'` field, which is the index in the data when the event occurs.
If this entry is fed into `example_data.loc[]`, then it should return a
single value. It follows the standard `pandas` indexing rules (for
example, see
[here](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)).
The `'c0'` field is the initial estimate by the business of what the
impact of this level event will be. It is fed into the model as an
informative prior on the level event mean; which is then updated in a
Bayesian way.

#### Forecasting with JudgyProphet before the event occurs

Now let’s pretend we’re still in January 2020, and see what
`judgyprophet` would have forecasted:

``` python
from judgyprophet import JudgyProphet

# Cutoff the data to January 2020
data_jan2020 = example_data.loc[:"2020-01-01"]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_jan2020,
    level_events=level_events,
    trend_events=[],
    sigma_base_bias=.1,
    sigma_base_trend=.1,
    unspecified_changepoints=0,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Post-event data for level event Expected event 1 less than 0 points. Event deactivated in model. Event index: 2020-04-01, training data end index: 2019-01-01 00:00:00
    WARNING:judgyprophet.utils:No active trend or level events (i.e. no event indexes overlap with data). The model will just fit a base trend to the data.


    Initial log joint probability = -56.9093
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
           3      -25.5963      0.272471   1.08052e-14           1           1        6
    Optimization terminated normally:
      Convergence detected: gradient norm is below tolerance

Let’s plot the results…

``` python
import pandas as pd
import seaborn as sns

predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_jan2020.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc["2020-02-01":]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T1%20-%20Level%20Events_files/T1%20-%20Level%20Events_9_1.png)

We can see from the plot that the forecast captures the event pretty
well. However the business estimate of the change-in-level event is
probably slightly too high; which leads to the forecast to slightly
overshoot the actuals.

#### Forecasting with JudgyProphet after the event occurs

This is where the Bayesian updating comes into its own. Let’s now fit
the forecast after the event has occurred. At this point, the impact of
the event will be updated in a Bayesian way given what has been seen in
the actuals.

``` python
# Cutoff the data to June 2020
data_june2020 = example_data.loc[:"2020-06-01"]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_june2020,
    level_events=level_events,
    trend_events=[],
    sigma_base_bias=1.,
    sigma_base_trend=1.,
    unspecified_changepoints=0,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding level event Expected event 1 to model. Event index: 2020-04-01, training data start index: 2019-01-01 00:00:00, training data end index: 2020-06-01 00:00:00. Initial level: 10.


    Initial log joint probability = -17.0116
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
           6      -2.93197   1.93616e-05   3.23115e-06           1           1        9
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance

Plotting the results again:

``` python
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_june2020.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc["2020-07-01":]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T1%20-%20Level%20Events_files/T1%20-%20Level%20Events_14_1.png)

When `judgyprophet` predicts after the event occurs, it decreases the
business estimate as it observes actuals.

## Trend Events

In this tutorial we use `judgyprophet` to forecast a time series with a
known/expected change in trend. This is called a `trend_event` and an
example would be the following:

``` python
from judgyprophet.tutorials.resources import get_trend_event

example_data = get_trend_event()
example_data.plot.line()
```

    <AxesSubplot:>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_2_1.png)

We can see from the plot that there is an uptick in trend around
September 2020.

#### Format the trend event expectation for JudgyProphet

Suppose we are in April 2020, and we have the prior knowledge that an
event will occur in September. The expected change of the trend gradient
will be an uptick of 6. We can encode this in `judgyprophet` as an
expected trend event as follows:

``` python
trend_events = [
    {'name': "New market entry", 'index': '2020-09-01', 'm0': 6}
]
```

Each trend event is encoded as a dict with two required entries: the
‘index’ field, which is the index in the data when the event occurs. If
this entry is fed into example_data.loc\[\], then it should return a
single value. It follows the standard pandas indexing rules (for
example, see here). The ‘m0’ field is the initial estimate by the
business of what the impact of this event on the trend will be (e.g. in
our case we estimate it will increase the trend by 6). It is fed into
the model as an informative prior on the level event mean; which is then
updated in a Bayesian way.

#### Forecasting with JudgyProphet before the event occurs

Now let’s pretend we’re still in April 2020, and see what `judgyprophet`
would have forecasted:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

# Cutoff the data to January 2020
data_apr2020 = example_data.loc[:"2020-04-01"]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_apr2020,
    level_events=[],
    trend_events=trend_events,
    sigma_base_bias=1.,
    sigma_base_trend=1.,
    unspecified_changepoints=0,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    WARNING:judgyprophet.judgyprophet:Post-event data for trend event New market entry less than 0 points. Event deactivated in model. Event index: 2020-09-01, training data end index: 2019-06-01 00:00:00
    WARNING:judgyprophet.utils:No active trend or level events (i.e. no event indexes overlap with data). The model will just fit a base trend to the data.


    Initial log joint probability = -17.6559
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
           5      -2.70162    0.00625172   1.10466e-05           1           1        8
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance

Plotting the results:

``` python
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_apr2020.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc["2020-05-01":]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_10_1.png)

#### Forecasting with JudgyProphet after the event occurs

We can see that with the business information, the forecast is better
able to handle the sudden uptick in trend. After a while though, we can
see that the business estimate of the trend is an overestimate. Let’s
see if the Bayesian updating can account for this. We now suppose we are
re-forecasting the product in January 2021.

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

# Cutoff the data to January 2020
data_jan2021 = example_data.loc[:"2021-01-01"]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_jan2021,
    level_events=[],
    trend_events=trend_events,
    unspecified_changepoints=0,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New market entry to model. Event index: 2020-09-01, training data start index: 2019-06-01 00:00:00, training data end index: 2021-01-01 00:00:00. Initial gradient: 6. Damping: None.


    Initial log joint probability = -18.4007
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
           7      -1.64341   1.13818e-05   8.53133e-05           1           1       10
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance

``` python
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[predictions.index <= "2021-06-01", ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_apr2020.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc["2020-05-01":]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_13_1.png)

Although the forecast is still a slight overestimate, the Bayesian
updating has downgraded the initial estimate somewhat.

### Trend event reducing the historic trend

A trend event can also reduce the previous trend. A common real world
examples for this case are a competitor entering the market and thus
over time reducing your products market share.

``` python
example_data = get_trend_event(uptake=False)
example_data.plot.line()
```

    <AxesSubplot:>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_16_1.png)

#### Format the trend event expectation for JudgyProphet

In this case, we simply add a minus to the expected trend impact:

``` python
trend_events = [
    {'name': "New market entry", 'index': '2020-09-01', 'm0': -6}
]
```

#### Forecasting with JudgyProphet before the event occurs

Again, we’re still in April 2020, and see what `judgyprophet` would have
forecasted:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

# Cutoff the data to January 2020
data_apr2020 = example_data.loc[:"2020-04-01"]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_apr2020,
    level_events=[],
    trend_events=trend_events,
    sigma_base_bias=1.,
    sigma_base_trend=1.,
    unspecified_changepoints=0,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    WARNING:judgyprophet.judgyprophet:Post-event data for trend event New market entry less than 0 points. Event deactivated in model. Event index: 2020-09-01, training data end index: 2019-06-01 00:00:00
    WARNING:judgyprophet.utils:No active trend or level events (i.e. no event indexes overlap with data). The model will just fit a base trend to the data.


    Initial log joint probability = -3.20978
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
           7      -2.70162   5.03247e-05   6.65978e-05           1           1        9
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance

Plotting the results:

``` python
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_apr2020.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc["2020-05-01":]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_22_1.png)

#### Forecasting with JudgyProphet after the event occurs

And after the event, we again learn from the additional data points:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

# Cutoff the data to January 2020
data_jan2021 = example_data.loc[:"2021-01-01"]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_jan2021,
    level_events=[],
    trend_events=trend_events,
    unspecified_changepoints=0,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New market entry to model. Event index: 2020-09-01, training data start index: 2019-06-01 00:00:00, training data end index: 2021-01-01 00:00:00. Initial gradient: -6. Damping: None.


    Initial log joint probability = -28.0997
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
           6      -7.46664   0.000740991   0.000446113           1           1        9
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance

``` python
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[predictions.index <= "2021-06-01", ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_apr2020.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc["2020-05-01":]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_25_1.png)

### Damping

In real world examples, we often observe that the initial trend after
the event changes over time. A good example is the total sales of a
product if this product is released to an additional new market. In this
scenario, there is usually a period of strong uptake initially, followed
by a slowing of uptake as the market saturates. We model this in
`judgyprophet` by using damping. The damping model is a linear trend
with damping, which is the same as that used in one of the most popular
exponential smoothing methods: Holt’s linear damped trend (see
[here](https://otexts.com/fpp3/holt.html#damped-trend-methods)).

Unlike the trend and level parameters, the damping is set by the user,
and is not learnt during fitting. This is because we found learning
using Bayesian fitting to be inaccurate. If you are not sure what the
damping term should be we recommend using cross-validation, or observing
similar market launches. The damping term is usually between 0.8 and 1
(where 1 means no damping, i.e. a linear model), it is equivalent to the
$\phi$ parameter in the description of Holt’s linear damped trend.

Those who know Prophet will remember it modelled this using logistic
curves. We found this was extremely sensitive to the choice of the
capacity parameter $C$ (the population the new entrant would eventually
reach). This is why we opted for the damped linear trend, which we found
to be more flexible.

Let’s look at a curve with damping:

``` python
from judgyprophet.tutorials.resources import get_damped_trend_event

example_data = get_damped_trend_event()
print(example_data.head())
example_data.plot.line()
```

    2019-06-01    3.287609
    2019-07-01    5.253766
    2019-08-01    4.955497
    2019-09-01    5.951812
    2019-10-01    7.345102
    Freq: MS, dtype: float64





    <AxesSubplot:>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_27_2.png)

We can see there is an initial uptick in trend, followed by a plateauing
effect as the market saturates.

#### Format the damped trend event expectation for JudgyProphet

We talk to the business and they assume the initial trend uptake is 5,
with a damping parameter from analysing similar market entrants of .9.
We encode this as a `trend_event` as follows, notice that we refer to
the damping parameter as `gamma`:

``` python
trend_events = [
    {'name': "New saturating market entry", 'index': '2020-01-01', 'm0': 5, 'gamma': .85}
]
```

#### Forecasting with JudgyProphet before the event occurs

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

# Cutoff the data to January 2020
data_oct2019 = example_data.loc[:"2019-10-01"]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_oct2019,
    level_events=[],
    trend_events=trend_events,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_oct2019.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc["2019-10-01":]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    WARNING:judgyprophet.utils:Arg 'data' has less than 5 data points. Expect a poor fit.
    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    WARNING:judgyprophet.judgyprophet:Post-event data for trend event New saturating market entry less than 0 points. Event deactivated in model. Event index: 2020-01-01, training data end index: 2019-06-01 00:00:00
    WARNING:judgyprophet.utils:No active trend or level events (i.e. no event indexes overlap with data). The model will just fit a base trend to the data.


    Initial log joint probability = -4.75577
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
           4      -2.08579     0.0190169   7.99064e-06           1           1        6
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_31_3.png)

We can see the initial estimates are quite off this time. This is
because the business overestimated the damping and the trend. Let’s see
what happens as we start to observe actuals.

#### Forecasting with JudgyProphet after the event occurs

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

# Cutoff the data to January 2020
cutoff = "2020-04-01"
data_cutoff = example_data.loc[:cutoff]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_cutoff,
    level_events=[],
    trend_events=trend_events,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New saturating market entry to model. Event index: 2020-01-01, training data start index: 2019-06-01 00:00:00, training data end index: 2020-04-01 00:00:00. Initial gradient: 5. Damping: 0.85.


    Initial log joint probability = -10.4473
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
           9      -2.49061   8.76664e-05    0.00143778      0.9171      0.9171       11
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_34_3.png)

While the trend is still a little overestimated, it is definitely
improving. But is there anything we can do to improve the situation?

### Unspecified Changepoints

Like prophet, we enable the user to include unspecified changepoints
into the forecast. Unexpected changepoints enables us to handle
unexpected changes in the trend of level of the time series. This is
done by setting the `unspecified_changepoints` parameter in the `fit`
method of `JudgyProphet`. Unspecified changepoints are initially set to
have no effect on the model, but if actuals are observed that deviate
from the model, they will be ‘turned on,’ and change the model.

The arg `unspecified_changepoints` either takes an integer as input,
which will intersperse the changepoints equally across time, or a list
of indexes, which will place changepoints at exactly those time points.
These changepoints are given a Laplace prior with a mean of 0, and a
scale set by the arg `sigma_unspecified_changepoints`. The
`sigma_unspecified_changepoints` affects the L1 penalty on the
unspecified changepoints. Set `sigma` to be high (e.g. .5 or greater)
and the unspecified changepoints will be very sensitive. Set it low
(e.g. 0.05), and it will be insensitive.

Unlike prophet, setting these changepoints does not change the
prediction. Only `level_events` and `trend_events` in the prediction
horizon will affect the forecast. Let’s see what affect these have on
the previous damping example…

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

# Cutoff the data to January 2020
cutoff = "2020-04-01"
data_cutoff = example_data.loc[:cutoff]

# Make the forecast with the business estimated level event
# We have no trend events, so just provide the empty list.
jp = JudgyProphet()
# Because the event is beyond the actuals, judgyprophet throws a warning.
#    This is just because the Bayesian model at the event has no actuals to learn from.
#    The event is still used in predictions.
jp.fit(
    data=data_cutoff,
    level_events=[],
    unspecified_changepoints=10,
    sigma_unspecified_changepoints=.2,
    trend_events=trend_events,
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=12)

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New saturating market entry to model. Event index: 2020-01-01, training data start index: 2019-06-01 00:00:00, training data end index: 2020-04-01 00:00:00. Initial gradient: 5. Damping: 0.85.
    WARNING:judgyprophet.utils:Unspecified changepoint with index 2020-01-01 00:00:00 also specified as a level or trend event. Removing this changepoint.


    Initial log joint probability = -85.6424
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          12      -2.55807    0.00253247       10.5928   0.0002553       0.001       57  LS failed, Hessian reset
          19      -2.53784    0.00129929       12.9114       2.828      0.4174       69
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          39      -2.49159   1.88991e-05       13.7481     0.05264           1      106
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          43      -2.49059   2.07785e-05        13.316   1.762e-06       0.001      150  LS failed, Hessian reset
          49      -2.48998    7.5683e-06       12.3042   6.778e-07       0.001      193  LS failed, Hessian reset
          59      -2.48991   3.40412e-07       12.2862      0.3482      0.3482      210
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          70       -2.4899   8.88435e-09       11.2343   6.928e-10       0.001      271  LS failed, Hessian reset
    Optimization terminated normally:
      Convergence detected: absolute parameter change was below tolerance


    /Users/kpxh622/github/judgyprophet/judgyprophet/utils.py:31: UserWarning: Unspecified changepoint with index 2020-01-01 00:00:00 also specified as a level or trend event. Removing this changepoint.
      warnings.warn(msg)





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T2%20-%20Trend%20Events_files/T2%20-%20Trend%20Events_37_4.png)

We can see in this case the unspecified changepoints have improved the
model fit significantly. We recommend caution with unspecified
changepoints though, they come with a cost, and when your time series is
quite noisy they might be overreactive. In this case it is recommended
to tune your `sigma_unspecified_changepoints` accordingly, or limit the
amount of changepoints you use.

## Seasonality

Similar to `prophet`, `judgyprophet` models seasonality as Fourier
series and can handle both additive and multiplicative seasonality.
However, the seasonality implementation is currently limited to the
index frequency and does not support the split into weekly, monthly, and
yearly.

To enable seasonality, simply set the `seasonal_period` arg to a
positive integer (e.g. 12 for monthly data, 7 for daily). The default
seasonality is additive, to change this to multiplicative set the arg
`seasonal_type` to be `'mult'`. The Fourier order is set via the
`fourier_order` parameter, with the default value set to
`seasonal_period` - 1. The Fourier order determines how quickly the
seasonality can change and the reducing order compared to the default
parameters might help to avoid overfitting.

### Additive Seasonality

In the case of additive seasonality, the amplitude of the seasonal
variation is independent of the trend and is hence roughly constant over
the time series. If additive seasonality is selected, `judgyprophet`
will rescale the time series onto zero mean and standard variance. An
example is shown here:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns
from judgyprophet.tutorials.resources import get_additive_seasonality_linear_trend

example_data = get_additive_seasonality_linear_trend()


# Cutoff the data to October 2020
cutoff = "2020-10-01"
data_cutoff = example_data.loc[:cutoff]

jp = JudgyProphet()
# We are passing in a simple time series without trend or level events. The seasonality is set to 12
#     and the seasonality component is simply additive.
jp.fit(
    data=data_cutoff,
    level_events=[],
    trend_events=[],
    seasonal_period=12,
    seasonal_type="add",
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=int(12))

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    WARNING:judgyprophet.utils:No active trend or level events (i.e. no event indexes overlap with data). The model will just fit a base trend to the data.


    Initial log joint probability = -891.479
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          19      -1.34039   1.12595e-05    0.00168501      0.5461      0.5461       30
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          21      -1.34039   1.01095e-06    0.00067455      0.1367      0.9362       33
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T3%20-%20Seasonality_files/T3%20-%20Seasonality_3_3.png)

### Multiplicative Seasonality

In case of multiplicative seasonality, the seasonal variations are
changing proportional to the level of the series. If multiplicative
seasonality is selected, `judgyprophet` will rescale the time series by
shifting all values positive with standard variance. An example is shown
here:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns
from judgyprophet.tutorials.resources import get_multiplicative_seasonality_linear_trend

example_data = get_multiplicative_seasonality_linear_trend()


# Cutoff the data to October 2020
cutoff = "2020-10-01"
data_cutoff = example_data.loc[:cutoff]

jp = JudgyProphet()
# The multiplicative example time series has a constant trend component, but the seasonality
#     is multiplicative and has a large amplitude. Again the period is set to 12.
jp.fit(
    data=data_cutoff,
    level_events=[],
    trend_events=[],
    seasonal_period=12,
    seasonal_type="mult",
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=int(12))

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling by shifting all values positive with 1-sd.
    WARNING:judgyprophet.utils:No active trend or level events (i.e. no event indexes overlap with data). The model will just fit a base trend to the data.


    Initial log joint probability = -775.468
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          19      -105.965     0.0490619       1203.53     0.04941           1       32
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          39      -12.2029     0.0634662       440.207      0.1485      0.1485       64
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          59     -0.683534     0.0372809       68.3501           1           1       84
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          79     -0.454028    0.00336031       7.70556      0.8053      0.8053      109
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          99     -0.442297   0.000894112       6.38619    0.006837           1      134
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         119     -0.402061     0.0581892       7.15489      0.5184           1      159
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         139     -0.287525     0.0223029      0.741186           1           1      187
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         159     -0.285187   0.000493407       1.35585      0.3357           1      213
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         179     -0.285081   5.78588e-05      0.106578      0.2866     0.02866      238
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         199      -0.28508   6.70665e-06    0.00713309      0.1648      0.1648      259
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         206      -0.28508   1.07542e-05    0.00613299           1           1      267
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T3%20-%20Seasonality_files/T3%20-%20Seasonality_5_3.png)

### Combining Seasonality with Events

Both seasonality can be combined with trend events, damping, level
events, and unspecified changepoints. We will now walk through an
example time series which contains a damped trend event and shows
additive seasonality. Let’s look at the data:

``` python
from judgyprophet.tutorials.resources import get_additive_seasonal_damped_trend_event

example_data = get_additive_seasonal_damped_trend_event()
example_data.plot.line()
```

    <AxesSubplot:>

![png](T3%20-%20Seasonality_files/T3%20-%20Seasonality_7_1.png)

We can see from the plot that there is an uptick in trend around January
2018. The uptick in trend is quite steep until the end of 2019 where we
observe stronger damping. We also see that the time series has a
seasonal pattern, with a `seasonal_period` of 12 and a peak in December
each year.

#### Forecasting with JudgyProphet before the event occurs

The estimate of the trend event is a trend increase of 6 with a damping
parameter of 0.9.

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

trend_events = [
    {'name': "New saturating market entry", 'index': '2018-01-01', 'm0': 6, 'gamma': .9}
]

# Cutoff the data to June 2017
cutoff = "2017-06-01"
data_cutoff = example_data.loc[:cutoff]

jp = JudgyProphet()
# We have one trend event and no level events. The seasonality is additive again.
jp.fit(
    data=data_cutoff,
    sigma_trend=0.1,
    level_events=[],
    unspecified_changepoints=10,
    sigma_unspecified_changepoints=.2,
    trend_events=trend_events,
    seasonal_period=12,
    seasonal_type="add",
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=int(36))

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    WARNING:judgyprophet.judgyprophet:Post-event data for trend event New saturating market entry less than 0 points. Event deactivated in model. Event index: 2018-01-01, training data end index: 2015-01-01 00:00:00
    WARNING:judgyprophet.utils:No active trend or level events (i.e. no event indexes overlap with data). The model will just fit a base trend to the data.


    Initial log joint probability = -5618.69
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          19      -19.5311      0.107624       53.5313           1           1       25
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          39      -14.4873    0.00450173       32.1225           1           1       54
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          59      -11.3416     0.0164259       63.5553      0.1261           1       83
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          79       -10.031    0.00192964       42.0027      0.4045      0.4045      111
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          99      -9.91568    0.00123052       20.6714      0.3407           1      136
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         119      -9.74583     0.0178502       45.6554           1           1      161
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         139      -9.58822    0.00240934       24.5607      0.1831           1      185
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         159      -9.55939   0.000174537        18.265           1           1      214
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         179      -9.55721   4.74996e-06       18.6232           1           1      243
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         199      -9.55707   1.07478e-05       21.7289           1           1      267
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         219       -9.5541    1.0315e-05       21.3723           1           1      288
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         239      -9.50413    0.00075237       20.6816      0.6394      0.6394      309
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         259      -9.41589   0.000411908       21.7004           1           1      331
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         279      -9.39884   1.20103e-05       20.8879      0.3439      0.3439      355
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         299      -9.39799   4.26641e-06       20.0502       0.476       0.476      379
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         319      -9.37685   5.71275e-05        19.885      0.3824      0.3824      405
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         339      -9.34918   0.000126588       18.6708      0.9182      0.9182      429
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         359      -9.34728    1.3233e-05       20.7019           1           1      455
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         379      -9.33447   2.78134e-05       20.1768      0.5231      0.5231      477
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         399      -9.33284   2.29302e-06       20.0762      0.6157      0.6157      501
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         419      -9.33248    1.2898e-05       19.1065     0.08486           1      529
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         420      -9.33247   3.42574e-06       21.2674   1.793e-07       0.001      576  LS failed, Hessian reset
         439      -9.33237   2.57199e-07       22.0328      0.3448           1      603
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         450      -9.33237   7.89272e-09       20.4987      0.2986           1      621
    Optimization terminated normally:
      Convergence detected: absolute parameter change was below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T3%20-%20Seasonality_files/T3%20-%20Seasonality_10_3.png)

#### Forecasting with JudgyProphet after the event occurs

We can see that the model picks up correctly the seasonal pattern and
incorporates the trend event. After a few more data points are observed,
the model learned that the initial trend event estimates were poorly and
corrects its forecast accordingly. Let’s look at the forecast repeated
in June 2018:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns

trend_events = [
    {'name': "New saturating market entry", 'index': '2018-01-01', 'm0': 6, 'gamma': .9}
]

# Cutoff the data to June 2017
cutoff = "2018-06-01"
data_cutoff = example_data.loc[:cutoff]

jp = JudgyProphet()
# We have one trend event and no level events. The seasonality is additive again.
jp.fit(
    data=data_cutoff,
    sigma_trend=0.1,
    level_events=[],
    unspecified_changepoints=10,
    sigma_unspecified_changepoints=.2,
    trend_events=trend_events,
    seasonal_period=12,
    seasonal_type="add",
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=int(36))

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New saturating market entry to model. Event index: 2018-01-01, training data start index: 2015-01-01 00:00:00, training data end index: 2018-06-01 00:00:00. Initial gradient: 6. Damping: 0.9.
    WARNING:judgyprophet.utils:Unspecified changepoint with index 2018-01-01 00:00:00 also specified as a level or trend event. Removing this changepoint.


    Initial log joint probability = -4219.81
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          19      -45.4011     0.0443425       326.047           1           1       26
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          39      -15.3423    0.00201394       41.5814           1           1       53
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          59      -11.4289     0.0222713       70.3068           1           1       82
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          79      -9.74468     0.0019911       38.3992      0.8659      0.8659      110
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          99      -9.55453    0.00466659       25.2859       0.949       0.949      137
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         119      -9.42124   0.000328639       21.1329      0.3671           1      163
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         139      -9.38936    0.00514148       27.2595           1           1      196
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         159       -9.3399    0.00760439       37.8239           1           1      223
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         179      -9.27552   0.000532321       24.6411      0.4927      0.4927      247
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         199      -9.23377    0.00261612       16.9745      0.8399      0.8399      269
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         202      -9.23013   0.000104263       18.2691   5.753e-06       0.001      364  LS failed, Hessian reset
         219      -9.19983    0.00127842       29.8317           1           1      381
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         239      -9.16268    0.00026094       21.9496       3.298      0.3298      403
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         259       -9.1381    0.00160896       29.4252      0.9121      0.9121      426
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         279      -9.11448   0.000430079       17.2828           1           1      451
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         299      -9.09913   0.000132984       15.3301      0.2225      0.2225      479
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         319      -9.09577   7.96264e-05       20.5328      0.9646      0.9646      507
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         339       -9.0927   5.35475e-06       18.2672        0.38           1      533
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         359      -9.09266   1.83214e-08       16.0514           1           1      567
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
         365      -9.09266   6.04342e-09       18.2719      0.6683      0.6683      573
    Optimization terminated normally:
      Convergence detected: absolute parameter change was below tolerance


    /Users/kpxh622/github/judgyprophet/judgyprophet/utils.py:31: UserWarning: Unspecified changepoint with index 2018-01-01 00:00:00 also specified as a level or trend event. Removing this changepoint.
      warnings.warn(msg)





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T3%20-%20Seasonality_files/T3%20-%20Seasonality_12_4.png)

## Hyperparameters

`judgyprophet` gives full control over hyperparameters. These are mainly
scale parameters which determine how sensitive `judgyprophet` is to
actuals. The most useful of these are:

-   `sigma_level` – each level event is assigned a Normal prior with
    mean the business estimate (‘c0’), and standard deviation is set to
    this arg. The lower this is, the closer the event will follow the
    business suggestions – its default is 0.1.
-   `sigma_trend` – each trend event is assigned a Normal prior with
    mean the business estimate (‘m0’), and standard deviation is set to
    this arg. The lower this is, the closer the event will follow the
    business suggestions – its default is 0.1.
-   `sigma_unspecified_changepoints` – each unspecified changepoint is
    assigned a Laplace prior (equivalent to L1 penalty) with mean 0, and
    standard deviation set to this arg. The lower this is, the less
    likely the model will use this changepoint.

Other hyperparameters are:

-   `sigma_base_bias` – the initial bias is assigned a Normal prior with
    mean 0, and standard deviation is set to this arg. The lower this
    is, the more this parameter will be penalised down. Default 1.
-   `sigma_base_trend` – the initial trend is assigned a Normal prior
    with mean 0, and standard deviation is set to this arg. The lower
    this is, the more this parameter will be penalised down. Default 1.
-   `sigma_seasonal` – the seasonality parameters are assigned a Normal
    prior with mean 0, and standard deviation is set to this arg. The
    lower this is, the more this parameter will be penalised down.
    Default 1.

### Example of Hyperparameter impact

The default hyperparameter values for `judgyprophet` tend to deliver
reasonable estimations. However, there might be situations in which you
would like to adjust those to pay more or less attention to prior
knowledge. Values between 1 and 0.01 seem to deliver quite stable
results.

We will now adjusting the prior for the trend event to understand the
hyperparameter impact. Let’s load again the time series which contains a
damped trend event and additive seasonality which we saw in the
Seasonality tutorial. This time, we added some correlated noise after
the trend event happened (between April and August 2018):

``` python
from judgyprophet.tutorials.resources import get_additive_seasonal_damped_trend_event_correlated_noise

example_data = get_additive_seasonal_damped_trend_event_correlated_noise()
example_data.plot.line()
```

    <AxesSubplot:>

![png](T4%20-%20Hyperparameters_files/T4%20-%20Hyperparameters_2_1.png)

The trend event parameters in this case are the exact solution to trend
shown in the above time series:

``` python
trend_events = [
    {'name': "New saturating market entry", 'index': '2018-01-01', 'm0': 9, 'gamma': .9}
]
```

#### Forecasting with the default hyperparameters

We are now creating a forecast in July 2017. Using the default value for
`sigma_trend` creates the following forecast:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns


# Cutoff the data to June 2017
cutoff = "2018-08-01"
data_cutoff = example_data.loc[:cutoff]

jp = JudgyProphet()
# We have one trend event and no level events. The seasonality is additive again.
jp.fit(
    data=data_cutoff,
    sigma_trend=0.1,
    level_events=[],
    unspecified_changepoints=0,
    sigma_unspecified_changepoints=.2,
    trend_events=trend_events,
    seasonal_period=12,
    seasonal_type="add",
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=int(36))

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New saturating market entry to model. Event index: 2018-01-01, training data start index: 2015-01-01 00:00:00, training data end index: 2018-08-01 00:00:00. Initial gradient: 9. Damping: 0.9.


    Initial log joint probability = -1125.09
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          17      -24.2744   0.000161304    0.00638672           1           1       31
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T4%20-%20Hyperparameters_files/T4%20-%20Hyperparameters_6_3.png)

Nevertheless, the default hyperparameter tend to underestimate the
impact of the trend event: Since the model is learning from the
previously observed data points (which contain the correlated noise),
the model assumes that the prior trend event estimates were inaccurate
and it corrects the estimated trend downwards.

#### Testing a very low `sigma_trend` value

If we are certain that our prior knowledge is correct and we assume a
high volatility in our time series, we can reduce the prior value for
`sigma_trend`. That will encourge the model to strictly follow initial
trend event estimations given to it. In this case, adjusting the
hyperparameter value can improve our forecast accuracy:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns


# Cutoff the data to June 2017
cutoff = "2018-08-01"
data_cutoff = example_data.loc[:cutoff]

jp = JudgyProphet()
# We have one trend event and no level events. The seasonality is additive again.
jp.fit(
    data=data_cutoff,
    sigma_trend=0.01,
    level_events=[],
    unspecified_changepoints=0,
    sigma_unspecified_changepoints=.2,
    trend_events=trend_events,
    seasonal_period=12,
    seasonal_type="add",
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=int(36))

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New saturating market entry to model. Event index: 2018-01-01, training data start index: 2015-01-01 00:00:00, training data end index: 2018-08-01 00:00:00. Initial gradient: 9. Damping: 0.9.


    Initial log joint probability = -1191.51
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          19      -42.6551     0.0255416       21.9616           1           1       25
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          39      -40.5184     0.0027457       2.62096           1           1       51
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          59      -40.4754   0.000243228      0.568846           1           1       74
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          79       -40.474   0.000184754      0.147366           1           1       99
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          97       -40.474   6.17252e-05     0.0258339           1           1      121
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T4%20-%20Hyperparameters_files/T4%20-%20Hyperparameters_9_3.png)

#### Testing a very high `sigma_trend` value

If you are not certain if your estimations are correct prior to the
event, it is sensible to assign a higher value to the `sigma_trend`
parameter. This is beneficial in situations where our prior estimations
are poorly. Let’s look at the forecast created in November 2018:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns


# Cutoff the data to June 2017
cutoff = "2018-11-01"
data_cutoff = example_data.loc[:cutoff]

poor_trend_events = [
    {'name': "New saturating market entry", 'index': '2018-01-01', 'm0': 1, 'gamma': .9}
]

jp = JudgyProphet()
# We have one trend event and no level events. The seasonality is additive again.
jp.fit(
    data=data_cutoff,
    sigma_trend=10,
    level_events=[],
    unspecified_changepoints=0,
    sigma_unspecified_changepoints=.2,
    trend_events=poor_trend_events,
    seasonal_period=12,
    seasonal_type="add",
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=int(36))

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New saturating market entry to model. Event index: 2018-01-01, training data start index: 2015-01-01 00:00:00, training data end index: 2018-11-01 00:00:00. Initial gradient: 1. Damping: 0.9.


    Initial log joint probability = -962.005
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          19      -16.6821   0.000169582       0.22355           1           1       27
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          39       -16.682   0.000103557     0.0974231           1           1       50
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          59       -16.682   5.07955e-06     0.0182171      0.1133           1       78
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T4%20-%20Hyperparameters_files/T4%20-%20Hyperparameters_11_3.png)

The model with a large prior for `simga_trend` learned that the prior
trend event estimates underestimated the trend change, and adjusted its
forecast accordingly, whereas the default value would produce a larger
forecasting error:

``` python
from judgyprophet import JudgyProphet
import pandas as pd
import seaborn as sns


# Cutoff the data to June 2017
cutoff = "2018-11-01"
data_cutoff = example_data.loc[:cutoff]

poor_trend_events = [
    {'name': "New saturating market entry", 'index': '2018-01-01', 'm0': 1, 'gamma': .9}
]

jp = JudgyProphet()
# We have one trend event and no level events. The seasonality is additive again.
jp.fit(
    data=data_cutoff,
    sigma_trend=0.1,
    level_events=[],
    unspecified_changepoints=0,
    sigma_unspecified_changepoints=.2,
    trend_events=poor_trend_events,
    seasonal_period=12,
    seasonal_type="add",
    # Set random seed for reproducibility
    seed=13
)
predictions = jp.predict(horizon=int(36))

# Plot the data:
predict_df = (
    predictions.reset_index()
    .rename(columns={'index': 'ds', 'forecast': 'value'})
    .assign(method="JudgyProphet")
    .loc[:, ["ds", "value", "insample", "method"]]
)
actuals_df = (
    data_cutoff.reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Actuals", insample=True)
)
future_actuals_df = (
    example_data.loc[cutoff:]
    .reset_index()
    .rename(columns={'index': 'ds', 0: 'value'})
    .assign(method="Future Actuals", insample=False)
)
plot_df = (
    pd.concat([predict_df, actuals_df, future_actuals_df])
    .reset_index(drop=True)
)

sns.lineplot(data=plot_df, x='ds', y='value', hue='method', style='insample', style_order=[True, False])
```

    INFO:judgyprophet.judgyprophet:Rescaling onto 0-mean, 1-sd.
    INFO:judgyprophet.judgyprophet:Adding trend event New saturating market entry to model. Event index: 2018-01-01, training data start index: 2015-01-01 00:00:00, training data end index: 2018-11-01 00:00:00. Initial gradient: 1. Damping: 0.9.


    Initial log joint probability = -979.779
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          19      -21.5681   0.000488491      0.799235           1           1       27
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes
          38      -21.5671   6.15078e-05    0.00396993       1.018      0.1018       65
    Optimization terminated normally:
      Convergence detected: relative gradient magnitude is below tolerance





    <AxesSubplot:xlabel='ds', ylabel='value'>

![png](T4%20-%20Hyperparameters_files/T4%20-%20Hyperparameters_13_3.png)
