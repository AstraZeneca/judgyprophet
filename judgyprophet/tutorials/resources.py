from typing import Any, Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet


def get_level_event() -> pd.Series:
    """Get example data for a level event tutorial"""
    dates = pd.date_range(start="2019-01-01", end="2021-01-01", freq="MS")
    T = len(dates)
    level_time = "2020-03-01"
    level_amount = 8

    # Set temporary seed for reproducibility
    state = np.random.get_state()
    np.random.seed(13)
    observations = np.random.normal(loc=4, size=T)
    np.random.set_state(state)
    observations += (dates > level_time) * level_amount
    output = pd.Series(observations, index=dates)
    return output


def get_trend_event(uptake: bool = True) -> pd.Series:
    """Get example data for a trend event tutorial"""
    dates = pd.date_range(start="2019-06-01", end="2021-06-01", freq="MS")
    T = len(dates)
    trend_time = "2020-09-01"
    base_trend = 1
    trend_coefficient = 4

    # Set temporary seed for reproducibility
    state = np.random.get_state()
    np.random.seed(13)
    observations = base_trend * np.arange(T) + np.random.normal(loc=4, size=T)
    np.random.set_state(state)
    if uptake:
        observations += np.cumsum(dates > trend_time) * trend_coefficient
    else:
        observations -= np.cumsum(dates > trend_time) * trend_coefficient
    output = pd.Series(observations, index=dates)
    return output


def get_damped_trend_event() -> pd.Series:
    """Get example data for a trend event tutorial"""
    dates = pd.date_range(start="2019-06-01", end="2021-06-01", freq="MS")
    T = len(dates)
    trend_time = "2020-01-01"
    base_trend = 0.5
    trend_coefficient = 4
    damping = 0.8

    # Set temporary seed for reproducibility
    state = np.random.get_state()
    np.random.seed(13)
    observations = base_trend * np.arange(T) + np.random.normal(loc=4, size=T)
    np.random.set_state(state)
    delta = dates > trend_time
    t = np.cumsum(delta)
    observations += delta * trend_coefficient * (1 - damping**t) / (1 - damping)
    output = pd.Series(observations, index=dates)
    return output


def get_additive_seasonality_linear_trend() -> pd.Series:
    """Get example data for additive seasonality tutorial"""
    dates = pd.date_range(start="2017-06-01", end="2021-06-01", freq="MS")
    T = len(dates)
    base_trend = 2
    state = np.random.get_state()
    np.random.seed(13)
    observations = base_trend * np.arange(T) + np.random.normal(loc=4, size=T)
    np.random.set_state(state)
    seasonality = 12
    time = np.arange(0, T / seasonality, 1 / seasonality)
    amplitude = 10
    sin_cos_wave = amplitude * np.cos(2 * np.pi * time) + amplitude * np.sin(
        2 * np.pi * time
    )
    observations += sin_cos_wave
    output = pd.Series(observations, index=dates)
    return output


def get_multiplicative_seasonality_linear_trend() -> pd.Series:
    """Get example data for multiplicative seasonality tutorial"""
    dates = pd.date_range(start="2017-06-01", end="2021-06-01", freq="MS")
    T = len(dates)
    base_trend = 2
    state = np.random.get_state()
    np.random.seed(13)
    observations = base_trend * np.arange(T) + np.random.normal(loc=4, size=T)
    np.random.set_state(state)
    seasonality = 12
    time = np.arange(0, T / seasonality, 1 / seasonality)
    amplitude = 6
    theta = 10
    sin_wave = amplitude * np.sin(2 * np.pi * time) + theta
    observations *= sin_wave
    output = pd.Series(observations, index=dates)
    return output


def get_additive_seasonal_damped_trend_event() -> pd.Series:
    """Get example data for additive seasonality and a trend event tutorial"""
    dates = pd.date_range(start="2015-01-01", end="2021-12-01", freq="MS")
    T = len(dates)
    trend_time = "2018-01-01"
    base_trend = 0.2
    trend_coefficient = 9
    damping = 0.9

    # Set temporary seed for reproducibility
    state = np.random.get_state()
    np.random.seed(13)
    observations = base_trend * np.arange(T) + np.random.normal(scale=5, loc=10, size=T)
    np.random.set_state(state)
    delta = dates > trend_time
    t = np.cumsum(delta)
    observations += delta * trend_coefficient * (1 - damping**t) / (1 - damping)

    # add sinus function for periodicity
    seasonality = 6
    time = np.arange(0, T / seasonality, 1 / seasonality)
    theta = 1.5
    amplitude = 12
    sinewave_1 = amplitude * np.sin(2 * np.pi * 0.5 * time + theta)
    sinewave_2 = amplitude / 2 * np.sin(2 * np.pi * 1 * time + theta)
    sinewave_3 = amplitude / 3 * np.sin(2 * np.pi * 1.5 * time + theta)
    observations += sinewave_1
    observations += sinewave_2
    observations += sinewave_3
    output = pd.Series(observations, index=dates)
    return output


def get_additive_seasonal_damped_trend_event_correlated_noise() -> pd.Series:
    """Get example data for additive seasonality and a trend event tutorial"""
    dates = pd.date_range(start="2015-01-01", end="2021-12-01", freq="MS")
    T = len(dates)
    trend_time = "2018-01-01"
    base_trend = 0.2
    trend_coefficient = 9
    damping = 0.9

    # Set temporary seed for reproducibility
    state = np.random.get_state()
    np.random.seed(13)
    observations = base_trend * np.arange(T) + np.random.normal(scale=5, loc=10, size=T)
    np.random.set_state(state)
    delta = dates > trend_time
    t = np.cumsum(delta)
    observations += delta * trend_coefficient * (1 - damping**t) / (1 - damping)

    # add correlated noise
    noise_start = "2018-04-01"
    noise_end = "2018-08-01"
    noise_trend_coefficient = -8
    noise_delta = (dates >= noise_start) & (dates <= noise_end)
    t = np.cumsum(noise_delta)
    observations += (
        noise_delta * noise_trend_coefficient * (1 - damping**t) / (1 - damping)
    )

    # add sinus function for periodicity
    seasonality = 6
    time = np.arange(0, T / seasonality, 1 / seasonality)
    theta = 1.5
    amplitude = 12
    sinewave_1 = amplitude * np.sin(2 * np.pi * 0.5 * time + theta)
    sinewave_2 = amplitude / 2 * np.sin(2 * np.pi * 1 * time + theta)
    sinewave_3 = amplitude / 3 * np.sin(2 * np.pi * 1.5 * time + theta)
    observations += sinewave_1
    observations += sinewave_2
    observations += sinewave_3
    output = pd.Series(observations, index=dates)
    return output


def base_prophet_forecast(
    actuals: pd.Series, cutoff: str, events: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Make a base forecast using prophet"""
    changepoint_list = [
        event_dict["index"]
        for event_dict in events
        if pd.to_datetime(event_dict["index"]) <= pd.to_datetime(cutoff)
    ]
    prophet_model = Prophet(changepoints=changepoint_list)
    cutoff_actuals = (
        actuals[:cutoff].reset_index().rename(columns={"index": "ds", 0: "y"})
    )
    prophet_model.fit(cutoff_actuals)
    future_df = actuals.reset_index().rename(columns={"index": "ds", 0: "y"})
    future_df = future_df[["ds"]]
    predictions = prophet_model.predict(future_df)
    predictions = predictions[["ds", "yhat"]].rename(columns={"yhat": "value"})
    predictions["insample"] = pd.to_datetime(predictions["ds"]) <= pd.to_datetime(
        cutoff
    )
    predictions["method"] = "prophet"
    return predictions


def plot_forecast(
    actuals: pd.Series,
    predictions: pd.Series,
    cutoff: str,
    events: List[Dict[str, Any]],
):
    """Plot a forecast that's cutoff before actuals finish"""
    base_prophet_df = base_prophet_forecast(
        actuals=actuals, cutoff=cutoff, events=events
    )
    # Find cutoff date for plot
    plot_cutoff = min(
        [base_prophet_df.ds.max(), actuals.index.max(), predictions.index.max()]
    )
    predict_df = (
        predictions.reset_index()
        .rename(columns={"index": "ds", "forecast": "value"})
        .assign(method="judgyprophet")
        .loc[:, ["ds", "value", "insample", "method"]]
    )
    actuals_df = (
        actuals.loc[:cutoff]
        .reset_index()
        .rename(columns={"index": "ds", 0: "value"})
        .assign(method="actuals", insample=True)
    )
    future_actuals_df = (
        actuals.loc[cutoff:]
        .reset_index()
        .rename(columns={"index": "ds", 0: "value"})
        .assign(method="actuals", insample=False)
    )
    plot_df = (
        pd.concat([predict_df, actuals_df, future_actuals_df, base_prophet_df])
        .reset_index(drop=True)
        .query(f"ds <= '{plot_cutoff}'")
    )
    sns.lineplot(
        data=plot_df,
        x="ds",
        y="value",
        hue="method",
        style="insample",
        style_order=[True, False],
    )
