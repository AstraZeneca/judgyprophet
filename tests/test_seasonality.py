"""
Test expert prophet seasonality
"""
import logging
import pprint

import numpy as np
import pandas as pd
import pytest

from judgyprophet import JudgyProphet


def _create_seasonal_data(
    seasonal_type="add", trig_function="sin", seasonality=12, amplitude=10
):
    """Create examples of seasonal data."""
    ds = pd.date_range(start="2017-06-01", end="2021-06-01", freq="MS")
    T = len(ds)
    base_trend = 2
    np.random.seed(13)
    observations = base_trend * np.arange(T) + np.random.normal(loc=4, size=T)
    time = np.arange(0, T / seasonality, 1 / seasonality)
    if trig_function == "sin":
        wave_func = amplitude * np.sin(2 * np.pi * time)
    if trig_function == "cos":
        wave_func = amplitude * np.cos(2 * np.pi * time)
    if trig_function == "mix":
        wave_func = amplitude * np.cos(2 * np.pi * time) + amplitude * np.sin(
            2 * np.pi * time
        )

    if seasonal_type == "add":
        observations += wave_func
    if seasonal_type == "mult":
        observations *= wave_func
    return pd.Series(observations, index=ds)


@pytest.mark.parametrize(
    "seasonal_type, trig_function, seasonality, amplitude, maximum, abs_cum_sum, sum_forecast",
    [
        ("add", "sin", 12, 10, 8.8, 74.7, 1116.3),
        ("add", "cos", 12, 10, -4.8, 67.0, 1092.5),
        ("add", "mix", 12, 10, 3.9, 100.4, 1092.8),
    ],
)
def test_additive_seasonality(
    seasonal_type,
    trig_function,
    seasonality,
    amplitude,
    maximum,
    abs_cum_sum,
    sum_forecast,
):
    """Testing additive seasonality (no trend or level events)."""
    jp = JudgyProphet()

    additive_seasonal_data = _create_seasonal_data(
        seasonal_type, trig_function, seasonality, amplitude
    )
    jp.fit(
        data=additive_seasonal_data,
        level_events=[],
        trend_events=[],
        seasonal_type=seasonal_type,
        seasonal_period=seasonality,
        unspecified_changepoints=0,
        seed=17,
    )
    model_output = jp.get_model_output()

    assert np.allclose(model_output["base"]["seasonal_level"][4], maximum, rtol=0.1)
    assert np.allclose(
        abs(model_output["base"]["seasonal_level"][0:12]).sum(), abs_cum_sum, rtol=0.1
    )

    predictions = jp.predict(horizon=10)
    assert np.allclose(
        predictions[predictions["insample"] == False].forecast.sum(),
        sum_forecast,
        rtol=0.1,
    )


@pytest.mark.parametrize(
    "seasonal_type, trig_function, seasonality, amplitude, maximum, abs_cum_sum, sum_forecast",
    [
        ("mult", "sin", 12, 10, 0.55, 5.1, 481.79),
        ("mult", "cos", 12, 10, -0.28, 5.0, 1226.6),
        ("mult", "mix", 12, 10, 0.19, 5.37, 979.7),
    ],
)
def test_multiplicative_seasonality(
    seasonal_type,
    trig_function,
    seasonality,
    amplitude,
    maximum,
    abs_cum_sum,
    sum_forecast,
):
    """Testing multiplicative seasonality (no trend or level events)."""
    jp = JudgyProphet()

    multiplicative_seasonal_data = _create_seasonal_data(
        seasonal_type, trig_function, seasonality, amplitude
    )
    jp.fit(
        data=multiplicative_seasonal_data[:-10],
        level_events=[],
        trend_events=[],
        seasonal_type=seasonal_type,
        seasonal_period=seasonality,
        unspecified_changepoints=0,
        seed=17,
    )
    model_output = jp.get_model_output(rescale=False)

    assert np.allclose(model_output["base"]["seasonal_level"][4], maximum, rtol=0.1)
    assert np.allclose(
        abs(model_output["base"]["seasonal_level"][0:12]).sum(), abs_cum_sum, rtol=0.1
    )

    predictions = jp.predict(horizon=10)
    assert np.allclose(
        sum(predictions[predictions["insample"] == False].forecast),
        sum_forecast,
        rtol=0.1,
    )
