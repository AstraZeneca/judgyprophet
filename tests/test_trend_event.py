"""
Test expert prophet with trend events
"""
import pprint

import numpy as np
import pandas as pd
import pytest

from judgyprophet import JudgyProphet


@pytest.fixture
def create_trend_event():
    np.random.seed(13)
    # Declare constants
    event_date = "2018-01-31"
    trend_events = [
        {"name": "T1", "index": event_date, "m0": 5.0, "gamma": 1.0},
    ]
    bias = 4.0
    obs_scale = 1.0
    ds = pd.date_range(start="2016-01-31", end="2021-05-31", freq="M")
    T = len(ds)
    y = bias + np.random.normal(scale=obs_scale, size=T)
    for trend_i in trend_events:
        delta_i = ds >= event_date
        t_i = np.cumsum(ds >= event_date) - 1
        y += delta_i * trend_i["m0"] * t_i
    return {
        "data": pd.Series(y, index=ds),
        "level_events": [],
        "trend_events": trend_events,
    }


@pytest.fixture
def create_damped_trend_event():
    np.random.seed(13)
    # Declare constants
    event_date = "2018-01-31"
    trend_events = [
        {"name": "T1", "index": event_date, "m0": 5.0, "gamma": 0.8},
    ]
    bias = 4.0
    obs_scale = 1.0
    ds = pd.date_range(start="2016-01-31", end="2021-05-31", freq="M")
    T = len(ds)
    y = bias + np.random.normal(scale=obs_scale, size=T)
    for trend_i in trend_events:
        delta_i = ds > event_date
        t_i = np.cumsum(ds > event_date)
        y += (
            delta_i
            * trend_i["m0"]
            * (1 - trend_i["gamma"] ** t_i)
            / (1 - trend_i["gamma"])
        )
    return {
        "data": pd.Series(y, index=ds),
        "level_events": [],
        "trend_events": trend_events,
    }


def test_trend_event(create_trend_event):
    jp = JudgyProphet()
    jp.fit(**create_trend_event)
    model_output = jp.get_model_output()
    assert np.isclose(model_output["trend_events"][0]["m"], 5.0, atol=2e-1)
    all_unspecified_params = [
        param
        for cp in model_output["changepoint_events"]
        for param in [cp["cp_c"], cp["cp_m"]]
    ]
    assert np.allclose(all_unspecified_params, 0, atol=1e-2)
    predictions = jp.predict(horizon=12)
    predictions_out_of_sample = predictions["forecast"].values[-12:]
    # Fetch expected
    ds = pd.date_range(start="2016-01-31", end="2022-05-31", freq="M")
    delta_i = ds >= "2018-01-31"
    t_i = np.cumsum(delta_i) - (delta_i > 0)
    expected = 4.0 + delta_i * t_i * 5.0
    expected_out_of_sample = expected[-12:]
    assert np.allclose(predictions_out_of_sample, expected_out_of_sample, atol=1.0)


def test_trend_event_damped(create_damped_trend_event):
    """Check when changepoint clashes with event it is deactivated"""
    jp = JudgyProphet()
    jp.fit(**create_damped_trend_event)
    model_output = jp.get_model_output()
    assert np.isclose(model_output["trend_events"][0]["m"], 5.0, atol=2e-1)
    all_unspecified_params = [
        param
        for cp in model_output["changepoint_events"]
        for param in [cp["cp_c"], cp["cp_m"]]
    ]
    assert np.allclose(all_unspecified_params, 0, atol=1e-2)
    assert np.isclose(model_output["base"]["beta_0"], 4.0, atol=1.0)
    predictions = jp.predict(horizon=8)
    predictions_out_of_sample = predictions["forecast"].values[-12:]
    # Fetch expected
    ds = pd.date_range(start="2016-01-31", end="2022-01-31", freq="M")
    delta_i = ds > "2018-01-31"
    t_i = np.cumsum(delta_i)
    expected = 4.0 + delta_i * 5.0 * (1 - 0.8**t_i) / (1 - 0.8)
    expected_out_of_sample = expected[-12:]
    assert np.allclose(predictions_out_of_sample, expected_out_of_sample, atol=1.0)
