"""
Test expert prophet with level events
"""
import pprint

import numpy as np
import pandas as pd
import pytest

from judgyprophet import JudgyProphet


@pytest.fixture
def create_single_event_data():
    np.random.seed(13)
    # Declare constants
    event_date = "2018-01-31"
    trend_events = [
        {"name": "T1", "index": event_date, "m0": 5.0, "gamma": 1.0},
    ]
    level_events = [
        {"name": "L1", "index": event_date, "c0": 5.0},
    ]
    bias = 4.0
    obs_scale = 1.0
    ds = pd.date_range(start="2016-01-31", end="2021-05-31", freq="M")
    T = len(ds)
    y = bias + np.random.normal(scale=obs_scale, size=T)
    for level_i in level_events:
        delta_i = ds >= event_date
        y += delta_i * level_i["c0"]
    for trend_i in trend_events:
        delta_i = ds > event_date
        t_i = np.cumsum(delta_i)
        y += delta_i * trend_i["m0"] * t_i
    return {
        "data": pd.Series(y, index=ds),
        "level_events": level_events,
        "trend_events": trend_events,
        "seed": 13,
    }


def test_level_trend_event(create_single_event_data):
    jp = JudgyProphet()
    jp.fit(**create_single_event_data)
    predictions = jp.predict(horizon=7)
    # Fetch expected
    ds = pd.date_range(start="2016-01-31", end="2021-12-31", freq="M")
    delta_i = ds >= "2018-01-31"
    t_i = np.cumsum(delta_i) - (delta_i > 0)
    expected = 4.0 + delta_i * 5.0 + delta_i * t_i * 5.0
    ape = np.abs((predictions["forecast"] - expected)) / expected
    predictive_ape = ape[-7:]
    assert np.allclose(predictive_ape, 0, atol=1e-3)


def test_level_trend_event_clashing_changepoint(create_single_event_data):
    """Check when changepoint clashes with event it is deactivated"""
    jp = JudgyProphet()
    with pytest.warns(UserWarning):
        jp.fit(**create_single_event_data, unspecified_changepoints=["2018-01-31"])
    model_output = jp.get_model_output()
    assert len(model_output["changepoint_events"]) == 0
    predictions = jp.predict()
    # Fetch expected
    ds = pd.date_range(start="2016-01-31", end="2021-05-31", freq="M")
    delta_i = ds >= "2018-01-31"
    t_i = np.cumsum(delta_i) - (delta_i > 0)
    expected = 4.0 + delta_i * 5.0 + delta_i * t_i * 5.0
    ape = np.abs((predictions["forecast"] - expected)) / expected
    predictive_ape = ape[-7:]
    assert np.allclose(predictive_ape, 0, atol=1e-3)
