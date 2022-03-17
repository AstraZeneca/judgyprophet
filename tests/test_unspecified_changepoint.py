"""
Test expert prophet with level events
"""
import pprint
import warnings

import numpy as np
import pandas as pd
import pytest

from judgyprophet import JudgyProphet


@pytest.fixture
def create_level_event():
    np.random.seed(13)
    # Declare constants
    event_date = "2018-01-31"
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
    return {
        "data": pd.Series(y, index=ds),
        "level_events": [],
        "trend_events": [],
        "unspecified_changepoints": 5,
    }


@pytest.fixture
def create_level_trend_event():
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
        delta_i = ds >= event_date
        t_i = np.cumsum(ds >= event_date) - 1
        y += delta_i * trend_i["m0"] * t_i
    return {
        "data": pd.Series(y, index=ds),
        "level_events": [],
        "trend_events": [],
        "unspecified_changepoints": 5,
    }


def test_level_event(create_level_event):
    jp = JudgyProphet()
    jp.fit(**create_level_event)
    predictions = jp.predict(horizon=7)
    ds = predictions.index
    delta_i = ds >= "2018-01-31"
    expected = 4.0 + delta_i * 5.0
    # Check predictions are pretty close
    assert np.allclose(expected, predictions["forecast"], atol=3.0)


def test_level_trend_event(create_level_trend_event):
    jp = JudgyProphet()
    jp.fit(**create_level_trend_event)
    predictions = jp.predict(horizon=7)
    # Fetch expected
    ds = predictions.index
    delta_i = ds >= "2018-01-31"
    t_i = np.cumsum(delta_i) - (delta_i > 0)
    expected = 4.0 + delta_i * 5.0 + delta_i * t_i * 5.0
    # Check predictions are pretty close
    predictions_out_of_sample = predictions["forecast"].values[-7:]
    expected_out_of_sample = expected[-7:]
    assert np.allclose(predictions_out_of_sample, expected_out_of_sample, atol=1.0)
