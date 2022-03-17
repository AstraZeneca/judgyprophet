"""
Test expert prophet with level events
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from judgyprophet import JudgyProphet


@pytest.fixture
def create_single_event_data():
    np.random.seed(13)
    # Declare constants
    event_date = "2017-12-31"
    level_events = [
        {"name": "L1", "index": event_date, "c0": 5.0},
    ]
    bias = 4.0
    obs_scale = 0.5
    ds = pd.date_range(start="2016-01-31", end="2021-05-31", freq="M")
    T = len(ds)
    y = bias + np.random.normal(scale=obs_scale, size=T)
    for level_i in level_events:
        delta_i = ds >= event_date
        y += delta_i * level_i["c0"]
    return {
        "data": pd.Series(y, index=ds),
        "level_events": level_events,
        "trend_events": [],
    }


def test_level_event(create_single_event_data):
    jp = JudgyProphet()
    jp.fit(**create_single_event_data)
    model_output = jp.get_model_output()
    assert np.isclose(model_output["level_events"][0]["c"], 5.0, atol=1e-1)
    all_unspecified_params = [
        param
        for cp in model_output["changepoint_events"]
        for param in [cp["cp_c"], cp["cp_m"]]
    ]
    assert np.allclose(all_unspecified_params, 0, atol=1e-2)
    assert np.isclose(model_output["base"]["beta_0"], 4.0, atol=1.0)
    predictions = jp.predict(horizon=5)
    # Fetch expected
    ds = pd.date_range(start="2016-01-31", end="2021-10-31", freq="M")
    delta_i = ds >= "2017-12-31"
    expected = 4.0 + delta_i * 5.0
    assert np.allclose(predictions["forecast"], expected, atol=1.0)
