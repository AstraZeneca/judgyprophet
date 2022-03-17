"""
Test expert prophet with different indexing
"""
import numpy as np
import pandas as pd
import pytest

from judgyprophet import JudgyProphet


@pytest.fixture
def create_int_index_event_data():
    """
    Data with int index
    """
    np.random.seed(13)
    # Declare constants
    event_date = 30
    level_events = [
        {"name": "L1", "index": event_date, "c0": 5.0},
    ]
    bias = 4.0
    obs_scale = 1.0
    ds = np.arange(60)
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


def test_int_index(create_int_index_event_data):
    jp = JudgyProphet()
    jp.fit(**create_int_index_event_data)
    model_output = jp.get_model_output()
    assert np.isclose(model_output["level_events"][0]["c"], 5.0, atol=1e-1)
    all_unspecified_params = [
        param
        for cp in model_output["changepoint_events"]
        for param in [cp["cp_c"], cp["cp_m"]]
    ]
    assert np.allclose(all_unspecified_params, 0, atol=1e-1)
    assert np.isclose(model_output["base"]["beta_0"], 4.0, atol=1.0)
    predictions = jp.predict(horizon=5)
    # Fetch expected
    ds = np.arange(65)
    delta_i = ds >= 30
    expected = 4.0 + delta_i * 5.0
    assert np.allclose(predictions["forecast"], expected, atol=1.0)


def test_float_index(create_int_index_event_data, caplog):
    """
    Check a float index results in an error
    """
    float_index_data = create_int_index_event_data.copy()
    float_index_data["data"].index += 0.1
    jp = JudgyProphet()
    with pytest.raises(ValueError):
        jp.fit(**float_index_data)
    assert "Arg 'data' must have index of type" in caplog.text
