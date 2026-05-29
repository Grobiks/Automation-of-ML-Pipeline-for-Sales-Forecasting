import numpy as np
import pytest
from src.evaluate import smape

def test_smape_perfect_prediction():
    actual    = np.array([100, 200, 300])
    predicted = np.array([100, 200, 300])
    assert smape(actual, predicted) == pytest.approx(0.0, abs=1e-5)

def test_smape_returns_positive():
    actual    = np.array([100, 200, 300])
    predicted = np.array([110, 190, 320])
    assert smape(actual, predicted) > 0

def test_smape_range():
    actual    = np.array([100, 200, 300])
    predicted = np.array([50,  400, 150])
    result = smape(actual, predicted)
    assert 0 <= result <= 200