import pandas as pd
import pytest
from src.etl import transform

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id":      [1, 2, 3],
        "date":    ["2020-01-01", "2020-01-02", "2020-01-03"],
        "country": ["Finland", "Norway", "Sweden"],
        "store":   ["KaggleMart", "KaggleRama", "KaggleMart"],
        "product": ["Holographic Goose", "Spooky Cup", "Holographic Goose"],
        "num_sold": [100, 200, 150],
    })

def test_transform_adds_date_features(sample_df):
    result = transform(sample_df)
    for col in ["year", "month", "day", "day_of_week", "quarter", "is_weekend"]:
        assert col in result.columns, f"Колонка {col} не найдена"

def test_transform_drops_date_column(sample_df):
    result = transform(sample_df)
    assert "date" not in result.columns

def test_transform_no_nulls(sample_df):
    result = transform(sample_df)
    assert result.isnull().sum().sum() == 0

def test_transform_is_weekend(sample_df):
    result = transform(sample_df)
    assert result["is_weekend"].isin([0, 1]).all()

def test_transform_cat_cols_encoded(sample_df):
    result = transform(sample_df)
    for col in ["country", "store", "product"]:
        assert result[col].dtype in ["int8", "int16", "int32", "int64"]