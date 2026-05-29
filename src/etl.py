import pandas as pd
from src.config import *

def extract():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def transform(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df["year"]        = df[DATE_COL].dt.year
    df["month"]       = df[DATE_COL].dt.month
    df["day"]         = df[DATE_COL].dt.day
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["quarter"]     = df[DATE_COL].dt.quarter
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    for col in CAT_COLS:
        df[col] = df[col].astype("category").cat.codes

    df = df.drop(columns=[DATE_COL], errors="ignore")
    return df

def load(train, test):
    train.to_parquet(PROCESSED_TRAIN_PATH, index=False)
    test.to_parquet(PROCESSED_TEST_PATH, index=False)
    print("Данные сохранены в data/processed/")

if __name__ == "__main__":
    train, test = extract()
    train_t = transform(train)
    test_t  = transform(test)
    load(train_t, test_t)
    print(f"Train: {train_t.shape}, Test: {test_t.shape}")