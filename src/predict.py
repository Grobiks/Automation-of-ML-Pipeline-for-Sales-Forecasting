import h2o
import pandas as pd
from src.config import *

def predict():
    h2o.init()

    test_df = pd.read_parquet(PROCESSED_TEST_PATH)
    raw_test = pd.read_csv(TEST_PATH)

    import os
    model_files = [f for f in os.listdir(MODELS_DIR) if not f.endswith(".csv")]
    if not model_files:
        raise FileNotFoundError("Модель не найдена. Сначала запусти train.py")

    model_path = os.path.join(MODELS_DIR, model_files[0])
    model = h2o.load_model(model_path)

    features = [c for c in test_df.columns if c != ID_COL]
    h_test = h2o.H2OFrame(test_df[features])
    preds = model.predict(h_test).as_data_frame()["predict"].values

    submission = pd.DataFrame({
        ID_COL: raw_test[ID_COL],
        TARGET_COL: preds
    })
    submission.to_csv(f"{TABLES_DIR}/submission.csv", index=False)
    print(f"Submission сохранён: reports/tables/submission.csv")

    h2o.cluster().shutdown()

if __name__ == "__main__":
    predict()