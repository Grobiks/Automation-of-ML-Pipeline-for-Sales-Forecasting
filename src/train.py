import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import mlflow
import mlflow.sklearn
import time
from src.config import *
from src.evaluate import smape, save_metrics, plot_predictions, plot_feature_importance

def train():
    h2o.init()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_df = pd.read_parquet(PROCESSED_TRAIN_PATH)

    split_idx = int(len(train_df) * (1 - VALIDATION_SPLIT))
    train_part = train_df.iloc[:split_idx]
    val_part   = train_df.iloc[split_idx:]

    h_train = h2o.H2OFrame(train_part)
    h_val   = h2o.H2OFrame(val_part)

    features = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]

    with mlflow.start_run():
        print("Запускаем H2O AutoML...")
        start = time.time()

        aml = H2OAutoML(
            max_models=H2O_MAX_MODELS,
            max_runtime_secs=H2O_MAX_RUNTIME_SECS,
            seed=H2O_SEED,
            sort_metric="RMSE"
        )
        aml.train(x=features, y=TARGET_COL, training_frame=h_train, leaderboard_frame=h_val)

        elapsed = round(time.time() - start, 1)
        print(f"Обучение завершено за {elapsed} сек")

        best = aml.leader
        preds = best.predict(h_val).as_data_frame()["predict"].values
        actual = val_part[TARGET_COL].values

        metrics = {
            "smape": round(smape(actual, preds), 4),
            "rmse":  round(float(best.rmse(valid=False)), 4),
            "training_time_sec": elapsed,
            "best_model": best.algo,
        }

        print(f"SMAPE: {metrics['smape']}%")
        print(f"Лучшая модель: {metrics['best_model']}")

        mlflow.log_metrics({"smape": metrics["smape"], "rmse": metrics["rmse"]})
        mlflow.log_param("best_model", metrics["best_model"])
        mlflow.log_param("training_time_sec", elapsed)
        mlflow.log_artifact(f"{FIGURES_DIR}/predictions.png")
        mlflow.log_artifact(f"{FIGURES_DIR}/feature_importance.png")
        mlflow.log_artifact(f"{TABLES_DIR}/leaderboard.csv")
        mlflow.log_artifact(f"{TABLES_DIR}/metrics.json")

        save_metrics(metrics)
        plot_predictions(val_part[TARGET_COL], preds)
        plot_feature_importance(best)

        model_path = h2o.save_model(best, path=MODELS_DIR, force=True)
        print(f"Модель сохранена: {model_path}")

        lb = aml.leaderboard.as_data_frame()
        lb.to_csv(f"{TABLES_DIR}/leaderboard.csv", index=False)
        print("Лидерборд сохранён: reports/tables/leaderboard.csv")

    h2o.cluster().shutdown()

if __name__ == "__main__":
    train()