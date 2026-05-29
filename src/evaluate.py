import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR, TABLES_DIR

def smape(actual, predicted):
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    mask = denominator > 0
    return 100 * np.mean(2 * np.abs(predicted[mask] - actual[mask]) / denominator[mask])

def save_metrics(metrics: dict):
    import math
    cleaned = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()}
    path = f"{TABLES_DIR}/metrics.json"
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"Метрики сохранены: {path}")

def plot_predictions(actual, predicted):
    plt.figure(figsize=(12, 4))
    plt.plot(actual.values[:200],  label="Факт")
    plt.plot(predicted[:200],      label="Прогноз")
    plt.title("Факт vs Прогноз (первые 200 точек)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/predictions.png")
    plt.close()
    print("График сохранён: reports/figures/predictions.png")

def plot_feature_importance(model):
    try:
        fi = model.varimp(use_pandas=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=fi.head(15), x="scaled_importance", y="variable")
        plt.title("Feature Importance (Top 15)")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/feature_importance.png")
        plt.close()
        print("График важности признаков сохранён")
    except Exception as e:
        print(f"Feature importance недоступен: {e}")