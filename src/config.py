import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Папки с данными ──────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(ROOT_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")

# ─── Пути к исходным файлам Kaggle ───────────────────────────────────────────
TRAIN_PATH      = os.path.join(RAW_DIR, "train.csv")
TEST_PATH       = os.path.join(RAW_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(RAW_DIR, "sample_submission.csv")

# ─── Пути к обработанным данным ───────────────────────────────────────────────
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "train_processed.parquet")
PROCESSED_TEST_PATH  = os.path.join(PROCESSED_DIR, "test_processed.parquet")

# ─── Папки для результатов ────────────────────────────────────────────────────
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR  = os.path.join(REPORTS_DIR, "tables")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")

# ─── Параметры датасета ───────────────────────────────────────────────────────
TARGET_COL = "num_sold"      # колонка которую предсказываем (количество продаж)
DATE_COL   = "date"          # колонка с датой
ID_COL     = "id"            # колонка с id строки

# Категориальные признаки — текстовые колонки
CAT_COLS = ["country", "store", "product"]

# ─── Параметры H2O AutoML ─────────────────────────────────────────────────────
H2O_MAX_MODELS       = 20    # сколько моделей попробует H2O
H2O_MAX_RUNTIME_SECS = 1800  # максимальное время обучения (30 минут)
H2O_SEED             = 42    # для воспроизводимости результатов

# ─── Параметры разбивки данных ────────────────────────────────────────────────
# Последние 20% данных по времени — валидация (важно для временных рядов!)
VALIDATION_SPLIT = 0.2

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "sticker-sales-forecasting"

# ─── Создаём все нужные папки автоматически при импорте config ───────────────
for _dir in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)