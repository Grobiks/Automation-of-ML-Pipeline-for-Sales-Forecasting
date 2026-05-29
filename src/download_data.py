# src/download_data.py
import os
import zipfile
from src.config import RAW_DIR

COMPETITION = "playground-series-s5e1"

def download():
    print("Грузим данные с Kaggle...")
    os.system(f"kaggle competitions download -c {COMPETITION} -p {RAW_DIR}")

    zip_path = os.path.join(RAW_DIR, f"{COMPETITION}.zip")
    if os.path.exists(zip_path):
        print("Распаковываем архив...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        os.remove(zip_path)
        print("Гуд. Файлы в папке data/raw/")
    else:
        print("Архив не найден — возможно данные уже скачаны или ошибка авторизации")

if __name__ == "__main__":
    download()