import time
import csv
import psutil
from src.config import TABLES_DIR

def monitor(duration_sec=30, interval_sec=2):
    path = f"{TABLES_DIR}/resource_monitoring.csv"
    print(f"Мониторинг ресурсов {duration_sec} сек → {path}")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cpu_percent", "ram_percent", "ram_used_mb"])

        for _ in range(duration_sec // interval_sec):
            row = [
                time.strftime("%H:%M:%S"),
                psutil.cpu_percent(interval=interval_sec),
                psutil.virtual_memory().percent,
                round(psutil.virtual_memory().used / 1024 / 1024, 1),
            ]
            writer.writerow(row)
            print(f"  {row}")

    print("Мониторинг завершён")

if __name__ == "__main__":
    monitor()