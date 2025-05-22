#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import fcntl
from pathlib import Path
from subprocess import run, CalledProcessError
import logging
import sys

# === Пути ===
BASE_DIR     = Path(__file__).parent.resolve()
MONITOR_DIR  = BASE_DIR / "LOAD_FILE"
TMP_DIR      = BASE_DIR / "TMP"
LOG_DIR      = BASE_DIR / "log"
LOCKFILE     = TMP_DIR / "process.lock"
LOGFILE      = LOG_DIR / "monitor.log"
PROC_LOGFILE = TMP_DIR / "processing_log.txt"

# Создаём нужные каталоги до настройки логирования
for d in (MONITOR_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# === Настройка логирования ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Файловый хендлер
fh = logging.FileHandler(LOGFILE, encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

# Консольный хендлер
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
logger.addHandler(ch)


def process_file(path: Path):
    # Эмулируем echo $NEWFILE >> TMP/processing_log.txt
    with PROC_LOGFILE.open("w", encoding="utf-8") as f:
        f.write(str(path) + "\n")

    logging.info(f"=== Start processing {path.name} ===")
    with LOGFILE.open("a", encoding="utf-8") as log_fh:
        scripts = [
            # Скрипты, которым передаём имя FITS как аргумент
            (BASE_DIR / "config_setting.py", True),
            (BASE_DIR / "SRC" / "WriteOutRegionFile_18072024.py", False),
            (BASE_DIR / "SRC" / "WFT_19072024.py", True),
            # (BASE_DIR / "radec_without_mode.py", False),
            (BASE_DIR / "radec_StarObservation.py", False),
        ]

        for script_path, needs_arg in scripts:
            cmd = [sys.executable, str(script_path)]
            if needs_arg:
                cmd.append(str(path))
            cmd_str = " ".join(cmd)
            logging.info(f"Running: {cmd_str}")
            try:
                run(cmd, check=True, stdout=log_fh, stderr=log_fh)
            except CalledProcessError as e:
                logging.error(f"Step failed ({script_path.name}): return code {e.returncode}")
                continue

    logging.info(f"=== Finished processing {path.name} ===\n")

    # Удаляем исходник
    try:
        path.unlink()
        logging.info(f"Deleted source file {path.name}")
    except Exception as e:
        logging.warning(f"Could not delete {path.name}: {e}")


def monitor_loop():
    processed = set()
    logging.info(f"Monitoring directory {MONITOR_DIR}")
    while True:
        for fits_file in MONITOR_DIR.glob("*.fits"):
            if fits_file not in processed:
                with open(LOCKFILE, "w") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    process_file(fits_file)
                    fcntl.flock(lf, fcntl.LOCK_UN)
                processed.add(fits_file)
        time.sleep(1)


if __name__ == "__main__":
    try:
        monitor_loop()
    except KeyboardInterrupt:
        logging.info("Monitor stopped by user")