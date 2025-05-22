# monitor.py

import sys
import time
import subprocess
import logging
import shutil # Для очистки директории, если потребуется
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from filelock import FileLock, Timeout # Установите: pip install filelock

# Импортируем наш модуль с путями
import path

# --- Функции обработки файлов ---

def process_file(file_path: Path):
    """
    It processes one fits file, letting it through the entire conveyor of the Python scripts
And external utilities.The entire conclusion of the offenses is logged in.

    Args:
        file_path (Path): The full path to the file that you need to process (object is Path).
    """
    logging.info(f"\n--- We start processing the file: {file_path.name} ---")

    try:
        # 1. Записываем имя файла в лог для отслеживания (используется radec скриптами)
        logging.info(f"Record the name of the file in {path.PROCESSING_LOG_FILE}")
        path.PROCESSING_LOG_FILE.write_text(str(file_path))

        # Функция для запуска субпроцесса и логирования его вывода
        def run_and_log_subprocess(command_list, description):
            logging.info(f'===========  Launch {description} ===========')
            logging.info(f"Command: {' '.join(map(str, command_list))}") # Логируем команду
            result = subprocess.run(command_list, capture_output=True, text=True, check=True, encoding='utf-8')
            if result.stdout:
                logging.info(f"STDOUT {description}:\n{result.stdout.strip()}")
            if result.stderr:
                logging.warning(f"STDERR {description}:\n{result.stderr.strip()}")
            logging.info(f'{description} Completed.')

        # 2. Запуск config_setting.py
        run_and_log_subprocess([sys.executable, str(SCRIPTS_DIR / "config_setting.py"), str(file_path)], "config_setting.py")

        # 3. Запуск WriteOutRegionFile_18072024.py
        run_and_log_subprocess([sys.executable, str(SCRIPTS_DIR / "region.py")], "region.py")

        # 4. Запуск WFT_19072024.py
        run_and_log_subprocess([sys.executable, str(SCRIPTS_DIR / "astrometry.py"), str(file_path)], "astrometry.py")

        # 5. Запуск radec_without_mode.py
        run_and_log_subprocess([sys.executable, str(SCRIPTS_DIR / "radec_without_mode.py")], "radec_without_mode.py")

        # 6. Проверка существования XY.fits (если он критичен для дальнейшей работы)
        if not path.XY_FITS_FILE.exists() or path.XY_FITS_FILE.stat().st_size == 0:
            logging.error(f"Error: File {path.XY_FITS_FILE} was not created or empty after processing.")
            raise RuntimeError(f"Absent or empty file {path.XY_FITS_FILE}")

        # 7. Удаление исходного файла после успешной обработки
        logging.info(f"Removing the processed file: {file_path.name}")
        file_path.unlink()

        # 8. Очистка лога обработки
        logging.info(f"File cleaning {path.PROCESSING_LOG_FILE}")
        path.PROCESSING_LOG_FILE.write_text("")

        logging.info(f"--- File processing {file_path.name} completed successfully ---\n")

    except subprocess.CalledProcessError as e:
        logging.error(f"An error of the script/team: {e.cmd}")
        logging.error(f"Return code: {e.returncode}")
        if e.stdout:
            logging.error(f"STDOUT (error):\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"STDERR (error):\n{e.stderr.strip()}")
        logging.error(f"File processing {file_path.name} It ended with an error.We miss.\n")
    except FileNotFoundError as e:
        logging.error(f"Error: NOT File: {e}")
        logging.error(f"File processing {file_path.name} It ended with an error.We miss.\n")
    except RuntimeError as e:
        logging.error(f"Processing error: {e}")
        logging.error(f"File processing {file_path.name} It ended with an error.We miss.\n")
    except Exception as e:
        logging.exception(f"Unexpected error when processing a file {file_path.name}:")
        logging.error(f"File processing {file_path.name} It ended with an error.We miss.\n")
    finally:
        # Убедимся, что лог-файл всегда очищается, если обработка не завершилась успешно до удаления файла
        if path.PROCESSING_LOG_FILE.exists() and path.PROCESSING_LOG_FILE.read_text() == str(file_path):
             logging.warning(f"Log File {path.PROCESSING_LOG_FILE} I was not cleared after the error, I clean.")
             path.PROCESSING_LOG_FILE.write_text("")

class NewFileHandler(FileSystemEventHandler):
    """
    File system events responding to creating new files.
    """
    def __init__(self, processing_function, lock_file_path):
        super().__init__()
        self.processing_function = processing_function
        self.lock_file_path = lock_file_path
        self.lock = FileLock(str(lock_file_path), timeout=1) # Таймаут для лок-файла

    def on_created(self, event):
        """
        It is called when the file or directory is created.
        Processes only files, ignoring the directory.
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        logging.info(f"A new file has been discovered: {file_path.name}")

        # Попытка получить блокировку
        try:
            with self.lock:
                logging.info(f"Blocking is obtained for {file_path.name}")
                self.processing_function(file_path)
                logging.info(f"The lock is removed for {file_path.name}")
        except Timeout:
            logging.warning(f"Failed to get a lock for {file_path.name}. The processing process has already been launched. We miss.")
        except Exception as e:
            logging.error(f"Error when working with blocking for {file_path.name}: {e}")

# Получаем путь к директории, где находится этот скрипт (ASTRO_PIPELINE/)
SCRIPTS_DIR = Path(__file__).resolve().parent

# --- Основная функция монитора ---
def main():
    # --- Настройка логирования (перенесено сюда) ---
    LOG_DIR = path.MAIN_DIR / "log"
    LOG_FILE_PATH = LOG_DIR / "monitor.log"

    # Убедимся, что директория для логов существует
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Очищаем лог-файл при каждом новом запуске monitor.py
    if LOG_FILE_PATH.exists():
        try:
            LOG_FILE_PATH.unlink() # Удаляем старый лог-файл
        except Exception as e:
            # Если не удалось удалить старый файл (например, он используется),
            # мы просто выводим сообщение об ошибке в консоль, так как логирование еще не настроено
            print(f"Error: failed to clean the old log-file {LOG_FILE_PATH}: {e}", file=sys.stderr)


    # Очистка всех предыдущих обработчиков логирования
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    

    # Настраиваем логирование:
    # 1. Запись в файл (добавляем новые записи, не перезаписываем в течение одного сеанса)
    # 2. Вывод в консоль
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(LOG_FILE_PATH, mode='a'), # mode='a' по умолчанию
                            logging.StreamHandler(sys.stdout)
                        ])

    # Теперь, когда логирование настроено, мы можем логировать статус очистки файла
    logging.info(f"Log File {LOG_FILE_PATH} Cleaned/created at launch.")


    # --- Остальная часть логики основной функции (создание директорий, настройка наблюдателя, цикл) ---

    # Убедимся, что директории существуют
    path.LOAD_FILE_DIR.mkdir(parents=True, exist_ok=True)
    path.TMP_DIR.mkdir(parents=True, exist_ok=True)
    path.PROCESSED_FITS_DIR.mkdir(parents=True, exist_ok=True)
    path.PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    # Директория логов уже создана выше

    logging.info("All the necessary project directors have been tested/created.")

    event_handler = NewFileHandler(process_file, path.LOCK_FILE)
    observer = Observer()
    observer.schedule(event_handler, str(path.LOAD_FILE_DIR), recursive=False) # Мониторим только саму директорию

    logging.info(f"--- Directorate of monitoring of directory: {path.LOAD_FILE_DIR} ---")
    logging.info(f"The temporary directory is used: {path.TMP_DIR}")
    logging.info(f"Lock file is used: {path.LOCK_FILE}")

    observer.start()

    try:
        while True:
            time.sleep(1) # Ждем 1 секунду, чтобы не загружать CPU
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Monitoring is stopped by the user.")
    except Exception as e:
        logging.exception("An unforeseen error of the main monitoring cycle occurred:")
    finally:
        observer.join()
        logging.info("The monitor completed the work.")

    logging.info("test log file after start monitor.py")

# Точка входа в скрипт:
if __name__ == "__main__":
    main() # ВАЖНО: теперь main() вызывается!