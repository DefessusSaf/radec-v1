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
    Обрабатывает один FITS-файл, пропуская его через весь конвейер Python-скриптов
    и внешних утилит. Весь вывод субпроцессов логируется.

    Args:
        file_path (Path): Полный путь к файлу, который нужно обработать (объект Path).
    """
    logging.info(f"\n--- Начинаем обработку файла: {file_path.name} ---")

    try:
        # 1. Записываем имя файла в лог для отслеживания (используется radec скриптами)
        logging.info(f"Записываем имя файла в {path.PROCESSING_LOG_FILE}")
        path.PROCESSING_LOG_FILE.write_text(str(file_path))

        # Функция для запуска субпроцесса и логирования его вывода
        def run_and_log_subprocess(command_list, description):
            logging.info(f'===========  Запуск {description} ===========')
            logging.info(f"Команда: {' '.join(map(str, command_list))}") # Логируем команду
            result = subprocess.run(command_list, capture_output=True, text=True, check=True, encoding='utf-8')
            if result.stdout:
                logging.info(f"STDOUT {description}:\n{result.stdout.strip()}")
            if result.stderr:
                logging.warning(f"STDERR {description}:\n{result.stderr.strip()}")
            logging.info(f'{description} завершен.')

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
            logging.error(f"Ошибка: Файл {path.XY_FITS_FILE} не был создан или пуст после обработки.")
            raise RuntimeError(f"Отсутствует или пустой файл {path.XY_FITS_FILE}")

        # 7. Удаление исходного файла после успешной обработки
        logging.info(f"Удаление обработанного файла: {file_path.name}")
        file_path.unlink()

        # 8. Очистка лога обработки
        logging.info(f"Очистка файла {path.PROCESSING_LOG_FILE}")
        path.PROCESSING_LOG_FILE.write_text("")

        logging.info(f"--- Обработка файла {file_path.name} завершена успешно ---\n")

    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка выполнения скрипта/команды: {e.cmd}")
        logging.error(f"Код возврата: {e.returncode}")
        if e.stdout:
            logging.error(f"STDOUT (ошибка):\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"STDERR (ошибка):\n{e.stderr.strip()}")
        logging.error(f"Обработка файла {file_path.name} завершилась с ошибкой. Пропускаем.\n")
    except FileNotFoundError as e:
        logging.error(f"Ошибка: Не найден файл или команда: {e}")
        logging.error(f"Обработка файла {file_path.name} завершилась с ошибкой. Пропускаем.\n")
    except RuntimeError as e:
        logging.error(f"Ошибка обработки: {e}")
        logging.error(f"Обработка файла {file_path.name} завершилась с ошибкой. Пропускаем.\n")
    except Exception as e:
        logging.exception(f"Неожиданная ошибка при обработке файла {file_path.name}:")
        logging.error(f"Обработка файла {file_path.name} завершилась с ошибкой. Пропускаем.\n")
    finally:
        # Убедимся, что лог-файл всегда очищается, если обработка не завершилась успешно до удаления файла
        if path.PROCESSING_LOG_FILE.exists() and path.PROCESSING_LOG_FILE.read_text() == str(file_path):
             logging.warning(f"Лог-файл {path.PROCESSING_LOG_FILE} не был очищен после ошибки, очищаю.")
             path.PROCESSING_LOG_FILE.write_text("")

class NewFileHandler(FileSystemEventHandler):
    """
    Обработчик событий файловой системы, реагирующий на создание новых файлов.
    """
    def __init__(self, processing_function, lock_file_path):
        super().__init__()
        self.processing_function = processing_function
        self.lock_file_path = lock_file_path
        self.lock = FileLock(str(lock_file_path), timeout=1) # Таймаут для лок-файла

    def on_created(self, event):
        """
        Вызывается, когда файл или директория созданы.
        Обрабатывает только файлы, игнорируя директории.
        """
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        logging.info(f"Обнаружен новый файл: {file_path.name}")

        # Попытка получить блокировку
        try:
            with self.lock:
                logging.info(f"Блокировка получена для {file_path.name}")
                self.processing_function(file_path)
                logging.info(f"Блокировка снята для {file_path.name}")
        except Timeout:
            logging.warning(f"Не удалось получить блокировку для {file_path.name}. Процесс обработки уже запущен. Пропускаем.")
        except Exception as e:
            logging.error(f"Ошибка при работе с блокировкой для {file_path.name}: {e}")

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
            print(f"ОШИБКА: Не удалось очистить старый лог-файл {LOG_FILE_PATH}: {e}", file=sys.stderr)


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
    logging.info(f"Лог-файл {LOG_FILE_PATH} очищен/создан при запуске.")


    # --- Остальная часть логики основной функции (создание директорий, настройка наблюдателя, цикл) ---

    # Убедимся, что директории существуют
    path.LOAD_FILE_DIR.mkdir(parents=True, exist_ok=True)
    path.TMP_DIR.mkdir(parents=True, exist_ok=True)
    path.PROCESSED_FITS_DIR.mkdir(parents=True, exist_ok=True)
    path.PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    # Директория логов уже создана выше

    logging.info("Все необходимые директории проекта проверены/созданы.")

    event_handler = NewFileHandler(process_file, path.LOCK_FILE)
    observer = Observer()
    observer.schedule(event_handler, str(path.LOAD_FILE_DIR), recursive=False) # Мониторим только саму директорию

    logging.info(f"--- Запуск мониторинга директории: {path.LOAD_FILE_DIR} ---")
    logging.info(f"Используется временная директория: {path.TMP_DIR}")
    logging.info(f"Используется лок-файл: {path.LOCK_FILE}")

    observer.start()

    try:
        while True:
            time.sleep(1) # Ждем 1 секунду, чтобы не загружать CPU
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Мониторинг остановлен пользователем.")
    except Exception as e:
        logging.exception("Произошла непредвиденная ошибка в основном цикле мониторинга:")
    finally:
        observer.join()
        logging.info("Монитор завершил работу.")

    logging.info("test log file after start monitor.py")

# Точка входа в скрипт:
if __name__ == "__main__":
    main() # ВАЖНО: теперь main() вызывается!