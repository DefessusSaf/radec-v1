# path.py
import os
import sys
import logging
import subprocess
from pathlib import Path


# Настройка логирования для path.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Основные директории проекта ---

# ROOT_DIR: Корневая директория проекта (там, где находится path.py)
ROOT_DIR = Path(__file__).resolve().parent.parent
logging.info(f"ROOT_DIR установлен: {ROOT_DIR}")

# SCRIPTS_DIR: Директория со всеми основными Python-скриптами.
# Согласно вашей структуре, скрипты находятся прямо в ROOT_DIR.
SCRIPTS_DIR = Path(__file__).resolve().parent
logging.info(f"SCRIPTS_DIR установлен: {SCRIPTS_DIR}")


# --- Рабочие директории ---

# INPUT_DIR: Директория для входных FITS-файлов
INPUT_DIR = ROOT_DIR / "input"
# Если вы используете LOAD_FILE_DIR как директорию для мониторинга, она должна быть такой же
LOAD_FILE_DIR = INPUT_DIR # Сохраняем это имя для совместимости с monitor.py
logging.info(f"INPUT_DIR / LOAD_FILE_DIR установлен: {LOAD_FILE_DIR}")


# MAIN_DIR: Директория 'main' на верхнем уровне
MAIN_DIR = SCRIPTS_DIR
logging.info(f"MAIN_DIR установлен: {MAIN_DIR}")


# CONFIGS_DIR: Директория с файлами конфигурации Astrometry.net и SExtractor
# Согласно структуре: ASTRO_PIPELINE/main/conf/
CONFIGS_DIR = MAIN_DIR / "conf"
logging.info(f"CONFIGS_DIR установлен: {CONFIGS_DIR}")


# TMP_DIR: Временная директория для промежуточных файлов
# Согласно структуре: ASTRO_PIPELINE/main/src/monitor/tmp/
TMP_DIR = MAIN_DIR / "src" / "tmp"
logging.info(f"TMP_DIR установлен: {TMP_DIR}")


# PROCESSED_FITS_DIR: Директория для обработанных FITS-файлов (с WCS)
# Согласно структуре: ASTRO_PIPELINE/processed/
PROCESSED_FITS_DIR = ROOT_DIR / "processed"
logging.info(f"PROCESSED_FITS_DIR установлен: {PROCESSED_FITS_DIR}")


# PROCESSED_RESULTS_DIR: Директория для выходных текстовых файлов с результатами
# Согласно структуре: ASTRO_PIPELINE/output/
PROCESSED_RESULTS_DIR = ROOT_DIR / "output"
logging.info(f"PROCESSED_RESULTS_DIR установлен: {PROCESSED_RESULTS_DIR}")


# Временная директория, используемая SExtractor для чек-изображений (из config_setting.py)
# Она находится внутри TMP_DIR, так что определим ее относительно TMP_DIR
TEMP_SEXTRACTOR_DIR_NAME = "temp_sextractor_files"
TEMP_SEXTRACTOR_DIR = TMP_DIR / TEMP_SEXTRACTOR_DIR_NAME

# --- Пути к конкретным файлам ---

# Путь к файлу блокировки для monitor.py
LOCK_FILE = TMP_DIR / "monitor_lock"
logging.info(f"LOCK_FILE установлен: {LOCK_FILE}")

# Путь к лог-файлу, который используется для передачи имени файла между скриптами
PROCESSING_LOG_FILE = TMP_DIR / "processing_log.txt"
logging.info(f"PROCESSING_LOG_FILE установлен: {PROCESSING_LOG_FILE}")

# Путь к выходному FITS-файлу SExtractor (с координатами XY)
XY_FITS_FILE = TMP_DIR / "XY.fits"
logging.info(f"XY_FITS_FILE установлен: {XY_FITS_FILE}")

XY_TXT_FILE = TMP_DIR / "XY.txt"
logging.info(f"XY_TXT_FILE установлен: {XY_TXT_FILE}")

# Путь к WCS-файлу, создаваемому solve-field
WCS_FILE = TMP_DIR / "XY.wcs"
logging.info(f"WCS_FILE установлен: {WCS_FILE}")

# Путь к выходному каталогу SExtractor (с подробными параметрами объектов)
SEX_CATALOG_FILE = TMP_DIR / "k1-imp.fts.sx" # Имя из вашей структуры
logging.info(f"SEX_CATALOG_FILE установлен: {SEX_CATALOG_FILE}")

# Путь к файлу региона DS9
REGION_FILE = TMP_DIR / "elik1-imp-field.reg" # Имя из вашей структуры
logging.info(f"REGION_FILE установлен: {REGION_FILE}")


# --- Конфигурационные файлы ---
SEX_CONFIG_FILE = CONFIGS_DIR / "default.sex" 
# Путь к файлу конфигурации Astrometry.net
ASTROMETRY_CONFIG_FILE = CONFIGS_DIR / "astrometry.cfg"
logging.info(f"ASTROMETRY_CONFIG_FILE установлен: {ASTROMETRY_CONFIG_FILE}")


# --- Пути к внешним утилитам ---
# Эти пути могут зависеть от окружения, в котором запускаются скрипты.
# Определим их на основе HOME директории пользователя.
# Пользователю может потребоваться настроить переменные окружения или эти пути.
HOME_DIR = Path.home()

# Путь к утилите text2fits (используется в WriteOutRegionFile)
# Предполагаем, что может использоваться одно из двух окружений
# Возможно, стоит сделать это настраиваемым
TEXT2FITS_BIN_ANACONDA = HOME_DIR / "anaconda3" / "envs" / "radec" / "bin" / "text2fits"
TEXT2FITS_BIN_MINICONDA = HOME_DIR / "miniconda3" / "envs" / "radec" / "bin" / "text2fits"

# Путь к утилите new-wcs (используется в WFT)
NEW_WCS_BIN_ANACONDA = HOME_DIR / "anaconda3" / "envs" / "radec" / "bin" / "new-wcs"
NEW_WCS_BIN_MINICONDA = HOME_DIR / "miniconda3" / "envs" / "radec" / "bin" / "new-wcs"

# Функция для выбора правильного пути к утилите (Anaconda/Miniconda)
# Можно добавить логику проверки существования файлов
def get_text2fits_path():
    if TEXT2FITS_BIN_ANACONDA.exists():
        return TEXT2FITS_BIN_ANACONDA
    elif TEXT2FITS_BIN_MINICONDA.exists():
        return TEXT2FITS_BIN_MINICONDA
    else:
        # Возможно, стоит вызвать ошибку или вернуть None и обработать это в вызывающем скрипте
        print("Warning: Neither Anaconda nor Miniconda text2fits path found. Check your environment.")
        return Path("text2fits") # Попробуем просто имя, если нет полного пути

def get_new_wcs_path():
    if NEW_WCS_BIN_ANACONDA.exists():
        return NEW_WCS_BIN_ANACONDA
    elif NEW_WCS_BIN_MINICONDA.exists():
        return NEW_WCS_BIN_MINICONDA
    else:
        print("Warning: Neither Anaconda nor Miniconda new-wcs path found. Check your environment.")
        return Path("new-wcs") # Попробуем просто имя



# --- Пути к внешним исполняемым файлам (например, new-wcs) ---
# Эти пути могут зависеть от вашей установки astrometry.net и активации conda-окружения.
# Функция пытается найти new-wcs в разных окружениях.

# def get_new_wcs_path():
#     """
#     Определяет и возвращает путь к исполняемому файлу 'new-wcs'.
#     Ищет в стандартных местах установки conda/miniconda окружений.
#     """
#     # Проверяем, активно ли какое-либо conda-окружение и где находится new-wcs в нем
#     conda_env_path = os.environ.get('CONDA_PREFIX')
#     if conda_env_path:
#         new_wcs_in_conda = Path(conda_env_path) / "bin" / "new-wcs"
#         if new_wcs_in_conda.exists():
#             logging.info(f"Найден new-wcs в активном conda-окружении: {new_wcs_in_conda}")
#             return new_wcs_in_conda

#     # Если не найдено в активном окружении, или окружение не активно,
#     # ищем в известных местах установки
#     home_dir = Path.home()
#     possible_paths = [
#         home_dir / "anaconda3" / "envs" / "radec" / "bin" / "new-wcs",
#         home_dir / "miniconda3" / "envs" / "radec" / "bin" / "new-wcs",
#         # Добавьте другие возможные пути, если у вас нестандартная установка
#     ]

#     for p in possible_paths:
#         if p.exists():
#             logging.info(f"Найден new-wcs по пути: {p}")
#             return p

#     # Если new-wcs не найден по известным путям, попробуем найти его в системном PATH
#     # (если он добавлен туда)
#     new_wcs_from_path = "new-wcs"
#     # Проверяем, доступен ли он в PATH
#     if subprocess.run(["which", new_wcs_from_path], capture_output=True, text=True).returncode == 0:
#         logging.info(f"new-wcs найден в системном PATH.")
#         return Path(new_wcs_from_path) # Возвращаем имя команды, чтобы subprocess.run нашел ее в PATH

#     logging.error("Не удалось найти исполняемый файл 'new-wcs'. Убедитесь, что astrometry.net установлен и 'new-wcs' доступен в PATH или по указанным путям.")
#     # Возвращаем имя команды, чтобы subprocess.run попробовал найти ее в PATH,
#     # но будет ошибка, если ее там нет.
#     return Path("new-wcs")

# # Присваиваем результат функции для использования в других скриптах
# ASTROMETRY_NEW_WCS_PATH = get_new_wcs_path()

