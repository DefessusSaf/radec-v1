# WFT_19072024.py

import sys
import os
import subprocess
from pathlib import Path # Импортируем pathlib
import astropy.io.fits as pyfits
import astropy.units as _u
from astropy import coordinates
import numpy as np
from src.utils import TicToc # Предполагается, что этот модуль существует и доступен
import path # Импортируем наш модуль path
import logging # Добавим логирование

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Получаем имя файла из аргументов командной строки
if len(sys.argv) != 2:
    print(f"Использование: python {sys.argv[0]} <путь_к_FITS_Файлу>")
    sys.exit(1)

# Преобразуем входной путь в объект Path
input_fits_path = Path(sys.argv[1])

# Убедитесь, что файл существует
if not input_fits_path.exists():
    logging.error(f"Ошибка: Входной файл не существует: {input_fits_path}")
    print(f"Error: {input_fits_path} does not exist")
    sys.exit(1)

# Пути берем из нашего модуля path
# DIR_IN = 'LOAD_FILE' # Удаляем жестко заданный путь
# DIR_OUT = 'PROCESSED' # Удаляем жестко заданный путь
processed_fits_dir = path.PROCESSED_FITS_DIR # Директория для сохранения обработанных FITS
xy_fits_file = path.XY_FITS_FILE # Путь к файлу с XY координатами для solve-field
wcs_file = path.WCS_FILE # Путь к выходному WCS файлу от solve-field
astrometry_config_file = path.ASTROMETRY_CONFIG_FILE # Путь к файлу конфигурации astrometry.net

# Путь для временного файла с WCS заголовком, который создается утилитой new-wcs
# Используем имя исходного файла, но сохраняем во временной директории (TMP_DIR)
temp_wcs_fits_path = path.TMP_DIR / f"WCS_{input_fits_path.name}"

# Путь к конечному обработанному FITS-файлу
# Добавляем префикс "TMP_" к имени исходного файла и сохраняем в директорию PROCESSED_FITS_DIR
output_fits_path = processed_fits_dir / f"TMP_{input_fits_path.name}"

# Чтение заголовка и данных из FITS-файла
try:
    with pyfits.open(input_fits_path) as hdul:
        header = hdul[0].header
        image_data = hdul[0].data
except Exception as e:
    logging.error(f"Ошибка при чтении FITS файла {input_fits_path}: {e}")
    print(f"Error reading FITS file {input_fits_path}: {e}")
    sys.exit(1)


# Извлечение параметров из заголовка
try:
    PIXSIZE = header['XPIXSZ']
    # FOCALLEN = header['FOCALLEN'] # В оригинале FOCALLEN берется из заголовка, но затем переопределяется
    NX = header['NAXIS2']
    NY = header['NAXIS1']
    RA = header['OBJCTRA']
    DEC = header['OBJCTDEC']
except KeyError as e:
    logging.error(f"Отсутствует обязательный ключ в заголовке FITS: {e}")
    print(f"Error: Missing key in FITS header: {e}")
    sys.exit(1)


# Параметры из оригинального скрипта (оставлены как были, если не требуют централизации в path.py)
FOCALLEN_STATIC = 551 # Используется жестко заданное значение FOCALLEN
SCALE_STATIC = 0.08
DOWNSAMPLE = 4
WIDTH = NY # Использование NY для ширины (NAXIS1)
HEIGHT = NX # Использование NX для высоты (NAXIS2)


# Расчет масштаба и радиуса поиска (используется статическое FOCALLEN_STATIC)
# Оригинальный расчет SCALE выглядит не совсем стандартным (умножение на um.to('mm'))
# SCALE = PIXSIZE / FOCALLEN_STATIC * _u.rad.to('arcsec') * _u.um.to('mm')
# Более стандартный расчет масштаба в arcsec/pixel: (PIXSIZE [um] / (FOCALLEN_STATIC [mm] * 1000 [um/mm])) * (180/pi * 3600) [arcsec/rad]
# Или используем astropy.units как в config_setting.py:
# pixel_size = PIXSIZE * _u.micron
# focal_length = FOCALLEN_STATIC * _u.mm
# scale_rad_per_pix = (pixel_size / focal_length).to(_u.rad / _u.pixel)
# scale_arcsec_per_pix = scale_rad_per_pix.to(_u.arcsec / _u.pixel).value
# Используем SCALE_STATIC из оригинального скрипта, если расчет выше не предполагался
# SCALE = SCALE_STATIC # Используем статическое значение
# Или используем более правильный расчет, если это приемлемо:
# Расчет масштаба и радиуса поиска


try:
    calculated_scale = PIXSIZE / FOCALLEN_STATIC * _u.rad.to('arcsec') * _u.um.to('mm')
    logging.info(f"Рассчитанный масштаб на основе заголовка и FOCALLEN_STATIC ({FOCALLEN_STATIC} мм): {calculated_scale:.4f} arcsec/pix")
    # Решаем, какой масштаб использовать: рассчитанный или статический.
    # Если статический SCALE_STATIC=0.08 важен, используем его. Если нет, то calculated_scale.
    # Предположим, что рассчитанный масштаб предпочтительнее, но используем 10% диапазон вокруг него.
    SCALE_FOR_SOLVE = calculated_scale
except Exception as e:
    logging.warning(f"Не удалось рассчитать масштаб: {e}. Используем статический SCALE_STATIC={SCALE_STATIC}")
    SCALE_FOR_SOLVE = SCALE_STATIC


# Расчет диапазона масштабов для solve-field (10% вокруг SCALE_FOR_SOLVE)
# SCALE_LOW = SCALE_FOR_SOLVE * 0.9
# SCALE_HIGH = SCALE_FOR_SOLVE * 1.1
SCALE_LOW = calculated_scale * _u.arcsec.to('deg') * np.min([NX, NY]) / 1.2
SCALE_HIGH = calculated_scale * _u.arcsec.to('deg') * np.max([NX, NY]) * 1.2

# Расчет радиуса поиска (на основе размера изображения и масштаба)
# Оригинальный расчет SEARCH_RAD был SCALE_HIGH, что кажется слишком маленьким (1.1 * масштаб * размер картинки в градусах)
# Более логично использовать размер всего изображения в градусах как минимальный радиус поиска
# Размер изображения по большей стороне в градусах: max(WIDTH, HEIGHT) * SCALE_FOR_SOLVE / 3600
# SEARCH_RAD = max(WIDTH, HEIGHT) * SCALE_FOR_SOLVE / 3600.0 # в градусах
# Или как в оригинале, но с использованием рассчитанного масштаба:
# SEARCH_RAD = SCALE_HIGH # Это значение в arcsecperpix, а не в градусах!
# Команда solve-field принимает --radius в градусах по умолчанию, если не указаны --radius-units
# Исходя из оригинального скрипта, похоже, SCALE_HIGH использовался как радиус в градусах.
# Это нелогично. Давайте использовать размер изображения в градусах как радиус поиска.
# Ширина и высота в градусах: WIDTH * SCALE_FOR_SOLVE / 3600.0, HEIGHT * SCALE_FOR_SOLVE / 3600.0
IMAGE_WIDTH_DEG = WIDTH * SCALE_FOR_SOLVE / 3600.0
IMAGE_HEIGHT_DEG = HEIGHT * SCALE_FOR_SOLVE / 3600.0
# Возьмем половину диагонали изображения как минимальный радиус поиска, умноженную на запас
# SEARCH_RAD_DEG = np.sqrt(IMAGE_WIDTH_DEG**2 + IMAGE_HEIGHT_DEG**2) * 0.7 # Примерный радиус
SEARCH_RAD = SCALE_HIGH

# Преобразуем RA/DEC из заголовка в объект SkyCoord
try:
    COORDS = coordinates.SkyCoord(RA, DEC, unit=('hour', 'deg'), frame='icrs', equinox='J2000')
    logging.info(f"RA (заголовок): {RA}, DEC (заголовок): {DEC}")
    logging.info(f"RA (deg): {COORDS.ra.deg:.5f}, DEC (deg): {COORDS.dec.deg:.5f}")
    RA_DEG = COORDS.ra.deg
    DEC_DEG = COORDS.dec.deg
except Exception as e:
    logging.warning(f"Не удалось преобразовать RA/DEC из заголовка: {e}. Будет использоваться поиск без указания центра.")
    RA_DEG = None
    DEC_DEG = None
    SEARCH_RAD_DEG = None # Если нет центра, нет и радиуса поиска


# Получаем путь к утилите solve-field (предполагается в PATH системы)
# Если solve-field не в PATH, нужно будет добавить его определение в path.py
SOLVE_FIELD_CMD = "solve-field"

# Формирование команды solve-field с обновленными параметрами
cmd_solve = [
    SOLVE_FIELD_CMD,
    "--config", str(astrometry_config_file), # Путь к конфигу
    "--overwrite",
    "--downsample", str(DOWNSAMPLE),
    "--cpulimit", "3600",
    "--no-plots", # Отключаем создание графиков
    "--scale-units", "arcsecperpix",
    "--scale-low", f"{SCALE_LOW:.4f}", # Нижний предел масштаба
    "--scale-high", f"{SCALE_HIGH:.4f}", # Верхний предел масштаба
    "--x-column", "X_CENTER", # Указываем колонки для X и Y
    "--y-column", "Y_CENTER",
    "--sort-column", "AREA", # Колонка для сортировки объектов (из XY.fits)
    "--width", str(WIDTH),   # Размеры изображения
    "--height", str(HEIGHT),
    # Указываем центр и радиус поиска, если координаты из заголовка валидны
    "--ra", f"{RA_DEG}",
    "--dec", f"{DEC_DEG}",
    "--radius", f"{SEARCH_RAD}",
    # *(["--ra", f"{RA_DEG:.5f}", "--dec", f"{DEC_DEG:.5f}", "--radius", f"{SEARCH_RAD:.5f}"] if RA_DEG is not None and SEARCH_RAD_DEG is not None else []),
    str(xy_fits_file) # Путь к входному файлу с XY координатами
]

# Формирование команды solve-field с обновленными параметрами
# cmd_solve = (
#     f"{SOLVE_FIELD_CMD} --config {astrometry_config_file} --overwrite --downsample {DOWNSAMPLE} --cpulimit 3600 --no-plots "
#     f"--scale-units arcsecperpix --scale-low {calculated_scale * 0.9} --scale-high {calculated_scale * 1.1} "
#     f"--x-column X_CENTER --y-column Y_CENTER --sort-column AREA "
#     f"--ra {COORDS.ra.deg} --dec {COORDS.dec.deg} --radius {SEARCH_RAD} "
#     f"--width {WIDTH} --height {HEIGHT} TMP/XY.fits"
    

# )

logging.info(f"Executing solve-field command: {' '.join(map(str, cmd_solve))}")

# Запуск solve-field
try:
    TicToc.tic() # Начало отсчета времени
    # Используем subprocess.run для лучшего контроля вывода и ошибок
    result_solve = subprocess.run(cmd_solve, capture_output=True, text=True, check=False)
    TicToc.toc() # Конец отсчета времени

    if result_solve.returncode != 0:
        logging.error(f"solve-field завершился с ошибкой (код {result_solve.returncode}).")
        logging.error(f"Stderr:\n{result_solve.stderr}")
        raise RuntimeError(f"Ошибка выполнения solve-field.")
    else:
        logging.info("solve-field успешно завершен.")
        # logging.debug(f"Stdout:\n{result_solve.stdout}")
        if result_solve.stderr:
             logging.warning(f"Stderr (Possible warnings from solve-field):\n{result_solve.stderr}")

except FileNotFoundError:
    logging.error("Ошибка: Команда 'solve-field' не найдена. Убедитесь, что astrometry.net установлен и доступен в PATH.")
    print("Error: 'solve-field' command not found. Make sure astrometry.net is installed and in your PATH.")
    sys.exit(1) # Критическая ошибка, без астрометрии дальше нет смысла
except RuntimeError as e:
    logging.error(f"Ошибка при выполнении solve-field: {e}")
    print(f"Error during solve-field execution: {e}")
    sys.exit(1) # Критическая ошибка
except Exception as e:
    logging.exception("Неожиданная ошибка при выполнении solve-field:")
    print(f"An unforeseen error occurred during solve-field: {e}")
    sys.exit(1) # Непредвиденная критическая ошибка


# Проверка создания файла WCS
if not wcs_file.exists() or wcs_file.stat().st_size == 0:
    logging.error(f"Ошибка: solve-field не создал WCS файл по ожидаемому пути: {wcs_file}")
    print(f"Error: solve-field did not create the WCS file at {wcs_file}")
    # Оригинальный скрипт пытался запустить alt_cmd здесь, но он закомментирован.
    # Если WCS файл не создан, дальнейшие шаги бессмысленны.
    sys.exit(1) # Критическая ошибка

logging.info(f"Найден WCS файл: {wcs_file}")


# Обновление WCS в исходном файле с помощью утилиты new-wcs
# Получаем путь к утилите new-wcs из нашего модуля path
new_wcs_cmd_path = path.get_new_wcs_path()

# Формируем команду для запуска new-wcs
cmd_new_wcs = [
    str(new_wcs_cmd_path), # Путь к утилите new-wcs
    "-i", str(input_fits_path), # Входной FITS файл
    "-w", str(wcs_file),      # WCS файл от solve-field
    "-o", str(temp_wcs_fits_path) # Выходной FITS файл с обновленным заголовком (временный)
]

logging.info(f"Executing new-wcs command: {' '.join(map(str, cmd_new_wcs))}")

try:
    # Убедимся, что родительская директория для временного выходного FITS существует
    temp_wcs_fits_path.parent.mkdir(parents=True, exist_ok=True)
    # Запускаем new-wcs
    # Используем subprocess.run для лучшего контроля вывода и ошибок
    result_new_wcs = subprocess.run(cmd_new_wcs, capture_output=True, text=True, check=False)

    if result_new_wcs.returncode != 0:
        logging.error(f"new-wcs завершился с ошибкой (код {result_new_wcs.returncode}).")
        logging.error(f"Stderr:\n{result_new_wcs.stderr}")
        raise RuntimeError(f"Ошибка выполнения new-wcs.")
    else:
        logging.info("new-wcs успешно завершен.")
        # logging.debug(f"Stdout:\n{result_new_wcs.stdout}")
        if result_new_wcs.stderr:
             logging.warning(f"Stderr (Possible warnings from new-wcs):\n{result_new_wcs.stderr}")

except FileNotFoundError:
    logging.error("Ошибка: Команда 'new-wcs' не найдена. Убедитесь, что astrometry.net установлен и доступен через пути, указанные в path.py.")
    print("Error: 'new-wcs' command not found. Make sure astrometry.net is installed and accessible via paths in path.py.")
    sys.exit(1) # Критическая ошибка
except RuntimeError as e:
    logging.error(f"Ошибка при выполнении new-wcs: {e}")
    print(f"Error during new-wcs execution: {e}")
    sys.exit(1) # Критическая ошибка
except Exception as e:
    logging.exception("Неожиданная ошибка при выполнении new-wcs:")
    print(f"An unforeseen error occurred during new-wcs: {e}")
    sys.exit(1) # Непредвиденная критическая ошибка


# Сохранение обновленного FITS-файла с новым WCS
logging.info(f"Сохранение обработанного FITS файла в {output_fits_path}")
try:
    # Читаем заголовок из временного файла, созданного new-wcs
    with pyfits.open(temp_wcs_fits_path) as hdul_wcs:
        header_wcs = hdul_wcs[0].header

    # Создаем новый HDUList с обновленным заголовком и исходными данными изображения
    # Используем pyfits.PrimaryHDU для создания основного HDU
    hdu = pyfits.PrimaryHDU(data=image_data, header=header_wcs)
    hdul_output = pyfits.HDUList([hdu])

    # Убедимся, что директория для выходного файла существует
    output_fits_path.parent.mkdir(parents=True, exist_ok=True)

    # Записываем новый FITS файл
    hdul_output.writeto(str(output_fits_path), output_verify='silentfix', overwrite=True)
    logging.info(f"Обработанный FITS файл успешно сохранен: {output_fits_path}")

except Exception as e:
    logging.error(f'Ошибка при сохранении обработанного FITS файла {output_fits_path}: {e}')
    print(f'Error saving processed FITS file {output_fits_path}: {e}')
    # Не считаем это критической ошибкой для всей программы, но сообщаем о проблеме.
    # sys.exit(1) # Возможно, стоит выйти, если сохранить не удалось?

# Очистка временного файла с WCS заголовком
if temp_wcs_fits_path.exists():
    try:
        temp_wcs_fits_path.unlink() # Удаляем файл
        logging.debug(f"Временный файл удален: {temp_wcs_fits_path}")
    except OSError as e:
        logging.warning(f"Не удалось удалить временный файл {temp_wcs_fits_path}: {e}")

logging.info("Работа скрипта astrometry.py завершена.")

# В этом скрипте нет блока if __name__ == "__main__": main(),
# он просто выполняется сверху вниз при вызове.
# Если нужна функция main, можно обернуть весь код в нее.