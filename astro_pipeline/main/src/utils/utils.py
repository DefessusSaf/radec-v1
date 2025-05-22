# utils.py

import numpy as np
import os
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy.table import QTable
import astropy.units as u
from astropy.time import Time
from datetime import timedelta
import logging 


# Настройка базового логирования, можно переопределить в основных скриптах
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_outlier(points, thresh=3.5):
    """
    Determines emissions in the data set using a modified Z-account.

    Args:
        points (np.ndarray): Input array of data.
        thresh (float): Threshold for determining emissions.

    Returns:
        np.ndarray: Array of emissions indices.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sqrt(np.sum((points - median) ** 2, axis=-1))
    med_abs_deviation = np.median(diff)

    if med_abs_deviation == 0:
        return np.array([], dtype=int)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    outlier_indices = np.where(modified_z_score > thresh)[0]
    # logging.debug(f"is_outlier found {len(outlier_indices)} outliers.")
    return outlier_indices


def load_sextractor_genfromtxt(file_path):
    """
    It downloads the SExtractor catalog from a text file using NP.genfromtxt.
    It implies the standard order of column from your scripts.

    Args:
        file_path (str): The path to the catalog file.

    Returns:
        Tuple: Numpy arrays for X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, TH, FLAG, FLUX.
    Raises:
        FileNotFoundError: If the file is not found.
        IOError: If an error occurs when reading a file.
    """
    if not os.path.exists(file_path):
        logging.error(f"The catalog file was not found: {file_path}")
        raise FileNotFoundError(f"The catalog file was not found: {file_path}")

    try:
        # Используем unpack=True для автоматического разделения столбцов
        # Явно указываем dtype=float, чтобы избежать ошибок при наличии нечисловых данных
        data = np.genfromtxt(file_path, unpack=True, dtype=float)
        # Проверяем, что количество столбцов соответствует ожидаемому
        expected_cols = 13 # X,Y,ERRX,ERRY,A,B,XMIN,YMIN,XMAX,YMAX,TH,FLAG,FLUX
        if data.shape[0] != expected_cols:
             logging.warning(f"Expected {expected_cols} columns in the file {file_path}, Found {data.shape[0]}. Format error is possible.")
             if data.shape[0] > expected_cols:
                 data = data[:expected_cols, :]
             else:
                 raise IOError(f"The wrong number of columns ({data.shape[0]} instead of{expected_cols}) In the file {file_path}.")


        X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, TH, FLAG, FLUX = data
        # logging.info(f"Загружено {len(X)} объектов из {file_path}")
        return X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, TH, FLAG, FLUX
    except Exception as e:
        logging.error(f"Error when downloading data from a file {file_path}: {e}")
        raise IOError(f"Error when downloading data from a file {file_path}: {e}")



def choose_fits_file(log_file_path, fits_dirs):
    """
    Selects the last processed FITS file based on the contents of the log-file.

    Args:
        log_file_path (str): The path to the log-file with the name of the last processed file.
        fits_dirs (list): List of directory in which look for fits.

    Returns:
        Tuple[str, str]: Cortege (path to fits file, basic name of the source file).
    Raises:
        FileNotFoundError: If the log file is not found or the corresponding fits file was not found.
        ValueError: If the log-file is empty.
    """
    if not os.path.exists(log_file_path):
        logging.error(f"The log-file does not exist: {log_file_path}")
        raise FileNotFoundError(f"The log-file does not exist: {log_file_path}")

    try:
        with open(log_file_path, 'r') as f:
            last_file = f.readline().strip()
    except Exception as e:
        logging.error(f"Error when reading a log-file {log_file_path}: {e}")
        raise IOError(f"Error when reading a log-file {log_file_path}: {e}")


    if not last_file:
        logging.error(f"In the log-file {log_file_path} no file name was found.")
        raise ValueError(f"In the log-file {log_file_path} no file name was found.")

    last_file_base = os.path.basename(last_file)
    # Извлекаем суффикс (все символы до расширения)
    last_file_suffix = last_file_base.split('.')[0]
    logging.debug(f"Eliminated suffix from Log: {last_file_suffix}")

    all_fits_files = []
    for fits_dir in fits_dirs:
        if os.path.isdir(fits_dir):
            all_fits_files.extend(glob.glob(os.path.join(fits_dir, '*.fits')))
            all_fits_files.extend(glob.glob(os.path.join(fits_dir, '*.fit')))
        else:
            logging.warning(f"Directory requested FITS Not faound: {fits_dir}")

    # logging.debug(f"Найденные FITS-файлы: {all_fits_files}")

    matching_file = None
    for fits_file in all_fits_files:
        file_name = os.path.basename(fits_file)
        # Проверка наличия суффикса в имени файла
        if last_file_suffix in file_name:
            matching_file = fits_file
            break

    if not matching_file:
        logging.error(f"No appropriate fits file for the suffix was found '{last_file_suffix}' in directors {fits_dirs}")
        raise FileNotFoundError(f"No appropriate fits file for the suffix was found '{last_file_suffix}'")

    logging.info(f"The fits file is selected: {matching_file}")
    return matching_file, last_file_base


# # Функция предварительной обработки данных (фильтрация по координатам), специфичная для этого скрипта (использует QTable)
# # Оригинальная функция preprocess_data из radec_without_mode.py
def preprocess_data(table, x_min, x_max, y_min, y_max):
    """Preliminary data processing using QTable."""
    y_mask = (table['Y'] >= y_min) & (table['Y'] <= y_max)
    x_mask = (table['X'] >= x_min) & (table['X'] <= x_max)
    mask = x_mask & y_mask
    return table[mask]


def compute_elongation(A, B):
    """
    Calculates the elongation (A/B ratio) for each object.

    Args:
        A (np.ndarray)
        B (np.ndarray)

    Returns:
        np.ndarray: An array of the values ​​of the elongation.
    """
    elongation = np.zeros_like(A)
    non_zero_b_mask = B != 0
    elongation[non_zero_b_mask] = A[non_zero_b_mask] / B[non_zero_b_mask]
    # logging.debug("Computed elongation.")
    return elongation

def compute_errores(ERRX, ERRY):
    """
    Calculates errors on X and Y based on Errxx_image and Erryy_image.

    Args:
        ERRX (np.ndarray)
        ERRY (np.ndarray)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Кортеж (массив ошибок по X, массив ошибок по Y).
    """
    erroreX = np.zeros_like(ERRX)
    erroreY = np.zeros_like(ERRY)

    valid_errx_mask = ERRX > 0
    valid_erry_mask = ERRY > 0

    erroreX[valid_errx_mask] = np.sqrt(1/ERRX[valid_errx_mask])
    erroreY[valid_erry_mask] = np.sqrt(1/ERRY[valid_erry_mask])

    # logging.debug("Computed errors.")
    return erroreX, erroreY


def convert_deg_to_hmsdms(ra_deg, dec_deg):
    """
    Converts the coordinates from degrees to the format: HH:MM:SS and YY:MM:CC.

    Args:
        ra_deg (float or np.ndarray): Direct ascent in degrees.
        dec_deg (float or np.ndarray): Declension in degrees.

    Returns:
        Tuple[str or list, str or list].
    """
    ra_angle = Angle(ra_deg, unit=u.deg)
    ra_hms = ra_angle.to_string(unit=u.hour, sep=':', precision=2, pad=True) # pad=True для выравнивания нулями

    dec_angle = Angle(dec_deg, unit=u.deg)
    dec_dms = dec_angle.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True, pad=True) # alwayssign=True и pad=True

    # logging.debug(f"Converted {ra_deg}, {dec_deg} to {ra_hms}, {dec_dms}")

    if isinstance(ra_deg, np.ndarray):
        return list(ra_hms), list(dec_dms)
    else:
        return ra_hms, dec_dms

# Возможно, функция для сохранения результатов тоже может быть универсальной,
# но пока они различаются (один или два кластера), оставим их в скриптах.
# def save_results(...):
#     pass