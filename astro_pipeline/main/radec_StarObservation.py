# Star observation mode
"""
@author: saf

Detection object and convert pixcoordinates on RA and DEC

"""

import os
import glob
import sys
import numpy as np
import astropy.units as u
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import QTable
from astropy.coordinates import Angle
from astropy.time import Time
from datetime import timedelta
from src.utils import path
from mpl_toolkits.mplot3d import Axes3D
from src.utils import utils



def save_results(coords, fits_filename, base_filename, X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX):
    output_dir = path.PROCESSED_RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    txt_filename = os.path.join(output_dir, f'{base_filename}.txt')
    with open(txt_filename, 'w') as f:
        f.write(f"File: {base_filename}\n")
        with fits.open(fits_filename) as hdul:
            header = hdul[0].header
            date_obs = header.get('DATE-OBS', '00000')
            exptime = header.get('EXPTIME', 0)
            # Вычисление среднего времени экспозиции, если EXPTIME задано
            if date_obs != '00000' and exptime > 0:
                # Извлекаем дату и время
                date_str, time_str = date_obs.split('T')
                time_obs = Time(f'{date_str} {time_str}', format='iso')
                avg_exposure_time = time_obs + timedelta(seconds=(exptime / 2.0))
                new_time_str = avg_exposure_time.iso.replace(" ", "T")  
                f.write(f"{new_time_str}\n")
            else:
                f.write(f"{date_obs}\n")
                
        with fits.open(fits_filename) as hdul:
            wcs = WCS(hdul[0].header)
        sky_coords = wcs.pixel_to_world(X, Y)
        for coord, x, y, errx, erry, a, b, xmin, ymin, xmax, ymax in zip(
                sky_coords, X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX):
            ra_deg = coord.ra.deg
            dec_deg = coord.dec.deg
            ra_hms, dec_dms = utils.convert_deg_to_hmsdms(ra_deg, dec_deg)
            f.write(f"{ra_hms} {dec_dms} {x} {y} {errx} {erry} {a} {b} {xmin} {ymin} {xmax} {ymax}\n")


def star_observation(X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, fits_filename, base_filename):
    ELONG = utils.compute_elongation(A, B)

    # Фильтрация по ELONG > 2.2
    elong_mask = ELONG > 3.10
    X_filtered = X[elong_mask]
    Y_filtered = Y[elong_mask]
    ELONG_filtered = ELONG[elong_mask]

    if len(X_filtered) == 0:
        print("No data passed the ELONG filter.")
        return

    erroreX, erroreY = utils.compute_errores(ERRX, ERRY)
    print(f"Filtered X: {X_filtered}, Y: {Y_filtered}")

    with fits.open(fits_filename) as hdul:
        wcs = WCS(hdul[0].header)

    try:
        sky_coords = wcs.pixel_to_world(X_filtered, Y_filtered)
    except Exception as e:
        print(f"WCS conversion error: {e}")
        return

    coords_filtered = []
    for coord in sky_coords:
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg
        ra_hms, dec_dms = utils.convert_deg_to_hmsdms(ra_deg, dec_deg)
        coords_filtered.append((ra_hms, dec_dms))
        print(f"RA: {ra_hms}, DEC: {dec_deg}")

    save_results(coords_filtered, fits_filename, base_filename, 
                 X_filtered, Y_filtered, erroreX[elong_mask], erroreY[elong_mask], 
                 A[elong_mask], B[elong_mask], XMIN[elong_mask], YMIN[elong_mask], 
                 XMAX[elong_mask], YMAX[elong_mask])

    return coords_filtered


def main():
    DIR = path.TMP_DIR
    fn = path.SEX_CATALOG_FILE
    # try:
    #     fits_filename, base_filename = utils.choose_fits_file()
    # except (FileNotFoundError, ValueError) as e:
    #     print(e)
    #     return
    try:
        # Используем choose_fits_file из utils, передавая путь к логу и директории поиска FITS
        # Ищем FITS файл в TMP_DIR и PROCESSED_FITS_DIR (на всякий случай)
        fits_filename, base_filename = utils.choose_fits_file(str(path.PROCESSING_LOG_FILE), [str(path.TMP_DIR), str(path.PROCESSED_FITS_DIR)])
        fits_file_path = Path(fits_filename) # Преобразуем результат в Path объект
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error when choosing a FITS file: {e}")
        logging.error(f"Error when choosing a FITS file: {e}")
        sys.exit(1) # Критическая ошибка, не можем продолжить без FITS файла

    # try:
    #     data_table = utils.load_sextractor_genfromtxt(f'{fn}')
    # except Exception as e:
    #     print(f"Data boot error: {e}")
    #     return

    try:
        # Используем локальную load_data, которая читает в QTable
        data_table = utils.load_sextractor_genfromtxt(fn)
        X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, TH, FLAG, FLUX = data_table
        data_table = QTable({
            'X': X, 'Y': Y, 'ERRX': ERRX, 'ERRY': ERRY,
            'A': A, 'B': B, 'XMIN': XMIN, 'YMIN': YMIN,
            'XMAX': XMAX, 'YMAX': YMAX, 'TH': TH,
            'FLAG': FLAG, 'FLUX': FLUX
        })
    except (FileNotFoundError, IOError) as e:
        print(f"Error when downloading data from a catalog file: {e}")
        logging.error(f"Error when downloading data from a catalog file: {e}")
        sys.exit(1) # Критическая ошибка, не можем продолжить без данных

    processed_table = utils.preprocess_data(data_table, x_min=100, x_max=3100, y_min=50, y_max=2105)
    # print(f"X: {X}, Y: {Y}, ERRX: {ERRX}, ERRY: {ERRY}, A: {A}, B: {B}, XMIN: {XMIN}, YMIN: {YMIN}, XMAX: {XMAX}, YMAX: {YMAX}, TH: {TH}, FLAG: {FLAG}, FLUX: {FLUX}")
    
    star_observation(X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, fits_filename, base_filename)

    
if __name__ == "__main__":
    main()