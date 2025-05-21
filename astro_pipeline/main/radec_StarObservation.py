# Star observation mode
"""
@author: saf

Detection object and convert pixcoordinates on RA and DEC

"""

import os
import glob
import numpy as np
import path
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy.time import Time
from datetime import timedelta
from mpl_toolkits.mplot3d import Axes3D
from src.utils import utils



def save_results(coords, fits_filename, base_filename, X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX):
    output_dir = 'PROCESS_FILE'
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
    try:
        fits_filename, base_filename = utils.choose_fits_file()
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, TH, FLAG, FLUX = utils.load_sextractor_genfromtxt(os.path.join(DIR, fn))
    # print(f"X: {X}, Y: {Y}, ERRX: {ERRX}, ERRY: {ERRY}, A: {A}, B: {B}, XMIN: {XMIN}, YMIN: {YMIN}, XMAX: {XMAX}, YMAX: {YMAX}, TH: {TH}, FLAG: {FLAG}, FLUX: {FLUX}")
    
    X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, TH, FLAG, FLUX = utils.preprocess_data(X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, TH, FLAG, FLUX, x_min=25, x_max=4785, y_min=15, y_max=3175)
    # print(f"X: {X}, Y: {Y}, ERRX: {ERRX}, ERRY: {ERRY}, A: {A}, B: {B}, XMIN: {XMIN}, YMIN: {YMIN}, XMAX: {XMAX}, YMAX: {YMAX}, TH: {TH}, FLAG: {FLAG}, FLUX: {FLUX}")
    
    star_observation(X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, fits_filename, base_filename)

    
if __name__ == "__main__":
    main()