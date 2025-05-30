import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from astropy.coordinates import Angle
import astropy.units as u
import os
import glob
import getpass
from mpl_toolkits.mplot3d import Axes3D

def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sqrt(np.sum((points - median) ** 2, axis=-1))
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation

    return np.where(modified_z_score > thresh)[0]


def load_data(file_path):
    data =  np.genfromtxt(file_path, unpack=True)
    return data


def preprocess_data(X, Y, A, B, TH, FLAG, FLUX, x_min, x_max, y_min, y_max, percentile=98):
    # y_threshold = np.percentile(Y, percentile)
    # y_mask = Y < y_threshold
    
    y_mask = (Y >= y_min) & (Y <= y_max)

    x_mask = (X >= x_min) & (X <= x_max)
    
    mask = x_mask & y_mask
    
    X, Y, A, B, TH, FLAG, FLUX = X[mask], Y[mask], A[mask], B[mask], TH[mask], FLAG[mask], FLUX[mask]

    return X, Y, A, B, TH, FLAG, FLUX
    

def compute_elongation(A, B):
    return A / B

def scale_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def ab_ratio(ELONG, threshold):
    return ELONG < threshold


def cluster_data(features_scaled, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(features_scaled)


def choose_fits_file():
    log_file = 'TMP/processing_log.txt'
    if not os.path.exists(log_file):
        raise FileNotFoundError("Log file does not exist.")

    with open(log_file, 'r') as f:
        last_file = f.readline().strip()

    if not last_file:
        raise ValueError("No file name found in the log file.")

    last_file_base = os.path.basename(last_file)
    # Извлекаем суффикс (все символы до расширения)
    last_file_suffix = last_file_base.split('.')[0]
    print(f"Extracted suffix: {last_file_suffix}")

    fits_files = glob.glob('TMP/*.fits') + glob.glob('TMP/*.fit')
    
    # Печать всех найденных файлов
    print("Found FITS files:")
    for f in fits_files:
        print(f)

    matching_file = None
    for fits_file in fits_files:
        file_name = os.path.basename(fits_file)
        # Печать для отладки
        print(f"Checking file: {file_name}")
        # Проверка наличия суффикса в имени файла
        if last_file_suffix in file_name:
            matching_file = fits_file
            break

    if not matching_file:
        raise FileNotFoundError(f"No matching FITS file found for suffix {last_file_suffix}")

    print(f"Selected FITS file: {matching_file}")

    return matching_file, last_file_base


def convert(ra_deg, dec_deg):
    ra_angle = Angle(ra_deg, unit=u.deg)
    ra_hms = ra_angle.to_string(unit=u.hour, sep=':', precision=2)
    
    dec_angle = Angle(dec_deg, unit=u.deg)
    dec_dms = dec_angle.to_string(unit=u.deg, sep=':', precision=2)
    
    return ra_hms, dec_dms

# def save_results(coords, second_coord, base_filename):
def save_results(coords_first, coords_second, base_filename, fits_filename):
    output_dir = 'PROCESS_FILE'
    os.makedirs(output_dir, exist_ok=True)
    
    txt_filename = os.path.join(output_dir, f'{base_filename}.txt')
    with open(txt_filename, 'w') as f:
        f.write(f"#File: {base_filename}\n")
        
        with fits.open(fits_filename) as hdul:
            header = hdul[0].header
            date_obs = header.get('DATE-OBS', '00000')
            f.write(f"#{date_obs}\n")
        
        # Сохранение первой кластеризации
        for ra_hms, dec_dms in coords_first:
            f.write(f"{ra_hms} {dec_dms}\n")
            
        # Если есть вторая кластеризация, сохраняем её
        if coords_second:
            f.write(f"#Second cluster:\n")
            for ra_hms, dec_dms in coords_second:
                f.write(f"{ra_hms} {dec_dms}\n")


def star_track(X, Y, A, B, TH, FLAG, FLUX, fits_filename, base_filename):
    
    # try:
    #     fits_filename, base_filename = choose_fits_file()
    # except (FileNotFoundError, ValueError) as e:
    #     print(e)
    #     return

    # X, Y, A, B, TH, FLAG, FLUX = load_data(f'{DIR}{fn}')

    # X, Y, A, B, TH, FLAG, FLUX = preprocess_data(X, Y, A, B, TH, FLAG, FLUX, x_min=100, x_max=3100, y_min=50, y_max=2105)

    ELONG = compute_elongation(A, B)
    
    likely_satelite = ab_ratio(ELONG, threshold=1.45)

    hight_flux = FLUX >= 1000
    
    outlier_indices = is_outlier(ELONG)
    satellites = np.zeros(len(ELONG), dtype=bool)
    satellites[outlier_indices] = True

    features = np.column_stack((TH, likely_satelite.astype(int), hight_flux.astype(int)))
    features_scaled = scale_features(features)

    labels = cluster_data(features_scaled, eps=5, min_samples=5)

    satellite_mask = labels == -1
    satellite_x_coords = X[satellite_mask]
    satellite_y_coords = Y[satellite_mask]
    print(f'X: {satellite_x_coords} Y: {satellite_y_coords}')

    with fits.open(fits_filename) as hdul:
        wcs = WCS(hdul[0].header)
    sky_coords = wcs.pixel_to_world(satellite_x_coords, satellite_y_coords)

    print("RA and DEC for satellites: ")
    coords_first = []
    for coord in sky_coords:
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg
        
        ra_hms, dec_dms = convert(ra_deg, dec_deg)
        print(f"RA: {ra_hms}, DEC: {dec_dms}")
        coords_first.append((ra_hms, dec_dms))
    
    # Исключаем объекты, найденные в первой кластеризации
    non_satellite_mask = ~satellite_mask
    X_non_satellite = X[non_satellite_mask]
    Y_non_satellite = Y[non_satellite_mask]
    features_non_satellite = features_scaled[non_satellite_mask]

    # Вторая кластеризация с большей чувствительностью на оставшихся данных
    labels_sensitive = cluster_data(features_non_satellite, eps=1.5, min_samples=3)

    satellite_mask_sensitive = labels_sensitive == -1
    satellite_x_coords_sensitive = X_non_satellite[satellite_mask_sensitive]
    satellite_y_coords_sensitive = Y_non_satellite[satellite_mask_sensitive]
    print(f'X: {satellite_x_coords_sensitive} Y: {satellite_y_coords_sensitive}')    

    # Вывод координат для второй кластеризации
    sky_coords_sensitive = wcs.pixel_to_world(satellite_x_coords_sensitive, satellite_y_coords_sensitive)

    print("Second cluster: ")
    coords_second = []
    for coord in sky_coords_sensitive:
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg
        
        ra_hms, dec_dms = convert(ra_deg, dec_deg)
        print(f"RA: {ra_hms}, DEC: {dec_dms}")
        coords_second.append((ra_hms, dec_dms))
    
    # Сохраняем результаты обеих кластеризаций
    save_results(coords_first, coords_second, base_filename, fits_filename)

    non_anomalies_mask = ~satellites
    features_non_anomalies_scaled = features_scaled[non_anomalies_mask]
    labels_no_anomalies = cluster_data(features_non_anomalies_scaled, eps=0.3, min_samples=10)

    fig, ax = plt.subplots(figsize=(12, 9))

    if np.any(satellite_mask):
        ax.scatter(TH[satellite_mask], ELONG[satellite_mask], c='red', alpha=0.5, s=10, label='Satellites')
        ax.scatter(TH[satellite_mask], ELONG[satellite_mask], facecolor='none', edgecolors='black', s=200, label='Satellites')

    unique_labels_no_anomalies = np.unique(labels_no_anomalies)
    for cluster in unique_labels_no_anomalies:
        cluster_mask = labels_no_anomalies == cluster
        color = 'blue' if cluster != -1 else 'green'
        label = f'Stars {cluster}' if cluster != -1 else 'Cluster'
        ax.scatter(TH[non_anomalies_mask][cluster_mask], ELONG[non_anomalies_mask][cluster_mask],
                   c=color, alpha=0.5, label=label, s=10)

    ax.set_xlabel('Angle (TH)')
    ax.set_ylabel('A/B')
    ax.legend()
    # plt.show()

    return coords_first, coords_second

def star_observation(X, Y, A, B, TH, FLAG, FLUX, fits_filename, base_filename):
    
    # Этап 1: Вычисляем удлинение
    ELONG = compute_elongation(A, B)
    print(f"elong: {ELONG}")
    # Фильтрация данных только по ELONG > 3.5
    elong_mask = ELONG > 3.2
    print(f"elong: {elong_mask}")
    X_filtered = X[elong_mask]
    Y_filtered = Y[elong_mask]
    # TH_filtered = TH[elong_mask]
    
    
    print('After filtering')
    print(f'X: {X_filtered} - {X_filtered.shape}')
    print(f'Y: {Y_filtered} - {Y_filtered.shape}')

    # Преобразование координат отфильтрованных объектов в небесные (RA, Dec)
    with fits.open(fits_filename) as hdul:
        wcs = WCS(hdul[0].header)
    sky_coords = wcs.pixel_to_world(X_filtered, Y_filtered)

    # Вывод координат RA и Dec для отфильтрованных объектов
    coords_filtered = []
    for coord in sky_coords:
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg
        ra_hms, dec_dms = convert(ra_deg, dec_deg)
        print(f"RA: {ra_hms} DEC: {dec_dms}")
        coords_filtered.append((ra_hms, dec_dms))
    
    # Сохраняем отфильтрованные координаты в файл
    save_results(coords_filtered, [], base_filename, fits_filename)
    
    return coords_filtered



def main():
    DIR = 'TMP/'
    fn = 'k1-impTEST.fts.sx'

    try:
        fits_filename, base_filename = choose_fits_file()
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return
    
    X, Y, A, B, TH, FLAG, FLUX = load_data(f'{DIR}{fn}')
    # X, Y, A, B, TH, FLAG, FLUX = preprocess_data(X, Y, A, B, TH, FLAG, FLUX, x_min=100, x_max=3100, y_min=50, y_max=2105)
    
    
    while True:
        user_input = getpass.getpass("Select mode: 1 for Star Track, 2 for Star Observation: ")
        print(f"User input received: '{user_input}'")
        try:
            mode = int(user_input)
            if mode in [1, 2]:
                print(f'Mode selected: {mode}')
                break
            else:
                print("Invalid mode selected. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("Proceeding with selected mode...")
    
    if mode == 1:
        X, Y, A, B, TH, FLAG, FLUX = preprocess_data(X, Y, A, B, TH, FLAG, FLUX, x_min=100, x_max=3100, y_min=50, y_max=2105)
        coords_first = star_track(X, Y, A, B, TH, FLAG, FLUX, fits_filename, base_filename)
        # save_results(coords_first, coords_second, base_filename)
    elif mode == 2:
        X, Y, A, B, TH, FLAG, FLUX = preprocess_data(X, Y, A, B, TH, FLAG, FLUX, x_min=100, x_max=4500, y_min=50, y_max=3500)
        coords_second = star_observation(X, Y, A, B, TH, FLAG, FLUX, fits_filename, base_filename)
        # save_results(coords_first, coords_second, base_filename)

if __name__ == "__main__":
    main()