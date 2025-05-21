import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from astropy.coordinates import Angle
import astropy.units as u
import os

def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sqrt(np.sum((points - median) ** 2, axis=-1))
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation

    return np.where(modified_z_score > thresh)[0]

def load_data(file_path):
    return np.genfromtxt(file_path, unpack=True)

def preprocess_data(X, Y, A, B, TH, FLAG, FLUX, percentile=99.5):
    y_threshold = np.percentile(Y, percentile)
    mask = Y < y_threshold
    return X[mask], Y[mask], A[mask], B[mask], TH[mask], FLAG[mask], FLUX[mask]

def compute_elongation(A, B):
    return A / B

def scale_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

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
    last_file_suffix = last_file_base[-16:-4]  # Предполагается, что имя файла имеет длину 16 символов перед расширением

    fits_files = [f for f in os.listdir('TMP') if f.endswith('.fit')]
    matching_file = None
    for fits_file in fits_files:
        if fits_file.endswith(last_file_suffix + '.fit'):
            matching_file = fits_file
            break

    if not matching_file:
        raise FileNotFoundError(f"No matching FITS file found for suffix {last_file_suffix}")

    return os.path.join('TMP', matching_file), last_file_base

def convert(ra_deg, dec_deg):
    ra_angle = Angle(ra_deg, unit=u.deg)
    ra_hms = ra_angle.to_string(unit=u.hour, sep=':', precision=2)
    
    dec_angle = Angle(dec_deg, unit=u.deg)
    dec_dms = dec_angle.to_string(unit=u.deg, sep=':', precision=2)
    
    return ra_hms, dec_dms

def save_results(coords, base_filename):
    output_dir = 'PROCESS_FILE'
    os.makedirs(output_dir, exist_ok=True)
    
    txt_filename = os.path.join(output_dir, f'{base_filename}.txt')
    with open(txt_filename, 'w') as f:
        f.write(f"#File: {base_filename}\n")
        for ra_hms, dec_dms in coords:
            f.write(f"{ra_hms} {dec_dms}\n")

def main():
    DIR = 'TMP/'
    fn = 'k1-imp.fts.sx'

    fits_filename, base_filename = choose_fits_file()
    if not fits_filename:
        print("No FITS file found")
        return

    X, Y, A, B, TH, FLAG, FLUX = load_data(f'{DIR}{fn}')

    X, Y, A, B, TH, FLAG, FLUX = preprocess_data(X, Y, A, B, TH, FLAG, FLUX)

    ELONG = compute_elongation(A, B)

    outlier_indices = is_outlier(ELONG)
    satelites = np.zeros(len(ELONG), dtype=bool)
    satelites[outlier_indices] = True

    features = np.column_stack((TH, ELONG))
    features_scaled = scale_features(features)

    labels = cluster_data(features_scaled, eps=5, min_samples=5)

    satelite_mask = labels == -1
    satelite_x_coods = X[satelite_mask]
    satelite_y_coods = Y[satelite_mask]
    print(f'X: {satelite_x_coods} Y: {satelite_y_coods}')

    with fits.open(fits_filename) as hdul:
        wcs = WCS(hdul[0].header)
    sky_coords = wcs.pixel_to_world(satelite_x_coods, satelite_y_coods)

    print("RA and DEC for satelites: ")
    coords = []
    for coord in sky_coords:
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg
        
        ra_hms, dec_dms = convert(ra_deg, dec_deg)
        print(f"RA: {ra_hms}, DEC: {dec_dms}")
        coords.append((ra_hms, dec_dms))
    
    save_results(coords, base_filename)

    non_anomalies_mask = ~satelites
    features_non_anomalies_scaled = features_scaled[non_anomalies_mask]
    labels_no_anomalies = cluster_data(features_non_anomalies_scaled, eps=0.3, min_samples=10)

    fig, ax = plt.subplots(figsize=(12, 9))

    if np.any(satelite_mask):
        ax.scatter(TH[satelite_mask], ELONG[satelite_mask], c='red', alpha=0.5, s=10, label='Satelites')
        ax.scatter(TH[satelite_mask], ELONG[satelite_mask], facecolor='none', edgecolors='black', s=200, label='Satelites')

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

if __name__ == "__main__":
    main()