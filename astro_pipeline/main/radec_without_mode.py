#
# Object target mode

"""
@author: saf

Detection object and convert pixcoordinates on RA and DEC

"""

import os
import glob
import numpy as np
import sys
import logging
from src.utils import path
import astropy.units as u
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from astropy.coordinates import Angle
from astropy.time import Time
from datetime import timedelta
from astropy.table import QTable
from main.src.utils import ut



def scale_features(features):
    """Scales signs."""
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def ab_ratio(ELONG, threshold):
    """Defines objects with a low ratio A/B."""
    return ELONG < threshold


def cluster_data(features_scaled, eps, min_samples):
    """Classification performs DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(features_scaled)


def save_results(coords_first, coords_second, base_filename, fits_filename, x_y_a_b_values, errors):
    """Saves the results of clusterings."""
    output_dir = path.PROCESSED_RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    txt_filename = os.path.join(output_dir, f'{base_filename}.txt')
    with open(txt_filename, 'w') as f:
        f.write(f"File: {base_filename}\n")

        with fits.open(fits_filename) as hdul:
            header = hdul[0].header
            date_obs = header.get('DATE-OBS', '00000')
            exptime = header.get('EXPTIME', 0)
            if date_obs != '00000' and exptime > 0:
                date_str, time_str = date_obs.split('T')
                time_obs = Time(f'{date_str} {time_str}', format='iso')
                avg_exposure_time = time_obs + timedelta(seconds=(exptime / 2.0))
                new_time_str = avg_exposure_time.iso.replace(" ", "T")
                f.write(f"{new_time_str}\n")
            else:
                f.write(f"{date_obs}\n")

        for (ra_hms, dec_dms), x, y, a, b, xmin, ymin, xmax, ymax, erroreX, erroreY in zip(
            coords_first,
            x_y_a_b_values['X_first'],
            x_y_a_b_values['Y_first'],
            x_y_a_b_values['A_first'],
            x_y_a_b_values['B_first'],
            x_y_a_b_values['XMIN_first'],
            x_y_a_b_values['YMIN_first'],
            x_y_a_b_values['XMAX_first'],
            x_y_a_b_values['YMAX_first'],
            errors["err_x_first"],
            errors["err_y_first"]
        ):
            f.write(f"{ra_hms} {dec_dms} {x} {y} {erroreX} {erroreY} {a} {b} {xmin} {ymin} {xmax} {ymax}\n")

        if coords_second:
            f.write(f"#Second cluster:\n")
            for (ra_hms, dec_dms), x, y, a, b, xmin, ymin, xmax, ymax, erroreX, erroreY in zip(
                coords_second,
                x_y_a_b_values['X_second'],
                x_y_a_b_values['Y_second'],
                x_y_a_b_values['A_second'],
                x_y_a_b_values['B_second'],
                x_y_a_b_values['XMIN_second'],
                x_y_a_b_values['YMIN_second'],
                x_y_a_b_values['XMAX_second'],
                x_y_a_b_values['YMAX_second'],
                errors["err_x_second"],
                errors["err_y_second"]
            ):
                f.write(f"{ra_hms} {dec_dms} {x} {y} {erroreX} {erroreY} {a} {b} {xmin} {ymin} {xmax} {ymax}\n")


def graf(TH, ELONG, satellite_mask, anomaly_mask, TH_non_satellite, ELONG_non_satellite):

    fig1, ax1 = plt.subplots(figsize=(12, 9))
    ax1.scatter(TH, ELONG, c='blue', s=10, label='All data')
    if np.any(satellite_mask):
        ax1.scatter(TH[satellite_mask], ELONG[satellite_mask],
                    facecolors='none', edgecolors='red', s=200, label='First cluster')
    if np.any(anomaly_mask):
        ax1.scatter(TH_non_satellite[anomaly_mask], ELONG_non_satellite[anomaly_mask],
                    facecolors='none', edgecolors='green', s=200,
                    label='Second cluster')

    ax1.set_xlabel('Angle (TH)')
    ax1.set_ylabel('A/B')
    ax1.set_title('DBSCAN in (TH, A/B) feature space')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    # plt.show()


def main():
    DIR = path.TMP_DIR
    fn = path.SEX_CATALOG_FILE

    # try:
    #     fits_filename, base_filename = utils.choose_fits_file()
    # except (FileNotFoundError, ValueError) as e:
    #     print(e)
    #     return

    #     # Выбираем FITS-файл на основе лог-файла
    try:
        # Используем choose_fits_file из utils, передавая путь к логу и директории поиска FITS
        # Ищем FITS файл в TMP_DIR и PROCESSED_FITS_DIR (на всякий случай)
        fits_filename, base_filename = ut.choose_fits_file(str(path.PROCESSING_LOG_FILE), [str(path.TMP_DIR), str(path.PROCESSED_FITS_DIR)])
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
        data_table = ut.load_sextractor_genfromtxt(fn)
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

    processed_table = ut.preprocess_data(data_table, x_min=100, x_max=3100, y_min=50, y_max=2105)

    X = processed_table['X']
    Y = processed_table['Y']
    ERRX = processed_table['ERRX']
    ERRY = processed_table['ERRY']
    A = processed_table['A']
    B = processed_table['B']
    XMIN = processed_table['XMIN']
    YMIN = processed_table['YMIN']
    XMAX = processed_table['XMAX']
    YMAX = processed_table['YMAX']
    TH = processed_table['TH']
    FLAG = processed_table['FLAG']
    FLUX = processed_table['FLUX']

    ELONG = ut.compute_elongation(A, B)

    likely_satelite = ab_ratio(ELONG, threshold=5)

    erroreX, erroreY = ut.compute_errores(ERRX, ERRY)

    hight_flux = FLUX >= 1000

    outlier_indices = ut.is_outlier(ELONG)
    satellites = np.zeros(len(ELONG), dtype=bool)
    satellites[outlier_indices] = True

    features = np.column_stack((TH, likely_satelite, hight_flux.astype(int)))
    features_scaled = scale_features(features)

    # Первая кластеризация для обнаружения потенциального спутника
    labels_first_cluster = cluster_data(features_scaled, eps=5, min_samples=5)
    satellite_mask = labels_first_cluster == -1
    satellite_x_coords = X[satellite_mask]
    satellite_y_coords = Y[satellite_mask]
    # print(f'First cluster:\n{satellite_x_coords}{satellite_y_coords}\n')
    print(f"First cluster:")
    for x, y in zip(satellite_x_coords, satellite_y_coords):
        print(f" X: {x}, Y: {y}")

    with fits.open(fits_filename) as hdul:
        wcs = WCS(hdul[0].header)
    sky_coords_first = wcs.pixel_to_world(satellite_x_coords, satellite_y_coords)

    print("RA and DEC: ")
    coords_first = []
    for coord in sky_coords_first:
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg

        ra_hms, dec_dms = ut.convert_deg_to_hmsdms(ra_deg, dec_deg)
        print(f"RA: {ra_hms}, DEC: {dec_dms}")
        coords_first.append((ra_hms, dec_dms))

    # Вторая кластеризация для обнаружения аномалий (треков)
    # Исключаем объекты, найденные в первой кластеризации
    non_satellite_mask = ~satellite_mask
    X_non_satellite = X[non_satellite_mask]
    Y_non_satellite = Y[non_satellite_mask]
    TH_non_satellite = TH[non_satellite_mask]
    ELONG_non_satellite = ELONG[non_satellite_mask]

    # Используем ELONG и TH в качестве признаков для обнаружения аномалий
    features_anomalies = np.column_stack((ELONG_non_satellite, TH_non_satellite))
    features_anomalies_scaled = scale_features(features_anomalies)

    labels_anomalies = cluster_data(features_anomalies_scaled, eps=0.5, min_samples=3)
    anomaly_mask = labels_anomalies == -1
    anomaly_x_coords = X_non_satellite[anomaly_mask]
    anomaly_y_coords = Y_non_satellite[anomaly_mask]
    # print(f'Second cluster:\n{anomaly_x_coords} {anomaly_y_coords}\n')
    print(f"Second cluster:")
    for x, y in zip(anomaly_x_coords, anomaly_y_coords):
        print(f" X: {x}, Y: {y}")

    # Вывод координат для второй кластеризации
    sky_coords_second = wcs.pixel_to_world(anomaly_x_coords, anomaly_y_coords)

    print("RA and DEC: ")
    coords_second = []
    for coord in sky_coords_second:
        ra_deg = coord.ra.deg
        dec_deg = coord.dec.deg

        ra_hms, dec_dms = ut.convert_deg_to_hmsdms(ra_deg, dec_deg)
        print(f"RA: {ra_hms}, DEC: {dec_dms}")
        coords_second.append((ra_hms, dec_dms))

    x_y_a_b_values = {
        'X_first': satellite_x_coords,
        'Y_first': satellite_y_coords,
        'A_first': A[satellite_mask],
        'B_first': B[satellite_mask],
        'XMIN_first': XMIN[satellite_mask],
        'YMIN_first': YMIN[satellite_mask],
        'XMAX_first': XMAX[satellite_mask],
        'YMAX_first': YMAX[satellite_mask],

        'X_second': anomaly_x_coords,
        'Y_second': anomaly_y_coords,
        'A_second': A[non_satellite_mask][anomaly_mask],
        'B_second': B[non_satellite_mask][anomaly_mask],
        'XMIN_second': XMIN[non_satellite_mask][anomaly_mask],
        'YMIN_second': YMIN[non_satellite_mask][anomaly_mask],
        'XMAX_second': XMAX[non_satellite_mask][anomaly_mask],
        'YMAX_second': YMAX[non_satellite_mask][anomaly_mask],
    }

    errors = {
        "err_x_first": erroreX[satellite_mask],
        "err_y_first": erroreY[satellite_mask],
        "err_x_second": erroreX[non_satellite_mask][anomaly_mask],
        "err_y_second": erroreY[non_satellite_mask][anomaly_mask]
    }

    save_results(coords_first, coords_second, base_filename, fits_filename, x_y_a_b_values, errors)
    # Визуализация 1: TH vs ELONG
    graf(TH, ELONG, satellite_mask, anomaly_mask, TH_non_satellite, ELONG_non_satellite)


if __name__ == "__main__":
    main()



# radec_without_mode.py

# import numpy as np
# from astropy.io import fits
# from astropy.wcs import WCS
# from matplotlib import pyplot as plt
# from sklearn.cluster import DBSCAN # Используется только здесь
# from sklearn.preprocessing import StandardScaler # Используется только здесь
# from astropy.coordinates import Angle # Используется в utils, но может понадобиться здесь для явности
# from astropy.time import Time # Используется в save_results
# from datetime import timedelta # Используется в save_results
# import astropy.units as u # Используется в utils, но может понадобиться здесь для явности
# import os # Используется, но лучше перейти на pathlib везде
# import glob # Используется в choose_fits_file (теперь в utils)
# from astropy.table import QTable # Используется здесь для загрузки данных

# # Импортируем наши модули
# from pathlib import Path # Для работы с путями как объектами
# import path # Модуль с централизованными путями
# from src.utils import utils # Модуль с общими утилитами
# import logging # Добавим логирование
# import sys

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# # --- Утилитарные функции (НЕПЕРЕМЕЩЕННЫЕ В utils, т.к. специфичны или используют специфические структуры) ---

# # Функция для загрузки данных из файла, специфичная для этого скрипта (использует QTable)
# # Оригинальная функция load_data из radec_without_mode.py
# def load_data(file_path: Path):
#     """
#     Загружает данные из файла SExtractor в QTable.

#     Args:
#         file_path (Path): Путь к файлу каталога (объект Path).

#     Returns:
#         QTable: Таблица с данными из файла.
#     Raises:
#         FileNotFoundError: Если файл не найден.
#         Exception: При ошибке чтения файла.
#     """
#     if not file_path.exists():
#          logging.error(f"Файл данных не найден: {file_path}")
#          raise FileNotFoundError(f"Файл данных не найден: {file_path}")

#     logging.info(f"Загрузка данных из {file_path} в QTable...")
#     try:
#         # Попробуем прочитать как ASCII-таблицу с разделителями
#         # Используем str(file_path) для совместимости с QTable.read
#         data = QTable.read(str(file_path), format='ascii.fast_no_header', delimiter=' ', names=('X', 'Y', 'ERRX', 'ERRY', 'A', 'B', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'TH', 'FLAG', 'FLUX'))
#         logging.info(f"Загружено {len(data)} строк.")
#         return data
#     except Exception as e:
#         logging.error(f"Ошибка при чтении файла {file_path} как ASCII в QTable: {e}")
#         raise # Перевыбрасываем исключение


# # Функция предварительной обработки данных (фильтрация по координатам), специфичная для этого скрипта (использует QTable)
# # Оригинальная функция preprocess_data из radec_without_mode.py
# def preprocess_data(table: QTable, x_min: float, x_max: float, y_min: float, y_max: float):
#     """
#     Предварительная обработка данных (фильтрация по диапазону координат X и Y) с использованием QTable.

#     Args:
#         table (QTable): Входная таблица с данными.
#         x_min (float): Минимальное значение X.
#         x_max (float): Максимальное значение X.
#         y_min (float): Минимальное значение Y.
#         y_max (float): Максимальное значение Y.

#     Returns:
#         QTable: Отфильтрованная таблица.
#     """
#     logging.info(f"Фильтрация данных по диапазону X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
#     initial_count = len(table)
#     y_mask = (table['Y'] >= y_min) & (table['Y'] <= y_max)
#     x_mask = (table['X'] >= x_min) & (table['X'] <= x_max)
#     mask = x_mask & y_mask
#     filtered_table = table[mask]
#     logging.info(f"Отфильтровано {initial_count - len(filtered_table)} объектов. Осталось {len(filtered_table)}.")
#     return filtered_table

# # Функция для масштабирования признаков (используется только здесь для DBSCAN)
# # Оригинальная функция scale_features из radec_without_mode.py
# def scale_features(features: np.ndarray):
#     """
#     Масштабирует признаки с использованием StandardScaler.

#     Args:
#         features (np.ndarray): Массив признаков.

#     Returns:
#         np.ndarray: Масштабированный массив признаков.
#     """
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)
#     # logging.debug("Features scaled.")
#     return scaled_features

# # Функция для кластеризации данных с помощью DBSCAN (используется только здесь)
# # Оригинальная функция cluster_data из radec_without_mode.py
# def cluster_data(features_scaled: np.ndarray, eps: float, min_samples: int):
#     """
#     Выполняет кластеризацию DBSCAN на масштабированных признаках.

#     Args:
#         features_scaled (np.ndarray): Масштабированный массив признаков.
#         eps (float): Максимальное расстояние между двумя образцами для одного
#                      соседствовать с другим.
#         min_samples (int): Количество образцов в окрестности для точки, которая будет считаться
#                          основной точкой.

#     Returns:
#         np.ndarray: Массив меток кластеров для каждого образца (-1 для выбросов).
#     """
#     logging.info(f"Запуск DBSCAN с eps={eps}, min_samples={min_samples}")
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(features_scaled)
#     # logging.debug(f"DBSCAN finished. Found {len(np.unique(labels)) - (1 if -1 in labels else 0)} clusters.")
#     return labels


# # Функция сохранения результатов (специфична для двух кластеров в этом скрипте)
# # Оригинальная функция save_results из radec_without_mode.py
# def save_results(
#     coords_first: list,
#     coords_second: list,
#     base_filename: str,
#     fits_file_path: Path, # Принимаем объект Path
#     x_y_a_b_values: dict,
#     errors: dict
# ):
#     """
#     Сохраняет результаты двух кластеризаций в текстовый файл.

#     Args:
#         coords_first (list): Список кортежей (ra_hms, dec_dms) для первого кластера.
#         coords_second (list): Список кортежей (ra_hms, dec_dms) для второго кластера.
#         base_filename (str): Базовое имя исходного FITS файла.
#         fits_file_path (Path): Путь к FITS файлу для извлечения времени (объект Path).
#         x_y_a_b_values (dict): Словарь с массивами X, Y, A, B, XMIN, YMIN, XMAX, YMAX для обоих кластеров.
#         errors (dict): Словарь с массивами ошибок erroreX, erroreY для обоих кластеров.
#     """
#     output_dir = path.PROCESSED_RESULTS_DIR # Берем директорию из path.py
#     logging.info(f"Сохранение результатов в директорию {output_dir}")

#     # Убедимся, что выходная директория существует
#     try:
#         output_dir.mkdir(parents=True, exist_ok=True)
#         # logging.debug(f"Создана/существует выходная директория: {output_dir}")
#     except OSError as e:
#         logging.error(f"Не удалось создать выходную директорию {output_dir}: {e}")
#         print(f"Error creating output directory {output_dir}: {e}")
#         return # Не можем сохранить, выходим из функции

#     txt_filename = output_dir / f'{base_filename}.txt' # Используем Path для объединения путей

#     try:
#         # Открываем файл для записи, используя объект Path
#         with txt_filename.open('w') as f:
#             f.write(f"File: {base_filename}\n")

#             # Извлекаем время из заголовка FITS
#             try:
#                 with fits_file_path.open('rb') as f_fits: # Открываем в бинарном режиме для fits.open
#                      with fits.open(f_fits) as hdul:
#                         header = hdul[0].header
#                         date_obs = header.get('DATE-OBS', '00000')
#                         exptime = header.get('EXPTIME', 0)

#                         if date_obs != '00000' and exptime > 0:
#                             try:
#                                 date_str, time_str = date_obs.split('T')
#                                 time_obs = Time(f'{date_str} {time_str}', format='iso')
#                                 avg_exposure_time = time_obs + timedelta(seconds=(exptime / 2.0))
#                                 new_time_str = avg_exposure_time.iso.replace(" ", "T")
#                                 f.write(f"{new_time_str}\n")
#                             except Exception as time_e:
#                                 logging.warning(f"Ошибка при расчете среднего времени экспозиции: {time_e}. Записываем исходный DATE-OBS.")
#                                 f.write(f"{date_obs}\n")
#                         else:
#                             f.write(f"{date_obs}\n")
#             except FileNotFoundError:
#                  logging.warning(f"FITS файл {fits_file_path} не найден при попытке извлечь время для сохранения результатов.")
#                  f.write("DATE-OBS: N/A\n")
#             except Exception as fits_header_e:
#                  logging.warning(f"Ошибка при чтении заголовка FITS файла {fits_file_path} для времени: {fits_header_e}")
#                  f.write("DATE-OBS: Error reading header\n")


#             # Сохранение первой кластеризации
#             if coords_first: # Проверяем, есть ли объекты в первом кластере
#                 logging.info(f"Сохранение данных для первого кластера ({len(coords_first)} объектов)...")
#                 for i, (ra_hms, dec_dms) in enumerate(coords_first):
#                     f.write(
#                         f"{ra_hms} {dec_dms} "
#                         f"{x_y_a_b_values['X_first'][i]:.2f} {x_y_a_b_values['Y_first'][i]:.2f} "
#                         f"{errors['err_x_first'][i]:.4f} {errors['err_y_first'][i]:.4f} "
#                         f"{x_y_a_b_values['A_first'][i]:.4f} {x_y_a_b_values['B_first'][i]:.4f} "
#                         f"{x_y_a_b_values['XMIN_first'][i]:.1f} {x_y_a_b_values['YMIN_first'][i]:.1f} "
#                         f"{x_y_a_b_values['XMAX_first'][i]:.1f} {x_y_a_b_values['YMAX_first'][i]:.1f}\n"
#                     )

#             # Если есть вторая кластеризация, сохраняем её
#             if coords_second: # Проверяем, есть ли объекты во втором кластере
#                 logging.info(f"Сохранение данных для второго кластера ({len(coords_second)} объектов)...")
#                 f.write(f"#Second cluster:\n")
#                 for i, (ra_hms, dec_dms) in enumerate(coords_second):
#                      f.write(
#                         f"{ra_hms} {dec_dms} "
#                         f"{x_y_a_b_values['X_second'][i]:.2f} {x_y_a_b_values['Y_second'][i]:.2f} "
#                         f"{errors['err_x_second'][i]:.4f} {errors['err_y_second'][i]:.4f} "
#                         f"{x_y_a_b_values['A_second'][i]:.4f} {x_y_a_b_values['B_second'][i]:.4f} "
#                         f"{x_y_a_b_values['XMIN_second'][i]:.1f} {x_y_a_b_values['YMIN_second'][i]:.1f} "
#                         f"{x_y_a_b_values['XMAX_second'][i]:.1f} {x_y_a_b_values['YMAX_second'][i]:.1f}\n"
#                     )
#         logging.info(f"Результаты сохранены в {txt_filename}")

#     except Exception as e:
#         logging.error(f"Ошибка при сохранении результатов в файл {txt_filename}: {e}")
#         print(f"Error saving results to {txt_filename}: {e}")


# # --- Основная функция ---
# def main():

#     # Пути берем из нашего модуля path
#     sextractor_catalog_file = path.SEX_CATALOG_FILE # Путь к файлу каталога SExtractor

#     # Выбираем FITS-файл на основе лог-файла
#     try:
#         # Используем choose_fits_file из utils, передавая путь к логу и директории поиска FITS
#         # Ищем FITS файл в TMP_DIR и PROCESSED_FITS_DIR (на всякий случай)
#         fits_filename, base_filename = utils.choose_fits_file(str(path.PROCESSING_LOG_FILE), [str(path.TMP_DIR), str(path.PROCESSED_FITS_DIR)])
#         fits_file_path = Path(fits_filename) # Преобразуем результат в Path объект
#     except (FileNotFoundError, ValueError, IOError) as e:
#         print(f"Ошибка при выборе FITS файла: {e}")
#         logging.error(f"Ошибка при выборе FITS файла: {e}")
#         sys.exit(1) # Критическая ошибка, не можем продолжить без FITS файла

#     # Загружаем данные из файла каталога SExtractor
#     try:
#         # Используем локальную load_data, которая читает в QTable
#         data_table = load_data(sextractor_catalog_file)
#     except (FileNotFoundError, IOError) as e:
#         print(f"Ошибка при загрузке данных из файла каталога: {e}")
#         logging.error(f"Ошибка при загрузке данных из файла каталога: {e}")
#         sys.exit(1) # Критическая ошибка, не можем продолжить без данных

#     # Предварительная обработка данных (фильтрация по координатам)
#     # Используем локальную preprocess_data, которая работает с QTable
#     # Указываем диапазоны координат для фильтрации (взяты из оригинального скрипта)
#     processed_table = preprocess_data(data_table, x_min=100, x_max=3100, y_min=50, y_max=2105)

#     # Извлекаем данные из отфильтрованной таблицы в numpy массивы
#     X = processed_table['X'].data # Используем .data для получения numpy массива из Column
#     Y = processed_table['Y'].data
#     ERRX = processed_table['ERRX'].data
#     ERRY = processed_table['ERRY'].data
#     A = processed_table['A'].data
#     B = processed_table['B'].data
#     XMIN = processed_table['XMIN'].data
#     YMIN = processed_table['YMIN'].data
#     XMAX = processed_table['XMAX'].data
#     YMAX = processed_table['YMAX'].data
#     TH = processed_table['TH'].data
#     FLAG = processed_table['FLAG'].data
#     FLUX = processed_table['FLUX'].data

#     # Вычисляем элонгацию (A/B)
#     # Используем функцию из utils
#     ELONG = utils.compute_elongation(A, B)

#     # Определяем объекты с низкой элонгацией (потенциально "точки")
#     # likely_satelite = ab_ratio(ELONG, threshold=5) # ab_ratio не перенесена в utils

#     # Вычисляем ошибки на основе ERRX и ERRY
#     # Используем функцию из utils
#     erroreX, erroreY = utils.compute_errores(ERRX, ERRY)

#     # Определяем объекты с высоким потоком (hight_flux)
#     hight_flux = FLUX >= 1000 # Этот порог взят из оригинального скрипта

#     # Определяем выбросы по элонгации с использованием модифицированного Z-счета
#     # Используем функцию is_outlier из utils
#     outlier_indices = utils.is_outlier(ELONG)

#     # Создаем маску для объектов, которые считаем "спутниками" (выбросы по элонгации)
#     # В оригинальном скрипте это называлось satellites, но маска строилась только по outlier_indices
#     satellite_mask_by_elong = np.zeros(len(ELONG), dtype=bool)
#     satellite_mask_by_elong[outlier_indices] = True

#     # Формируем признаки для первой кластеризации (потенциальные "спутники")
#     # Оригинальный скрипт использовал (TH, likely_satelite, hight_flux.astype(int))
#     # likely_satelite основывался на пороге ELONG > 5. Давайте используем саму ELONG
#     # Признаки: Угол TH, Элонгация ELONG, Высокий поток FLUX
#     features_first_cluster = np.column_stack((TH, ELONG, hight_flux.astype(int)))

#     # Масштабируем признаки для DBSCAN
#     # Используем локальную функцию scale_features
#     features_first_cluster_scaled = scale_features(features_first_cluster)

#     # Первая кластеризация DBSCAN для обнаружения потенциального спутника (выбросы DBSCAN)
#     # Используем локальную функцию cluster_data
#     # Параметры eps и min_samples взяты из оригинального скрипта
#     labels_first_cluster = cluster_data(features_first_cluster_scaled, eps=5, min_samples=5)
#     # DBSCAN помечает выбросы как -1
#     first_cluster_mask = labels_first_cluster == -1

#     satellite_x_coords = X[first_cluster_mask]
#     satellite_y_coords = Y[first_cluster_mask]

#     print(f"First cluster (potential satellites):")
#     if len(satellite_x_coords) > 0:
#         for x, y in zip(satellite_x_coords, satellite_y_coords):
#             print(f" X: {x:.2f}, Y: {y:.2f}")
#     else:
#         print(" Нет объектов в первом кластере.")


#     # Преобразование пиксельных координат первого кластера в небесные (RA/DEC)
#     logging.info("Преобразование координат первого кластера в RA/DEC...")
#     coords_first = []
#     if len(satellite_x_coords) > 0:
#         try:
#             # Открываем FITS файл для получения WCS
#             with fits_file_path.open('rb') as f_fits: # Открываем в бинарном режиме для fits.open
#                 with fits.open(f_fits) as hdul:
#                     # Создаем объект WCS из заголовка
#                     wcs = WCS(hdul[0].header)

#             # Преобразуем пиксельные координаты в небесные
#             # Убедимся, что WCS корректно инициализирован
#             if wcs.celestial is None:
#                  raise ValueError("WCS в заголовке FITS файла не содержит небесной привязки.")

#             sky_coords_first = wcs.pixel_to_world(satellite_x_coords, satellite_y_coords)

#             logging.info("Координаты RA и DEC (первый кластер):")
#             for coord in sky_coords_first:
#                 ra_deg = coord.ra.deg
#                 dec_deg = coord.dec.deg
#                 # Используем функцию convert_deg_to_hmsdms из utils
#                 ra_hms, dec_dms = utils.convert_deg_to_hmsdms(ra_deg, dec_deg)
#                 print(f"RA: {ra_hms}, DEC: {dec_dms}")
#                 coords_first.append((ra_hms, dec_dms))

#         except FileNotFoundError:
#              logging.error(f"FITS файл {fits_file_path} не найден для преобразования WCS.")
#         except ValueError as wcs_err:
#              logging.error(f"Ошибка при инициализации WCS или преобразовании координат: {wcs_err}")
#              print(f"Error converting coordinates (first cluster): {wcs_err}")
#         except Exception as e:
#             logging.exception("Неожиданная ошибка при преобразовании координат первого кластера:")
#             print(f"An unforeseen error occurred during coordinate conversion (first cluster): {e}")


#     # Вторая кластеризация DBSCAN для обнаружения аномалий (треков)
#     # Исключаем объекты, найденные в первом кластере (выбросы DBSCAN первой кластеризации)
#     non_satellite_mask = ~first_cluster_mask # Инвертируем маску первого кластера

#     # Фильтруем исходные данные по этой маске
#     X_non_satellite = X[non_satellite_mask]
#     Y_non_satellite = Y[non_satellite_mask]
#     TH_non_satellite = TH[non_satellite_mask]
#     ELONG_non_satellite = ELONG[non_satellite_mask] # Используем рассчитанную ELONG

#     # Используем ELONG и TH в качестве признаков для обнаружения аномалий (треков)
#     features_anomalies = np.column_stack((ELONG_non_satellite, TH_non_satellite))

#     # Масштабируем признаки для DBSCAN
#     # Используем локальную функцию scale_features
#     features_anomalies_scaled = scale_features(features_anomalies)

#     # Вторая кластеризация DBSCAN для обнаружения аномалий (выбросы DBSCAN второй кластеризации)
#     # Параметры eps и min_samples взяты из оригинального скрипта
#     labels_anomalies = cluster_data(features_anomalies_scaled, eps=0.5, min_samples=3)
#     # DBSCAN помечает выбросы как -1
#     anomaly_mask = labels_anomalies == -1

#     anomaly_x_coords = X_non_satellite[anomaly_mask]
#     anomaly_y_coords = Y_non_satellite[anomaly_mask]

#     print(f"\nSecond cluster (anomalies/tracks):")
#     if len(anomaly_x_coords) > 0:
#         for x, y in zip(anomaly_x_coords, anomaly_y_coords):
#             print(f" X: {x:.2f}, Y: {y:.2f}")
#     else:
#         print(" Нет объектов во втором кластере.")


#     # Преобразование пиксельных координат второго кластера в небесные (RA/DEC)
#     logging.info("Преобразование координат второго кластера в RA/DEC...")
#     coords_second = []
#     if len(anomaly_x_coords) > 0:
#         try:
#              # Открываем FITS файл для получения WCS (делаем это еще раз, если функция выше не смогла)
#              with fits_file_path.open('rb') as f_fits:
#                 with fits.open(f_fits) as hdul:
#                     wcs = WCS(hdul[0].header)

#              if wcs.celestial is None:
#                  raise ValueError("WCS в заголовке FITS файла не содержит небесной привязки.")

#              sky_coords_second = wcs.pixel_to_world(anomaly_x_coords, anomaly_y_coords)

#              logging.info("Координаты RA и DEC (второй кластер):")
#              for coord in sky_coords_second:
#                  ra_deg = coord.ra.deg
#                  dec_deg = coord.dec.deg
#                  # Используем функцию convert_deg_to_hmsdms из utils
#                  ra_hms, dec_dms = utils.convert_deg_to_hmsdms(ra_deg, dec_deg)
#                  print(f"RA: {ra_hms}, DEC: {dec_dms}")
#                  coords_second.append((ra_hms, dec_dms))

#         except FileNotFoundError:
#              logging.error(f"FITS файл {fits_file_path} не найден для преобразования WCS.")
#         except ValueError as wcs_err:
#              logging.error(f"Ошибка при инициализации WCS или преобразовании координат: {wcs_err}")
#              print(f"Error converting coordinates (second cluster): {wcs_err}")
#         except Exception as e:
#             logging.exception("Неожиданная ошибка при преобразовании координат второго кластера:")
#             print(f"An unforeseen error occurred during coordinate conversion (second cluster): {e}")


#     # Собираем значения X, Y, A, B, XMIN, YMIN, XMAX, YMAX для сохранения
#     # для первого кластера (спутники) берем из исходных данных по маске first_cluster_mask
#     # для второго кластера (аномалии) берем из отфильтрованных ~first_cluster_mask по маске anomaly_mask
#     x_y_a_b_values = {
#         'X_first': X[first_cluster_mask],
#         'Y_first': Y[first_cluster_mask],
#         'A_first': A[first_cluster_mask],
#         'B_first': B[first_cluster_mask],
#         'XMIN_first': XMIN[first_cluster_mask],
#         'YMIN_first': YMIN[first_cluster_mask],
#         'XMAX_first': XMAX[first_cluster_mask],
#         'YMAX_first': YMAX[first_cluster_mask],

#         'X_second': X_non_satellite[anomaly_mask],
#         'Y_second': Y_non_satellite[anomaly_mask],
#         'A_second': A[non_satellite_mask][anomaly_mask], # Берем из исходных данных по обеим маскам
#         'B_second': B[non_satellite_mask][anomaly_mask],
#         'XMIN_second': XMIN[non_satellite_mask][anomaly_mask],
#         'YMIN_second': YMIN[non_satellite_mask][anomaly_mask],
#         'XMAX_second': XMAX[non_satellite_mask][anomaly_mask],
#         'YMAX_second': YMAX[non_satellite_mask][anomaly_mask],
#     }

#     # Собираем ошибки для сохранения
#     errors = {
#         "err_x_first": erroreX[first_cluster_mask],
#         "err_y_first": erroreY[first_cluster_mask],
#         "err_x_second": erroreX[non_satellite_mask][anomaly_mask],
#         "err_y_second": erroreY[non_satellite_mask][anomaly_mask]
#     }


#     # Сохранение результатов обеих кластеризаций в файл
#     # Используем локальную функцию save_results
#     save_results(coords_first, coords_second, base_filename, fits_file_path, x_y_a_b_values, errors)

#     # Визуализация 1: TH vs ELONG
#     # Логика визуализации оставлена без изменений, кроме использования рассчитанных ELONG и TH
#     fig1, ax1 = plt.subplots(figsize=(12, 9))

#     # Все объекты
#     ax1.scatter(TH, ELONG, c='blue', s=10, label='All data')

#     # «Спутники» по первой кластеризации (выбросы DBSCAN)
#     if np.any(first_cluster_mask):
#         ax1.scatter(TH[first_cluster_mask], ELONG[first_cluster_mask],
#                     facecolors='none', edgecolors='red', s=200, label='First cluster (Satellites)')

#     # «Аномалии» по второй кластеризации (выбросы DBSCAN на не-спутниках)
#     if np.any(anomaly_mask):
#         ax1.scatter(TH_non_satellite[anomaly_mask], ELONG_non_satellite[anomaly_mask],
#                     facecolors='none', edgecolors='green', s=200,
#                     label='Second cluster (Anomalies/Tracks)')

#     ax1.set_xlabel('Angle (TH) [deg]')
#     ax1.set_ylabel('Elongation (A/B)')
#     ax1.set_title('DBSCAN in (TH, A/B) feature space')
#     ax1.legend()
#     ax1.grid(True)
#     plt.tight_layout()
#     # plt.show() # Показ графика


#     # Визуализация 2: Scatter plot XY с выделением кластеров
#     fig2, ax2 = plt.subplots(figsize=(10, 8))
#     ax2.scatter(X, Y, c='blue', s=5, alpha=0.5, label='All Objects')
#     if np.any(first_cluster_mask):
#         ax2.scatter(X[first_cluster_mask], Y[first_cluster_mask], facecolors='none', edgecolors='red', s=150, label='First Cluster (Satellites)')
#     if np.any(anomaly_mask):
#          ax2.scatter(X_non_satellite[anomaly_mask], Y_non_satellite[anomaly_mask], facecolors='none', edgecolors='green', s=150, label='Second Cluster (Anomalies)')

#     ax2.set_xlabel('X [pixel]')
#     ax2.set_ylabel('Y [pixel]')
#     ax2.set_title('Object locations with clusters')
#     ax2.set_aspect('equal', adjustable='box') # Сохраняем пропорции
#     ax2.invert_yaxis() # Ось Y в астрономических изображениях часто инвертирована
#     ax2.legend()
#     ax2.grid(True)
#     plt.tight_layout()
#     # plt.show() # Показ обоих графиков

# # Точка входа в скрипт. В оригинале отсутствовал __main__ блок, но добавим его
# # для лучшей практики, хотя bash скрипт вызывает его напрямую.
# if __name__ == "__main__":
#     main()