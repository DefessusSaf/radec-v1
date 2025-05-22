import subprocess
import re
import sys
import os
import shutil
import timeit
import numpy as np
from astropy.io import fits
import concurrent.futures


# Извлечение данных из заголовка FITS-файла
def extract_header(fits_file):
    start = timeit.default_timer()
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        xpixsz = header["XPIXSZ"]
        exptime = header["EXPTIME"]
        
        if "EGAIN" in header:
            egain = header["EGAIN"]
            gain = egain
            print(f"EGAIN: {egain} to GAIN: {gain}")
        elif "GAIN" in header:
            gain = header["GAIN"]
            print(f"GAIN: {gain}")
        else: 
            raise KeyError(f"No GAIN or EGAIN in header")   
        time = timeit.default_timer() - start
        print(f"TIME: {time}")
        
    return xpixsz, gain, exptime


# Асинхронное вычисление масштаба изображения
def calcul_scale_async(fits_file, temp_dir):
    start = timeit.default_timer()
    
    result = subprocess.run(
        ["solve-field", "--scale-units", "arcsecperpix", "--no-plots", "--overwrite", "--downsample", "2", "--dir", temp_dir, fits_file],
        capture_output=True, text=True
    )
    
    print("solve-field:", result.stdout)
    print("solve-files:", result.stderr)

    # Извлечение масштаба из вывода solve-field
    scale_match = re.search(r"pixel scale (\d+\.\d+)", result.stdout)
    if not scale_match:
        raise RuntimeError("Failed to calculate scale")
    time = timeit.default_timer() - start
    print(f"TIME: {time}")
    return float(scale_match.group(1))


# Вычисление параметров DETECT_MINAREA и DETECT_THRESH
def calculate_param(xpixsz, gain, exptime, scale):
    start = timeit.default_timer()

    D = xpixsz * scale
    DETECT_MINAREA = (D / scale) ** 2  # Это может быть упрощено до D^2, если scale не меняется
    
    signal = gain * exptime
    noise = np.sqrt(signal)
    DETECT_THRESH = signal / noise
    time = timeit.default_timer() - start
    print(f"TIME: {time}")
    return DETECT_MINAREA, DETECT_THRESH


# Выполнение SExtractor и подсчёт количества объектов
def detected_obj(fits_file, config_file, temp_dir, detect_minarea, detect_thresh):
    start = timeit.default_timer()

    sextractor_command = [
        "sex", fits_file,
        "-c", config_file,
        "-DETECT_MINAREA", f"{detect_minarea:.2f}",
        "-DETECT_THRESH", f"{detect_thresh:.2f}",
        "-CHECKIMAGE_NAME", f"{temp_dir}/check.fits"
    ]

    subprocess.run(sextractor_command)
    
    # Чтение каталога объектов
    catalog_file = "TMP/k1-impTEST.fts.sx"
    if not os.path.exists(catalog_file):
        raise FileNotFoundError(f"{catalog_file} not found")

    with open(catalog_file, "r") as catalog:
        lines = catalog.readlines()
    time = timeit.default_timer() - start
    print(f"TIME: {time}")
    return len(lines)


# Очистка временной директории после выполнения
def cleanup_temp_dir(temp_dir):
    start = timeit.default_timer()

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Directory {temp_dir} has been cleaned up.")
    time = timeit.default_timer() - start
    print(f"TIME: {time}")


# Основная функция
def main():
    # fits_file = "testData/Meteor1_19R_00045.fits"
    # Параметры запуска
    fits_file = sys.argv[1]  # Имя FITS файла передаётся в качестве аргумента
    temp_dir = "./temp_files"
    config_file = "CONFIGS/defaultTEST.sex"
    # config_file = "CONFIGS/default2.sex"
    
    
    # Создание временной директории
    os.makedirs(temp_dir, exist_ok=True)

    # Извлекаем данные заголовка FITS
    xpixsz, gain, exptime = extract_header(fits_file)

    # Параллельный запуск вычисления масштаба изображения
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_scale = executor.submit(calcul_scale_async, fits_file, temp_dir)
        scale = future_scale.result()  
        print(f"SCALE: {scale}")# Ждём завершения параллельного выполнения
    
    # Вычисление начальных параметров
    detect_minarea, detect_thresh = calculate_param(xpixsz, gain, exptime, scale)

    target_objects = 80
    max_iter = 45
    iter = 0

    # Итеративное уменьшение DETECT_MINAREA для нахождения объектов
    while iter < max_iter:
        detect_obj_count = detected_obj(fits_file, config_file, temp_dir, detect_minarea, detect_thresh)

        if detect_obj_count >= target_objects:
            print(f"Found: {detect_obj_count} objects with DETECT_MINAREA: {detect_minarea:.2f}")
            break

        detect_minarea *= 0.9  # Уменьшаем DETECT_MINAREA для следующей итерации
        iter += 1

    if iter == max_iter:
        print(f"Could not find {target_objects} objects with the current settings.")
    
    print(f"Final DETECT_MINAREA: {detect_minarea:.2f}")
    print(f"Final DETECT_THRESH: {detect_thresh:.2f}")

    # Очистка временных файлов
    cleanup_temp_dir(temp_dir)

    print("All temporary files created by calcul_param have been deleted.")


if __name__ == "__main__":
    main()