# config_setting.py
import subprocess
import sys
import os
import shutil
import timeit
import numpy as np
from astropy.io import fits
import astropy.units as u
import logging
from pathlib import Path 
import path 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FOCALLEN = 551 * u.mm


def extract_header_and_scale(fits_file_path: Path):
    """
    Extracts the necessary data from the FITS file header and calculates the image scale.

    Args:
        fits_file_path (Path): Путь к входному FITS-файлу (объект Path).

    Returns:
        Tuple: Cortege (xpixsz_um, gain, exptime, scale_arcsec_per_pix).
    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file is empty or there are no mandatory keys in the header,
            or with an error in transformation of units of measurement.
        KeyError: If there are no Gain/Egain in the header.
        Exception: With other errors of reading the heading.
    """
    start = timeit.default_timer()
    logging.info(f"Открытие FITS файла: {fits_file_path}")
    try:
        # Открываем файл, используя объект Path
        with fits.open(fits_file_path) as hdul:
            if not hdul:
                raise ValueError("Failed to open a FITS file or it is empty.")
            header = hdul[0].header

            try:
                xpixsz_um = header["XPIXSZ"]
                exptime = header["EXPTIME"]
            except KeyError as e:
                logging.error(f"There is no required key in the header: {e}")
                raise

            if "EGAIN" in header:
                gain = header["EGAIN"]
                logging.info(f"Used EGAIN: {gain}")
            elif "GAIN" in header:
                gain = header["GAIN"]
                logging.info(f"Used GAIN: {gain}")
            else:
                raise KeyError("Absent GAIN or EGAIN in FITS header.")

            pixel_size = xpixsz_um * u.micron

            try:
                # 1. Вычисляем безразмерное отношение (размер пикселя / фокусное расстояние)
                dimensionless_ratio = (pixel_size / FOCALLEN).to(u.dimensionless_unscaled)
                # 2. Физически это отношение соответствует углу в радианах для пикселя
                #    Назначаем нужные единицы
                scale_rad_per_pix = dimensionless_ratio * u.rad / u.pixel
                # 3. Конвертируем в угловые секунды на пиксель
                scale_arcsec_per_pix_quantity = scale_rad_per_pix.to(u.arcsec / u.pixel)
                scale_arcsec_per_pix = scale_arcsec_per_pix_quantity.value

                logging.info(f"Pixel size (XPIXSZ): {xpixsz_um} um")
                logging.info(f"Focal length: {FOCALLEN}")
                logging.info(f"Calculated scale: {scale_arcsec_per_pix_quantity:.4f}")

            except u.UnitConversionError as e:
                 logging.error(f"Error when converting units of measurement when calculating scale: {e}")
                 raise ValueError("Incompatible units of measurements for calculating scale.") from e

    except FileNotFoundError:
        logging.error(f"File not found: {fits_file_path}")
        raise
    except Exception as e:
        logging.error(f"There was an error when reading the header FITS: {e}")
        raise

    time_taken = timeit.default_timer() - start
    logging.info(f"The title is extracted and the scale is designed for {time_taken:.4f} seconds.")

    return xpixsz_um, gain, exptime, scale_arcsec_per_pix


def calculate_initial_sextractor_params(scale_arcsec_per_pix: float):
    """
    Calculates the initial parameters for SExtractor.
    At the moment, it simply returns rigidly set initial values.

    Args:
        scale_arcsec_per_pix (float): Image scale in ArcSec/Pixel (for information).

    Returns:
        Tuple: Cortege (detect_minarea, detect_thresh)
        - detect_minarea (int): The initial minimum area of ​​the object in pixels.
        - detect_thresh (float): The initial threshold of detection (in the sigma above the background).
    """
    start = timeit.default_timer()

    DETECT_MINAREA = 9
    logging.info(f"The initial value of Detect_Minarea is set in {DETECT_MINAREA} pix.")

    DETECT_THRESH = 1.5
    logging.info(f"The initial value of Detect_thresh is installed in {DETECT_THRESH} sigma.")

    time_taken = timeit.default_timer() - start
    logging.info(f"The initial Sextractor parameters are designed for {time_taken:.4f} seconds.")
    return DETECT_MINAREA, DETECT_THRESH


def run_sextractor_and_count(
    fits_file_path: Path,
    config_file_path: Path,
    temp_dir_path: Path,
    catalog_output_path: Path,
    detect_minarea: float,
    detect_thresh: float
):
    """
    Start SExtractor with specified parameters and calculates the number of objects discovered
    From the catalog file.

    Args:
        Fits_file_path (Path): the path to the input fits file (object Path).
        Config_file_path (Path): Way to the SExtractor (.SEX) configuration file.
        TEMP_DIR_PATH (PATH): the path to the temporary directory for some output files (for example, Checkimage) (Path object).
        Catalog_outPut_path (Path): the expected path to the output file of the SExtractor catalog (Path object).
        Detect_Minarea (Float): The value of the Detect_Minarea parameter.
        DETECT_THRESH (FLOAT): Detect_thresh parameter value.

    Returns:
        int: The number of objects detected.
    Raises:
        FileNotFoundError: If the SExtractor configuration file is not found or the expected catalog is not created.
        RuntimeError: If the SExtractor ends with the error.
        IOError: If an error occurred when reading a catalog file.
    """
    start = timeit.default_timer()

    if not config_file_path.exists():
         logging.error(f"Configuration file SExtractor not was found: {config_file_path}")
         raise FileNotFoundError(f"Configuration file SExtractor not was found: {config_file_path}")

    # Убедимся, что директория для каталога существует (родительская директория для catalog_output_path)
    catalog_output_path.parent.mkdir(parents=True, exist_ok=True)
    # logging.info(f"Создана/существует директория для каталога: {catalog_output_path.parent}")

    # Убедимся, что временная директория для чек-изображений существует
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    # logging.info(f"Создана/существует временная директория: {temp_dir_path}")


    base_name = fits_file_path.stem # Имя файла без расширения
    check_image_path = temp_dir_path / f"{base_name}_check.fits" # Используем Path для объединения путей

    # Формируем команду для запуска SExtractor
    # Путь к исполняемому файлу 'sex' предполагается в PATH системы
    sextractor_command = [
        "sex", str(fits_file_path), 
        "-c", str(config_file_path), 
        "-PARAMETERS_NAME", str(path.CONFIGS_DIR / "default.param"),
        "-FILTER_NAME", str(path.CONFIGS_DIR / "gauss_4.0_7x7.conv"),
        "-DETECT_MINAREA", f"{detect_minarea:.2f}",
        "-DETECT_THRESH", f"{detect_thresh:.2f}",
        "-CATALOG_NAME", str(catalog_output_path), 
        "-CHECKIMAGE_NAME", str(check_image_path) 
    ]

    logging.info(f"Start SExtractor с DETECT_MINAREA={detect_minarea:.2f}, DETECT_THRESH={detect_thresh:.2f}")
    logging.info(f"The expected catalog file: {catalog_output_path}")
    logging.debug(f"Command: {' '.join(map(str, sextractor_command))}") # Используем map(str, ...) для корректного вывода Path объектов

    try:
        # Запускаем SExtractor
        result = subprocess.run(sextractor_command, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logging.error(f"SExtractor ended with an error (code {result.returncode}).")
            logging.error(f"Stderr:\n{result.stderr}")
            # Важно: Даже если SEXTRACTOR упал, он мог создать пустой или неполный каталог
            # Проверка существования каталога будет ниже
            raise RuntimeError(f"Error execution SExtractor.")
        else:
            logging.info("SExtractor Successfully completed.")
            # logging.debug(f"Stdout:\n{result.stdout}")
            if result.stderr:
                 # Stderr может содержать предупреждения даже при успешном выполнении
                 logging.warning(f"Stderr (Possible warnings):\n{result.stderr}")

        logging.info(f"Reading the catalog file: {catalog_output_path}")
        # Проверяем, был ли создан файл каталога и не пустой ли он
        if not catalog_output_path.exists() or catalog_output_path.stat().st_size == 0:
             # Эта ошибка может произойти, если Sextractor упал, и если он отработал, но не создал файл
             logging.error(f"The expected catalog file was not found or empty: {catalog_output_path}")
             raise FileNotFoundError(f"Catalog file {catalog_output_path} not found or empty after execution SExtractor.")

        # Читаем количество строк в файле каталога (количество объектов)
        try:
            with catalog_output_path.open("r") as catalog: # Используем .open() для Path объекта
                # Фильтруем строки, чтобы не считать комментарии (начинаются с #)
                object_count = sum(1 for line in catalog if line.strip() and not line.strip().startswith('#'))
        except Exception as e:
            logging.error(f"Error when reading a catalog file {catalog_output_path}: {e}")
            raise IOError(f"Error when reading a catalog file {catalog_output_path}: {e}")


    except FileNotFoundError as e:
        # Сюда попадем, если не найден config_file_path
        raise
    except RuntimeError as e:
        # Сюда попадем при ошибке выполнения SExtractor
        raise
    except IOError as e:
         # Сюда попадем при ошибке чтения каталога
         raise
    except Exception as e:
        logging.error(f"An error occurred during the performance or processing of the results SExtractor: {e}")
        raise

    time_taken = timeit.default_timer() - start
    logging.info(f"SExtractor I processed the file and found{object_count} objects (from {catalog_output_path}) For {time_taken:.4f} seconds.")

    return object_count


def cleanup_temp_dir(temp_dir_path: Path):
    """
    Removes the temporary directory and its contents.
    Args:
        temp_dir_path (Path): The path to the temporary directory (object Path).
    """
    start = timeit.default_timer()
    if temp_dir_path.exists():
        try:
            # Используем shutil.rmtree, который работает с Path объектами
            shutil.rmtree(temp_dir_path)
            logging.info(f"Temporary Directory {temp_dir_path} Successfully removed.")
        except OSError as e:
            logging.error(f"Failed to remove the temporary directory {temp_dir_path}: {e}")
    else:
        logging.info(f"Temporary Directory {temp_dir_path} not found, cleaning is not required.")
    time_taken = timeit.default_timer() - start
    logging.info(f"Cleaning is performed for {time_taken:.4f} seconds.")


def main():
    """
    The main function of the script.Processing FITS-файл, start SExtractor
    to achieve the target number of objects and cleans temporary files.
    """
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_FITS_file>")
        sys.exit(1)

    # Преобразуем входной путь в объект Path
    fits_file_path = Path(sys.argv[1])

    # Пути берем из нашего модуля path
    temp_dir_path = path.TEMP_SEXTRACTOR_DIR
    config_file_path = path.SEX_CONFIG_FILE
    catalog_output_path = path.SEX_CATALOG_FILE # Путь к выходному каталогу SExtractor


    if not fits_file_path.exists():
        logging.error(f"The entrance FITS file was not found: {fits_file_path}")
        sys.exit(1)
    # Проверка config_file_path теперь делается внутри run_sextractor_and_count,
    # но можно оставить здесь для быстрой проверки до запуска SExtractor
    if not config_file_path.exists():
         logging.error(f"Configuration file SExtractor was not found: {config_file_path}")
         sys.exit(1)

    # Создаем временную директорию для SExtractor перед началом работы
    try:
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Temporary Directory for SExtractor Created/exists: {temp_dir_path}")
    except OSError as e:
        logging.error(f"Failed to create a temporary directory for SExtractor {temp_dir_path}: {e}")
        sys.exit(1)


    try:
        # Извлекаем параметры из заголовка
        xpixsz_um, gain, exptime, scale_arcsec_per_pix = extract_header_and_scale(fits_file_path)
        logging.info(f"Parameters from heading: XPIXSZ={xpixsz_um} um, GAIN={gain}, EXPTIME={exptime} s")

        # Рассчитываем начальные параметры SExtractor
        detect_minarea, detect_thresh = calculate_initial_sextractor_params(scale_arcsec_per_pix)

        # Целевое количество объектов и параметры итераций
        target_objects = 80
        max_iter = 50
        min_minarea = 1
        adjustment_factor = 0.9

        current_minarea = float(detect_minarea)
        final_minarea = current_minarea
        found_target = False

        logging.info(f"The beginning of an iterative search {target_objects} objects (max. Iterations: {max_iter}).")

        # Итеративный процесс подбора DETECT_MINAREA
        for iteration in range(max_iter):
            logging.info(f"--- Iteration {iteration + 1}/{max_iter} ---")
            try:
                # Запускаем SExtractor с текущими параметрами
                detect_obj_count = run_sextractor_and_count(
                    fits_file_path,
                    config_file_path,
                    temp_dir_path,
                    catalog_output_path, # Передаем путь к каталогу
                    current_minarea,
                    detect_thresh
                )

                # Проверяем количество найденных объектов
                if detect_obj_count >= target_objects:
                    logging.info(f"Found {detect_obj_count} objects (>= {target_objects}) with DETECT_MINAREA = {current_minarea:.2f}")
                    final_minarea = current_minarea
                    found_target = True
                    break # Цель достигнута, выходим из цикла

                else:
                    logging.info(f"found {detect_obj_count} objects (< {target_objects}). Reduce DETECT_MINAREA.")
                    current_minarea *= adjustment_factor
                    # Ограничиваем минимальное значение DETECT_MINAREA
                    if current_minarea < min_minarea:
                        logging.warning(f"DETECT_MINAREA ({current_minarea:.2f}) reached a minimum limit ({min_minarea}). Stopping iterations.")
                        final_minarea = min_minarea
                        try:
                            # Последний запуск с минимальным DETECT_MINAREA
                            detect_obj_count = run_sextractor_and_count(
                                fits_file_path,
                                config_file_path,
                                temp_dir_path,
                                catalog_output_path,
                                final_minarea,
                                detect_thresh
                            )
                            logging.info(f"The number of objects with minimal DETECT_MINAREA ({final_minarea:.2f}): {detect_obj_count}")
                        except Exception as final_run_e:
                            logging.error(f"Error at the last launch SExtractor with min_minarea: {final_run_e}")
                        break # Достигнут минимум, выходим из цикла

            except (FileNotFoundError, RuntimeError, IOError) as e:
                 logging.error(f"An error for iteration {iteration + 1}: {e}. Interruption of the cycle.")
                 # В случае ошибки очищаем временные файлы перед выходом
                 cleanup_temp_dir(temp_dir_path)
                 sys.exit(1)
            except Exception as e:
                 logging.exception(f"Unexpected error for iteration {iteration + 1}.") # Логируем traceback
                 # В случае ошибки очищаем временные файлы перед выходом
                 cleanup_temp_dir(temp_dir_path)
                 sys.exit(1)

        if found_target:
            print(f"\nSuccess!A sufficient number of objects found.")
        elif iteration == max_iter - 1 and not found_target:
             logging.warning(f"The iteration limit has been reached ({max_iter}). It was not possible to find {target_objects} objects.")
             print(f"\nIt was not possible to find {target_objects} objects for {max_iter} iterations.")
        else:
             print(f"\nThe search is stopped due to achievementминимального DETECT_MINAREA ({final_minarea:.2f}).")

        print(f"Final meaning DETECT_MINAREA: {final_minarea:.2f}")
        print(f"Final meaning DETECT_THRESH: {detect_thresh:.2f}")

    except (FileNotFoundError, KeyError, ValueError, RuntimeError, IOError) as e:
        logging.error(f"A critical error occurred: {e}")
        print(f"Error execution: {e}")
        # В случае ошибки очищаем временные файлы перед выходом
        cleanup_temp_dir(temp_dir_path)
        sys.exit(1)
    except Exception as e:
        logging.exception("An unforeseen mistake occurred:") # Логируем traceback
        print(f"An unforeseen mistake: {e}")
        cleanup_temp_dir(temp_dir_path)
        sys.exit(1)
    finally:
        cleanup_temp_dir(temp_dir_path)
        logging.info("The work of the script config_setting.py completed.")


if __name__ == "__main__":
    main()