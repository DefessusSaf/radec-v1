# region.py 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:37:37 2024

@author: alex

Rewrite results after SExtractor for region plot on DS9, etc.

Input information format:
    X_IMAGE         Object position along x                     [pixel]
    Y_IMAGE         Object position along y                     [pixel]
    ERRCXX_IMAGE    Cxx error ellipse parameter                 [pixel**(-2)]
    ERRCYY_IMAGE    Cyy error ellipse parameter                 [pixel**(-2)]
    A_IMAGE         Profile RMS along major axis                [pixel]
    B_IMAGE         Profile RMS along minor axis                [pixel]
    XMIN_IMAGE      Minimum x-coordinate among detected pixels  [pixel]
    YMIN_IMAGE      Minimum y-coordinate among detected pixels  [pixel]
    XMAX_IMAGE      Maximum x-coordinate among detected pixels  [pixel]
    YMAX_IMAGE      Maximum y-coordinate among detected pixels  [pixel]
    THETA_IMAGE     Position angle (CCW/x)                      [deg]
    FLAGS           Extraction flags                                         
    FLUX_ISOCOR     Corrected isophotal flux                   [count]

output format:
    ellipse(   269.4581,   397.0928,   46.4340,   1.7650,  -69.43)
    
Launch under the sheath Astroconda (there is a conflict in Numpy versions  
1.21.5 в astropy vs. 1.26.4 в seiferts etc.

"""

import numpy as np
from os import system 
from pathlib import Path 
from src.utils import path
from src.utils import utils


sextractor_results_file = path.SEX_CATALOG_FILE
region_output_file = path.REGION_FILE 
xy_txt_file = path.XY_TXT_FILE 
xy_fits_file = path.XY_FITS_FILE 


# --- Загрузка данных ---
# Используем универсальную функцию загрузки из utils.py
try:
    X, Y, ERRX, ERRY, A, B, XMIN, YMIN, XMAX, YMAX, TH, FLAG, FLUX = utils.load_sextractor_genfromtxt(str(sextractor_results_file))
    # load_sextractor_genfromtxt возвращает распакованные массивы, как np.genfromtxt(..., unpack=True)
except FileNotFoundError:
    print(f"Error: SExtractor results were not found on the way {sextractor_results_file}.")
    exit(1) # Выходим, если файл не найден
except IOError as e:
    print(f"Error: Failed to download data from the results file SExtractor: {e}")
    exit(1) # Выходим при ошибке чтения

# --- Создание Region File для DS9 ---
output_lines = []
# zip объединяет элементы массивов по индексу, удобно для итерации
for x, y, a, b, th in zip(X, Y, A, B, TH):
    # Форматируем строку для файла региона
    output_lines.append(f'ellipse( {x} , {y} , {a:.4f} , {b:.4f},{th:.2f})')

# Сохраняем строки в файл
try:
    # Убедимся, что родительская директория существует
    region_output_file.parent.mkdir(parents=True, exist_ok=True)
    # Используем .open() для записи, или str() для совместимости с np.savetxt если требуется
    np.savetxt(str(region_output_file), output_lines, fmt='%s')
    print(f"Region file saved to {region_output_file}")
except Exception as e:
    print(f"Error saving region file {region_output_file}: {e}")


# --- Создание файла XY.txt и преобразование в XY.fits ---

# Собираем нужные данные в двумерный массив (X, Y, FLUX)
arr = np.vstack((X, Y, FLUX))

# Сохраняем массив в текстовый файл
try:
    # Убедимся, что родительская директория существует
    xy_txt_file.parent.mkdir(parents=True, exist_ok=True)
    # Используем str() для совместимости с np.savetxt
    np.savetxt(str(xy_txt_file), arr.T, fmt=('%s %s %s'), header='X_CENTER  Y_CENTER  AREA')
    print(f"XY text file saved to {xy_txt_file}")
except Exception as e:
    print(f"Error saving XY text file {xy_txt_file}: {e}")


# Преобразование текстового файла XY.txt в FITS XY.fits с помощью утилиты text2fits
# Получаем путь к утилите из нашего модуля path
text2fits_cmd_path = path.get_text2fits_path()

# Формируем команду для запуска утилиты
# Используем str() для преобразования Path объектов в строки
cmd = f'{str(text2fits_cmd_path)} {str(xy_txt_file)} {str(xy_fits_file)}'

print(f"Executing command: {cmd}")

# Запускаем внешнюю команду
try:
    # Убедимся, что родительская директория для выходного FITS существует
    xy_fits_file.parent.mkdir(parents=True, exist_ok=True)
    # Используем os.system. Можно рассмотреть переход на subprocess для лучшего контроля.
    system(cmd)
    print(f"XY FITS file created at {xy_fits_file}")
except Exception as e:
    print(f"Error executing text2fits command: {e}")
    print("Please ensure text2fits is installed and accessible via the paths specified in path.py")
    # Не вызываем exit(1) здесь, чтобы остальная часть конвейера могла попытаться работать,
    # но если XY.fits критически важен, возможно, стоит выйти.
    # monitor2.bash уже проверяет существование TMP/XY.fits после этого шага.
