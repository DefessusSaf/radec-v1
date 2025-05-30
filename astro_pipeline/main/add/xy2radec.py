import sys
import os
from astropy.wcs import WCS
from astropy.io import fits
import numpy as np

# Получаем имя файла из аргументов командной строки
if len(sys.argv) < 2:
    print("Ошибка: не передано имя файла.")
    sys.exit(1)

newfile = sys.argv[1]
print(f'Обрабатываем файл: {newfile}')

filename = os.path.splitext(os.path.basename(newfile))[0]

# Читаем WCS из FITS-файла (замените имя файла WCS на соответствующее)
wcs_file = None
extensions = ['.fits', '.fit', '.fts']
for ext in extensions:
    wcs_file_ext = f"TMP/WCS_{filename}{ext}"
    if os.path.exists(wcs_file_ext):
        wcs_file = wcs_file_ext
        break

if not os.path.exists(wcs_file):
    print(f"Errore: {wcs_file} not found")
    sys.exit(1)
print(f'Используем WCS-файл: {wcs_file}')


# Читаем данные WCS
with fits.open(wcs_file) as hdu:
    wcs = WCS(hdu[0].header)

# Чтение XY.txt файла
xy_file = 'TMP/XY.txt'
data = np.loadtxt(xy_file, skiprows=1)
x_values, y_values, areas = data[:, 0], data[:, 1], data[:, 2]

# Преобразование координат x и y в ra и dec
ra_dec_values = wcs.pixel_to_world(x_values, y_values)

# Сохранение нового .txt файла с ra и dec
output_file = f'PROCESS_FILE/RA_DEC_output_{filename}.txt'
with open(output_file, 'w') as f_out:
    f_out.write('# RA  DEC  AREA\n')
    for (ra, dec, area) in zip(ra_dec_values.ra.deg, ra_dec_values.dec.deg, areas):
        f_out.write(f'{ra:.6f}  {dec:.6f}  {area:.6f}\n')

print(f'Файл {output_file} успешно создан.')