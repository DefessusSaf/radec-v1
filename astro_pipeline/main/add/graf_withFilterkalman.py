import os
import matplotlib.pyplot as plt
import mplcursors
from astropy.coordinates import Angle
import numpy as np
from scipy.interpolate import interp1d

# Параметры
# path = "PROCESS_FILE"
path = "testProcces"

ra_val = []
dec_val = []
ra_str_val = []
dec_str_val = []
file_names = []
file_map = []

# Пороговые значения
ra_break_threshold = 300  # Разрыв по RA (в градусах)
max_ra_dist = 360  # Максимальное расстояние по RA
max_dec_dist = 50  # Максимальное расстояние по DEC


# Преобразование RA и DEC
def ra_to_dec(ra):
    h, m, s = map(float, ra.split(":"))
    return (h + m / 60 + s / 3600) * 15


def dec_to_dec(dec):
    d, m, s = map(float, dec.split(":"))
    if d < 0:
        return d - m / 60 - s / 3600
    return d + m / 60 + s / 3600


# Преобразование RA и DEC в градусы (с использованием Astropy)
def ra_dec_to_degrees(ra_values, dec_values):
    ra_degrees = [Angle(ra, unit='hourangle').degree for ra in ra_values]
    dec_degrees = [Angle(dec, unit='deg').degree for dec in dec_values]
    return ra_degrees, dec_degrees


# Поиск разрывов с учетом цикличности
def find_ra_dec_gaps(ra_degrees, threshold):
    # Сортировка RA
    sorted_ra = np.sort(ra_degrees)
    # Вычисление разностей
    diffs = np.diff(sorted_ra)
    # Обработка перехода через 360°
    diffs = np.append(diffs, (sorted_ra[0] + 360) - sorted_ra[-1])
    # Нахождение разрывов
    gaps = np.where(diffs > threshold)[0]
    return gaps, sorted_ra


# Обработка файлов
def process_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                continue
            try:
                x_str, y_str = line.split()
                ra = ra_to_dec(x_str)
                dec = dec_to_dec(y_str)
                ra_val.append(ra)
                dec_val.append(dec)
                ra_str_val.append(x_str)
                dec_str_val.append(y_str)
                file_names.append(os.path.basename(file_path))
            except ValueError:
                continue




for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    if os.path.isfile(file_path):
        process_file(file_path)

# Преобразование RA и DEC в градусы для поиска разрывов
ra_degrees, dec_degrees = ra_dec_to_degrees(ra_str_val, dec_str_val)

# Нахождение разрывов в RA
ra_gaps, sorted_ra = find_ra_dec_gaps(ra_degrees, ra_break_threshold)

print("BREAK:")
for gap in ra_gaps:
    start_ra = sorted_ra[gap]
    end_ra = sorted_ra[(gap + 1) % len(sorted_ra)]
    
    start_ra_deg = Angle(start_ra, unit="deg")
    end_ra_deg = Angle(end_ra, unit="deg")
    
    start_ra_str = start_ra_deg.to_string(unit="hourangle", sep=":")
    end_ra_str = end_ra_deg.to_string(unit="hourangle", sep=":")
    
    print(f"BREAK between {sorted_ra[gap]:.2f}° and {sorted_ra[(gap + 1) % len(sorted_ra)]:.2f}°")
    print(f"BREAK between {start_ra_str} and {end_ra_str}")
    
    start_dec = dec_degrees[ra_degrees.index(start_ra)]
    end_dec = dec_degrees[ra_degrees.index(end_ra)]
    
    print(f"START RA: {start_ra_str} DEC: {start_dec:.6f}")
    print(f"END RA: {end_ra_str} DEC: {end_dec:.6f}")
    
    is_new_object = abs(start_ra - end_ra) > ra_break_threshold and abs(start_dec - end_dec) > max_dec_dist
    print(abs(start_ra - end_ra))
    print(abs(start_dec - end_dec))
    
    print("NEW OBJ " if is_new_object else "OLD OBJ")
    
    

# Сохранение результатов разрывов
output_gaps_file = "ra_gaps_identification.txt"
with open(output_gaps_file, "w") as gap_file:
    gap_file.write("Start_RA\tEnd_RA\n")
    for gap in ra_gaps:
        start_ra = sorted_ra[gap]
        end_ra = sorted_ra[(gap + 1) % len(sorted_ra)]
        gap_file.write(f"{start_ra:.6f}\t{end_ra:.6f}\n")
print(f"Разрывы сохранены в файл {output_gaps_file}")

def ids_object(ra, dec, break_points, max_ra_dist, max_dec_dist):
    labels = np.zeros(len(ra), dtype=int)
    cluster_id = 1
    for i in range(len(ra)):
        if labels[i] == 0:
            labels[i] = cluster_id
            for j in range(i + 1, len(ra)):
                if labels[j] == 0:
                    if abs(ra[i] - ra[j]) <= max_ra_dist and abs(dec[i] - dec[j]) <= max_dec_dist:
                        labels[j] = cluster_id
            cluster_id += 1
    return labels

# Продолжение обработки (разметка объектов)
break_points = find_ra_dec_gaps(ra_degrees, ra_break_threshold)[0]
object_labels = ids_object(ra_degrees, dec_degrees, break_points, max_ra_dist, max_dec_dist)

# Сохранение результатов объектов
output_file = "objects_identification.txt"
with open(output_file, "w") as f:
    f.write("RA\tDEC\tObject_ID\n")
    for ra, dec, obj_id in zip(ra_val, dec_val, object_labels):
        f.write(f"{ra:.6f}\t{dec:.6f}\t{obj_id}\n")
print(f"Результаты объектов сохранены в файл {output_file}")

unique_files = list(set(file_names))
inx_dir = {}
scatter_dir = {}


# Построение графика
fig, ax = plt.subplots()

for i in range(len(ra_val)):
    color = "blue" if object_labels[i] == 1 else "red"
    ax.scatter(ra_val[i], dec_val[i], color=color, s=20)

for unique_file in enumerate(unique_files):
    inx_file = [j for j in range(len(file_names)) if file_names[j] == unique_file]
    scatter_dir[unique_file] = ax
    inx_dir[unique_file] = inx_file
    




cursor = mplcursors.cursor(ax, hover=True)
# @cursor.connect('add')
# def on_add(sel):
#     # idx = sel.index
#     scatter = sel.artist
#     file_name = [name for name, obj in scatter_dir.items() if obj == scatter][1]
#     inx = sel.index
    
#     # sel.annotation.set_text(f"File: {file_name}\nRA: {sel.target[0]:.6f}\nDEC: {sel.target[1]:.6f}")
#     sel.annotation.set_text(f"File: {file_name}\nRA: {ra_str_val}\nDEC: {dec_str_val}")    
#     # sel.annotation.set_text(f"File: {file_name}\nRA: {ra_val[idx]:.6f}\nDEC: {dec_val[idx]:.6f}")    
    
ax.set_xlabel("RA")
ax.set_ylabel("DEC")
ax.grid(True)
plt.tight_layout()

# output_image = "ra_dec_identification.png"
# plt.savefig(output_image)
# print(f"График сохранён в файл {output_image}")
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Выбор третьей координаты (например, ID объекта)
z_val = object_labels  # Используем ID объектов как третье измерение

for i in range(len(ra_val)):
    color = "blue" if object_labels[i] == 1 else "red"
    ax.scatter(ra_val[i], dec_val[i], ra_val[i], color=color, s=20)

# Подписи осей
ax.set_xlabel("RA (градусы)")
ax.set_ylabel("DEC (градусы)")
ax.set_zlabel("Object ID")



# Сетка
ax.grid(True)

# Отображение графика
plt.tight_layout()
plt.show()
