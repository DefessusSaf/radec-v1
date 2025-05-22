import os
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
import numpy as np
from sklearn.cluster import DBSCAN

# Параметры
path = "testProcces"

ra_val = []
dec_val = []
ra_str_val = []
dec_str_val = []
file_names = []

# Пороговые значения для DBSCAN
eps = 0.5  # Радиус соседства (в градусах)
min_samples = 10  # Минимальное число точек в кластере


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

# Преобразование RA и DEC в градусы
ra_degrees, dec_degrees = ra_dec_to_degrees(ra_str_val, dec_str_val)

# Применение DBSCAN
data = np.array(list(zip(ra_degrees, dec_degrees)))
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data)

# Результаты кластеризации
unique_labels = set(labels)
print(f"Кластеры: {unique_labels}")

# Сохранение результатов в файл
output_file = "dbscan_clustering.txt"
with open(output_file, "w") as f:
    f.write("RA\tDEC\tCluster_ID\n")
    for ra, dec, cluster_id in zip(ra_val, dec_val, labels):
        f.write(f"{ra:.6f}\t{dec:.6f}\t{cluster_id}\n")
print(f"Результаты кластеризации сохранены в файл {output_file}")

# Построение графика
fig, ax = plt.subplots()

colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    if label == -1:  # Кластер шума
        color = "black"
        marker = "x"
    else:
        marker = "o"
    class_member_mask = labels == label
    ax.scatter(
        data[class_member_mask, 0], 
        data[class_member_mask, 1], 
        c=[color], 
        label=f"Cluster {label}" if label != -1 else "Noise", 
        s=50, 
        edgecolors="k", 
        marker=marker
    )

ax.set_xlabel("RA (degrees)")
ax.set_ylabel("DEC (degrees)")
ax.legend()
ax.grid(True)
plt.tight_layout()

# Сохранение графика
output_image = "dbscan_clusters.png"
plt.savefig(output_image)
print(f"График кластеров сохранен в файл {output_image}")
plt.show()