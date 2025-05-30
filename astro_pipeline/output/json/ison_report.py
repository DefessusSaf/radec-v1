import xml.etree.ElementTree as ET
import os

def ra_to_dec(ra):
    h, m, s = map(float, ra.split(":"))
    ra_to_dec = h + m / 60 + s / 3600
    return ra_to_dec

def dec_to_dec(dec):
    sign = -1 if dec.startswith("-") else 1
    d, m, s = map(float, dec.lstrip("+-").split(":"))
    dec_to_dec = sign * (d + m / 60 + s / 3600)
    return dec_to_dec

def parse_data_with_filename(raw_data, sensor_id=10121):
    """
    Преобразует данные в структуру, добавляя информацию о файле.
    
    :param raw_data: Список строк данных (имя файла, временная метка, координаты).
    :param sensor_id: Идентификатор источника данных.
    :return: Список словарей, соответствующих объектам.
    """
    # Извлекаем имя файла и временную метку
    file_name = raw_data[0].split(": ")[1].strip()  # Первой строкой - имя файла
    utc = raw_data[1].strip()  # Вторая строка - временная метка
    
    # Остальные строки - координаты объектов
    coordinates = raw_data[2:]
    parsed_data = []
    
    # Преобразуем данные координат в структуру
    for idx, coord in enumerate(coordinates, start=1):
        parts = coord.strip().split()  # Разделяем все значения по пробелам
        ra, dec = parts[0], parts[1]  # RA и Dec
        x = float(parts[2])  # x
        y = float(parts[3])  # y
        x_error = float(parts[4])  # x_error
        y_error = float(parts[5])  # y_error
        a = float(parts[6])  # a
        b = float(parts[7])  # b
        x_min = float(parts[8])  # x_min
        y_min = float(parts[9])  # y_min
        x_max = float(parts[10])  # x_max
        y_max = float(parts[11])  # y_max
        
        # Преобразуем RA и Dec в их десятичные значения
        ra_dec = ra_to_dec(ra)
        dec_dec = dec_to_dec(dec)
        
        parsed_data.append({
            "file": file_name,
            "sensor": sensor_id,
            "id": idx,  # Генерация ID объекта
            "utc": utc,
            "ra_j2000": str(ra_dec),
            "dec_j2000": str(dec_dec),
            "x": x,
            "y": y,
            "x_error": x_error,
            "y_error": y_error,
            "a": a,
            "b": b,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "mag": None,
            "suspicious": False,
        })
    
    return parsed_data

def create_ison_report(data, output_file):
    """
    Создает отчет в формате XML из переданных данных.
    
    :param data: Список данных, которые нужно записать.
    :param output_file: Путь к файлу для сохранения отчета.
    """
    root = ET.Element('data')
    
    for entry in data:
        meas = ET.SubElement(root, 'meas')
        ET.SubElement(meas, 'sensor').text = str(entry['sensor'])
        ET.SubElement(meas, 'id').text = str(entry['id'])
        ET.SubElement(meas, 'utc').text = entry['utc']
        ET.SubElement(meas, 'file').text = entry['file']
        ET.SubElement(meas, 'ra_j2000').text = entry['ra_j2000']
        ET.SubElement(meas, 'dec_j2000').text = entry['dec_j2000']
        # Добавляем новые параметры в XML
        ET.SubElement(meas, 'x').text = str(entry['x'])
        ET.SubElement(meas, 'y').text = str(entry['y'])
        ET.SubElement(meas, 'x_error').text = str(entry['x_error'])
        ET.SubElement(meas, 'y_error').text = str(entry['y_error'])
        ET.SubElement(meas, 'a').text = str(entry['a'])
        ET.SubElement(meas, 'b').text = str(entry['b'])
        ET.SubElement(meas, 'x_min').text = str(entry['x_min'])
        ET.SubElement(meas, 'y_min').text = str(entry['y_min'])
        ET.SubElement(meas, 'x_max').text = str(entry['x_max'])
        ET.SubElement(meas, 'y_max').text = str(entry['y_max'])
        if entry['mag'] is not None:
            ET.SubElement(meas, 'mag').text = str(entry['mag'])
        ET.SubElement(meas, 'suspicious').text = str(entry['suspicious']).lower()
    
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def process_file_in_dir(dir):
    all_data = []
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                raw_data = file.readlines()
            raw_data = [line.strip() for line in raw_data if line.strip() and not line.startswith("#")]
            
            parsed_data = parse_data_with_filename(raw_data)
            all_data.extend(parsed_data)
    
    return all_data

dir = "PROCESS_FILE"

# Обрабатываем данные
all_data  = process_file_in_dir(dir)
# Создаем отчет в XML формате
create_ison_report(all_data, "ison_report.xml")

print("Successfuly!")