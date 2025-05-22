import codecs
from os import getcwd, listdir
from astropy.time import Time
from datetime import datetime
import os

COUNTER_FILE = "id_counter.txt"

def get_next_id():
    """
    Generate unique ID
    """
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as file:
            current_id = int(file.read().strip())
    else:
        current_id = 0

    next_id = current_id
    with open(COUNTER_FILE, "w") as file:
        file.write(str(next_id + 1))

    return f"{next_id:07d}"

def gen_out_filename():
    """
    Generate a unique filename based on the current date and scan number.
    """
    today = datetime.now().strftime('%Y%m%d')
    scan_number = 1
    while True:
        filename = f"{today}_SCAN{scan_number}.RES"
        if not os.path.exists(filename):
            return filename
        scan_number += 1

cwd = getcwd()
filesR = [f for f in listdir(cwd) if f.endswith('.txt') and f.startswith('cobj')]
filesR.sort()

with codecs.open('Template.RES', encoding='cp1252') as f:
    datalines = f.readlines()

S0 = datalines[0]
S2 = datalines[-1]

fn_out = gen_out_filename()
id_mapping = {}
object_counters = {}

template_id = get_next_id()  # Generate a template ID
S0 = S0.replace("099999", template_id)

with open(fn_out, 'w', encoding='cp1252') as f_out:
    for fn in filesR:
        with codecs.open(fn, encoding='utf-8') as f:
            datalines = f.readlines()

        DATE_OBS = datalines[1].strip()
        T = Time(DATE_OBS, format='isot').datetime
        YY = T.year - 2000
        MO = T.month
        DD = T.day
        HH = T.hour
        MM = T.minute
        SS = T.second
        MS = int(T.microsecond / 10000)

        sDATE_OBS = f'{DD:02d}{MO:02d}{YY:02d} {HH:02d}{MM:02d}{SS:02d}{MS:02d}'

        if fn not in id_mapping:
            file_id = get_next_id()
            id_mapping[fn] = file_id
            object_counters[file_id] = 0
        else:
            file_id = id_mapping[fn]

        for i, REC in enumerate(datalines[2:]):
            DATA = REC.strip()
            RA, DEC = DATA.split(' ')[:2]

            sRA = f'{int(RA.split(":")[0]):02d}{int(RA.split(":")[1]):02d}'
            sRA += f'{int(RA.split(":")[2].split(".")[0]):02d}'
            sRA += f'{int(RA.split(":")[2].split(".")[1]):02d}'

            dec_sign = '+' if DEC.startswith('+') else '-'
            DEC = DEC.lstrip('+-')

            sDEC = f'{dec_sign}{int(DEC.split(":")[0]):02d}{int(DEC.split(":")[1]):02d}'
            sDEC += f'{int(DEC.split(":")[2].split(".")[0]):02d}'
            sDEC += f'{int(DEC.split(":")[2].split(".")[1]):02d}'

            object_counters[file_id] += 1
            suffix = f"{object_counters[file_id]:02d}"

            obj_id_with_suffix = f"{file_id}{suffix}"
            obj_S0 = S0.replace(template_id, obj_id_with_suffix)
            STR = f'{sDATE_OBS} {sRA} {sDEC} 000100\n'  # Убрали лишний ноль
            obj_S2 = S2

            f_out.write(obj_S0)
            f_out.write(STR)
            f_out.write(obj_S2)