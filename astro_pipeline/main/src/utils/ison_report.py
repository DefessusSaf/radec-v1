import xml.etree.ElementTree as ET
import os
from pathlib import Path
from datetime import datetime
from filelock import FileLock, Timeout
from src.utils import path

LOCK_TIMEOUT = 5  # seconds
LOCK_FILE = path.ISON / "ison_report.lock"


def ra_to_dec(ra: str) -> float:
    """
    Convert RA in H:M:S format to decimal degrees.
    """
    h, m, s = map(float, ra.split(":"))
    return h + m / 60 + s / 3600


def dec_to_dec(dec: str) -> float:
    """
    Convert Dec in ±D:M:S format to decimal degrees.
    """
    sign = -1 if dec.startswith("-") else 1
    d, m, s = map(float, dec.lstrip("+-").split(":"))
    return sign * (d + m / 60 + s / 3600)


def parse_data_with_filename(raw_data: list[str], sensor_id: int = 10121) -> list[dict]:
    """
    Parse raw data lines into list of measurement dictionaries.
    """
    file_name = raw_data[0].split(": ")[1].strip()
    utc = raw_data[1].strip()
    parsed = []
    for idx, line in enumerate(raw_data[2:], start=1):
        parts = line.split()
        ra, dec = parts[0], parts[1]
        x, y, x_err, y_err, a, b, x_min, y_min, x_max, y_max = map(float, parts[2:12])
        entry = {
            "file": file_name,
            "sensor": sensor_id,
            "id": idx,
            "utc": utc,
            "ra_j2000": str(ra_to_dec(ra)),
            "dec_j2000": str(dec_to_dec(dec)),
            "x": x,
            "y": y,
            "x_error": x_err,
            "y_error": y_err,
            "a": a,
            "b": b,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "mag": None,
            "suspicious": False,
        }
        parsed.append(entry)
    return parsed


def process_results_dir(dir_path: Path) -> list[dict]:
    """
    Process all result files in directory and return list of entries.
    """
    all_entries = []
    for file in dir_path.iterdir():
        if file.is_file():
            with open(file, 'r') as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            all_entries.extend(parse_data_with_filename(lines))
    return all_entries


def group_files_by_prefix(dir_path: Path) -> dict[str, list[Path]]:
    """
    Group files in directory by prefix before first underscore.
    """
    groups: dict[str, list[Path]] = {}
    for file in dir_path.iterdir():
        if file.is_file():
            prefix = file.stem.split('_')[0]
            groups.setdefault(prefix, []).append(file)
    return groups


def create_ison_report(data: list[dict], output_file: Path) -> None:
    """
    Create XML report from data list and save to output_file.
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


def generate_reports(mode: str = 'date') -> None:
    """
    Generate ISON reports in the specified mode: 'date' or 'prefix'.
    Reports are saved in path.ISON directory.
    """
    path.ISON.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(LOCK_FILE), timeout=LOCK_TIMEOUT)
    try:
        with lock:
            if mode == 'date':
                date_str = datetime.utcnow().strftime('%Y%m%d')
                out_file = path.ISON / f'ison_report_{date_str}.xml'
                data = process_results_dir(path.PROCESSED_RESULTS_DIR)
                create_ison_report(data, out_file)
                print(f'Ison report created: {out_file}')

            # in developing 
            elif mode == 'prefix':
                groups = group_files_by_prefix(path.PROCESSED_RESULTS_DIR)
                for prefix, files in groups.items():
                    # Temporarily change processing_dir
                    temp_dir = Path(path.TMP_DIR) / f'tmp_{prefix}'
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    for f in files:
                        (temp_dir / f.name).write_bytes(f.read_bytes())
                    data = process_results_dir(temp_dir)
                    out_file = path.ISON / f'ison_report_{prefix}.xml'
                    create_ison_report(data, out_file)
                    print(f'Ison report created: {out_file}')
                    # cleanup temp_dir
                    for f in temp_dir.iterdir():
                        f.unlink()
                    temp_dir.rmdir()
            else:
                raise ValueError("Unknown mode, use 'date' or 'prefix'.")
    except Timeout:
        print('Could not acquire lock for report generation.')


if __name__ == '__main__':
    # By default generate daily report
    generate_reports(mode=os.environ.get('ISON_REPORT_MODE', 'date'))