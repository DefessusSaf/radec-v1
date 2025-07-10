# ison_report.py
import xml.etree.ElementTree as ET
from pathlib import Path
from filelock import FileLock, Timeout
from src.utils import path

LOCK_TIMEOUT = 5  # seconds
LOCK_FILE = path.ISON / "ison_report.lock"


def ra_to_dec(ra: str) -> float:
    """
    Convert RA in H:M:S format to decimal degrees.
    """
    h, m, s = map(float, ra.split(':'))
    return h + m / 60 + s / 3600


def dec_to_dec(dec: str) -> float:
    """
    Convert Dec in ±D:M:S format to decimal degrees.
    """
    sign = -1 if dec.startswith('-') else 1
    d, m, s = map(float, dec.lstrip('+-').split(':'))
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


def generate_report_for_file(result_file: Path, sensor_id: int = 10121) -> None:
    """
    Generate an ISON report for a single result file in streaming mode.
    The report filename is based on the UTC timestamp from the file's data.
    """
    # Read raw data
    with open(result_file, 'r', encoding='utf-8') as f:
        raw_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    # Parse data: yields list with each object entry
    entries = parse_data_with_filename(raw_lines, sensor_id=sensor_id)
    if not entries:
        return

    # Determine report name from UTC, sanitize for filename
    utc_str = entries[0]['utc']
    safe_utc = utc_str.replace(':', '').replace(' ', '_')
    out_file = path.ISON / f'ison_report_{safe_utc}.xml'

    # Ensure directory and lock
    path.ISON.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(LOCK_FILE), timeout=LOCK_TIMEOUT)
    try:
        with lock:
            create_ison_report(entries, out_file)
    except Timeout:
        # lock acquisition failed — skip or log
        pass


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python ison_report.py <result_file> [sensor_id]')
        sys.exit(1)
    sensor = int(sys.argv[2]) if len(sys.argv) > 2 else 10121
    generate_report_for_file(Path(sys.argv[1]), sensor)