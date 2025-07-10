import sys
import time
import subprocess
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from filelock import FileLock, Timeout
from src.utils import path
from src.utils.ison_report import generate_report_for_file

# Directories and paths
SCRIPTS_DIR = Path(__file__).resolve().parent
LOAD_DIR = path.LOAD_FILE_DIR
TMP_DIR = path.TMP_DIR
PROCESSED_FITS_DIR = path.PROCESSED_FITS_DIR
RESULTS_DIR = path.PROCESSED_RESULTS_DIR
LOCK_FILE = path.LOCK_FILE
LOG_DIR = path.MAIN_DIR / "log"
LOG_FILE = LOG_DIR / "monitor.log"
PROCESSING_LOG = path.PROCESSING_LOG_FILE


def run_and_log(command: list, description: str):
    """
    Run a subprocess command and log its output.

    Args:
        command (list): Command and arguments to run.
        description (str): Short description for logging.
    """
    logging.info(f"=== Running {description} ===")
    logging.info(f"Command: {' '.join(map(str, command))}")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    if result.stdout:
        logging.info(f"{description} stdout:\n{result.stdout.strip()}")
    if result.stderr:
        logging.warning(f"{description} stderr:\n{result.stderr.strip()}")


def process_file(file_path: Path):
    """
    Process a single FITS file through the pipeline and generate an ISON report.

    Steps:
    1. Record file path in processing log
    2. Run config_setting, region, astrometry, and radec scripts
    3. Validate XY file
    4. Locate output result files in RESULTS_DIR
    5. Generate ISON report for each valid result
    6. Clean up: remove original FITS and clear log
    """
    logging.info(f"Start processing: {file_path.name}")
    try:
        # Mark file as processing
        PROCESSING_LOG.write_text(str(file_path))

        # Run external processing scripts
        run_and_log([sys.executable, SCRIPTS_DIR / "config_setting.py", str(file_path)], "config_setting.py")
        run_and_log([sys.executable, SCRIPTS_DIR / "region.py", str(file_path)], "region.py")
        run_and_log([sys.executable, SCRIPTS_DIR / "astrometry.py", str(file_path)], "astrometry.py")
        run_and_log([sys.executable, SCRIPTS_DIR / "radec_without_mode.py", str(file_path)], "radec_without_mode.py")

        # Verify XY FITS
        if not path.XY_FITS_FILE.exists() or path.XY_FITS_FILE.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty XY FITS: {path.XY_FITS_FILE}")

        # Find result files matching the processed stem
        stem = file_path.stem
        matches = list(RESULTS_DIR.glob(f"{stem}*"))
        if not matches:
            logging.error(f"No result files for '{stem}' in {RESULTS_DIR}")
        for result_file in matches:
            if result_file.exists() and result_file.stat().st_size > 0:
                try:
                    generate_report_for_file(result_file)
                    logging.info(f"ISON report generated for {result_file.name}")
                except Exception:
                    logging.exception(f"Failed ISON report for {result_file.name}")
            else:
                logging.error(f"Invalid result file: {result_file}")

        # Remove original FITS
        file_path.unlink()
        logging.info(f"Removed processed file: {file_path.name}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error in {e.cmd}: return code {e.returncode}")
        if e.stdout:
            logging.error(f"stdout: {e.stdout.strip()}")
        if e.stderr:
            logging.error(f"stderr: {e.stderr.strip()}")
    except Exception as e:
        logging.exception(f"Error processing {file_path.name}: {e}")
    finally:
        # Clear processing log if it still points to this file
        if PROCESSING_LOG.exists():
            PROCESSING_LOG.write_text("")
        logging.info(f"Processing log cleared.")


class NewFileHandler(FileSystemEventHandler):
    """
    Watchdog event handler that processes newly created files.
    """
    def __init__(self, processing_fn, lock_path: Path):
        super().__init__()
        self.processing_fn = processing_fn
        self.lock = FileLock(str(lock_path), timeout=1)

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        logging.info(f"Detected new file: {file_path.name}")
        try:
            with self.lock:
                logging.info(f"Acquired lock for {file_path.name}")
                self.processing_fn(file_path)
                logging.info(f"Released lock for {file_path.name}")
        except Timeout:
            logging.warning(f"Lock timeout for {file_path.name}, skipping.")
        except Exception as e:
            logging.error(f"Error handling file {file_path.name}: {e}")


def main():
    # Prepare logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Monitor started.")

    # Ensure directories exist
    for d in [LOAD_DIR, TMP_DIR, PROCESSED_FITS_DIR, RESULTS_DIR, path.CONFIGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logging.info("All required directories are ready.")

    # Set up file watcher
    handler = NewFileHandler(process_file, LOCK_FILE)
    observer = Observer()
    observer.schedule(handler, str(LOAD_DIR), recursive=False)
    observer.start()
    logging.info(f"Watching directory: {LOAD_DIR}")

    # Main loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Monitor stopped by user.")
    except Exception:
        logging.exception("Unhandled exception in monitor loop.")
    finally:
        observer.join()
        logging.info("Monitor shutdown complete.")


if __name__ == "__main__":
    main()