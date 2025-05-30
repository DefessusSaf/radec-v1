import os
import time
import subprocess  # для выполнения команды
import astropy.io.fits as pyfits
from astropy import coordinates
from astropy import units as u
from astropy.wcs import WCS
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

processed_dir = "PROCESSED/"
process_file_dir = "PROCESS_FILE/"
apex_dir = "APEX/"
objid_file = "last_objid.txt"  # Файл для хранения последнего OBJID

if not os.path.exists(apex_dir):
    os.makedirs(apex_dir)

def wait_for_file_to_stabilize(file_path, wait_time=1):
    """Ожидает, пока файл не перестанет изменяться."""
    initial_size = os.path.getsize(file_path)
    time.sleep(wait_time)  
    final_size = os.path.getsize(file_path)

    while initial_size != final_size:
        initial_size = final_size
        time.sleep(wait_time)
        final_size = os.path.getsize(file_path)

def load_last_objid():
    """Загружает последний использованный OBJID из файла."""
    if os.path.exists(objid_file):
        with open(objid_file, 'r') as f:
            return int(f.read().strip())
    else:
        return 1  # Начинаем с 1, если файла нет

def save_last_objid(objid):
    """Сохраняет последний использованный OBJID в файл."""
    with open(objid_file, 'w') as f:
        f.write(str(objid))

class FitsFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.fits'):
            return

        print(f'Found new file: {event.src_path}')
        
        # Ожидание стабилизации файла
        wait_for_file_to_stabilize(event.src_path)
        self.process_fits_file(event.src_path)

    def process_fits_file(self, fits_path):
        base_name = os.path.basename(fits_path).split('.')[0].replace("TMP_", "")  # Исключаем "TMP_"
        
        txt_file = os.path.join(process_file_dir, f"{base_name}.fits.txt")
        
        if os.path.exists(txt_file):
            ra_dec_list = []
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        ra, dec = line.split()
                        ra_dec_list.append((ra, dec))
                    except ValueError:
                        print(f"Ошибка в строке: {line}. Ожидается две колонки (RA и DEC).")
                        continue
            
            fits = pyfits.open(fits_path, mode='update')
            header = fits[0].header
            wcs = WCS(header)
            
            # Загружаем последний OBJID и увеличиваем его
            OBJID = f"{load_last_objid():06d}"  # Форматируем как 6-значный идентификатор
            
            save_last_objid(int(OBJID) + 1)  # Увеличиваем на 1 для следующего файла

            NPIX1 = header['NAXIS1']
            NPIX2 = header['NAXIS2']

            RA = header['OBJCTRA']
            DEC = header['OBJCTDEC']
            
            c = coordinates.SkyCoord(RA, DEC, unit=('hourangle', 'deg'), frame='icrs')
            header['CRVAL1'] = (c.ra.deg, 'Location of the target in deg in RA')
            header['CRVAL2'] = (c.dec.deg, 'Location of the target in deg in DEC')

            xc = NPIX1 / 2.0 + 0.5
            yc = NPIX2 / 2.0 + 0.5
            header['CRPIX1'] = xc
            header['CRPIX2'] = yc
            
            for i, (ra, dec) in enumerate(ra_dec_list):
                c = coordinates.SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
                x, y = wcs.world_to_pixel(c)
                OBJX = x 
                OBJY = y  

                if i == 0:
                    header['OBJECT'] = (OBJID, "ID")
                    header['OBJX'] = (float(OBJX), 'X coord of the first object in pixels')
                    header['OBJY'] = (float(OBJY), 'Y coord of the first object in pixels')
                else:
                    obj_num = f'{i:02d}' 
                     
                    header[f'OBJ{obj_num}X'] = (float(OBJX), f'X coord of object {i+1} in pixels')
                    header[f'OBJ{obj_num}Y'] = (float(OBJY), f'Y coord of object {i+1} in pixels')
                    header[f'OBJECT{obj_num}'] = (f"{i + 1}", "ID")

            fits.flush()
            
            try:
                data = fits[0].data.copy()
            except Exception as e:
                print(f'Error in {fits_path}: {e}')
                fits.close()
                return
            
            fits.close()  

            # Копируем файл в директорию APEX
            new_fits_path = os.path.join(apex_dir, os.path.basename(fits_path))
            pyfits.writeto(new_fits_path, data, header, overwrite=True)

            print(f'Processed: {new_fits_path}')
            
            # Выполнение команды1 с использованием нового файла
            command = [
                "apex_geo.py", 
                "/c:/home/hellnim/apex.conf.Assy-new-trails", 
                "trail_len_tol=0", 
                "disable_calib=1", 
                "deblend=0", 
                "object_area=0.8", 
                "auto_threshold=0", 
                "overwrite_frame=1", 
                "threshold=3.0", 
                "auto_postfilter=0", 
                "default_seeing=8.0", 
                new_fits_path  # Используем новый файл вместо TMP_Meteor1_19R_00077.fits
            ]
            subprocess.run(command)  # Запуск команды1
            print(f'Executed command1 for: {new_fits_path}')
        else:
            print(f'Not found .txt file {base_name}')

def clear_processed_directory():
    """Очищает директорию PROCESSED."""
    for filename in os.listdir(processed_dir):
        file_path = os.path.join(processed_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f'Removed file: {file_path}')
        except Exception as e:
            print(f'Error removing file {file_path}: {e}')

if __name__ == "__main__":
    # Обработка всех существующих файлов в директории PROCESSED
    for filename in os.listdir(processed_dir):
        file_path = os.path.join(processed_dir, filename)
        if os.path.isfile(file_path) and filename.endswith('.fits'):
            wait_for_file_to_stabilize(file_path)
            FitsFileHandler().process_fits_file(file_path)

    # Очищаем директорию PROCESSED
    clear_processed_directory()
    
    if not any(filename.endswith('.fits') for filename in os.listdir(processed_dir)):
        print("No more files ti process. Done")
        exit(0)
    
    # Запуск наблюдателя для новых файлов
    event_handler = FitsFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=processed_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)  # Ожидание
    except KeyboardInterrupt:
        observer.stop()
    observer.join()