#!/bin/bash

# --- Настройки путей ---
# Эти переменные должны соответствовать путям, определенным в scripts/path.py
# Убедитесь, что они правильно настроены для вашей структуры директорий
MONITOR_DIR="input/"                # Соответствует path.LOAD_FILE_DIR (относительно корня проекта)
LOCK_FILE="/tmp/process.lock"         # Соответствует path.LOCK_FILE (абсолютный путь)
TMP_DIR="main/src/tmp"                         # Соответствует path.TMP_DIR (относительно корня проекта)
CONFIGS_DIR="main/conf"                 # Соответствует path.CONFIGS_DIR (относительно корня проекта)
SEX_CONFIG="${CONFIGS_DIR}/default.sex" # Соответствует path.SEX_CONFIG_FILE
PROCESSING_LOG="${TMP_DIR}/processing_log.txt" # Соответствует path.PROCESSING_LOG_FILE
XY_FITS_FILE="${TMP_DIR}/XY.fits"         # Соответствует path.XY_FITS_FILE

# Лог файл (используется для чего-то другого, оставлен как был в оригинале)
# LOGFILE="${TMP_DIR}/accuracy_log.txt"


# --- Функции ---

# Очистка TMP перед началом работы
clear_tmp_dir() {
    echo "cleaning ${TMP_DIR}/*..."
    # Закомментировано в оригинале, но если нужно чистить:
    # find ${TMP_DIR} -mindepth 1 -delete
    echo "Done."
}

# Обработка файла
process_file() {
    NEWFILE="$1" # Полный путь к новому файлу, который появился в MONITOR_DIR
    echo "--- Processing ${NEWFILE} ---"

    # Записываем имя файла в лог для отслеживания (используется в radec скриптах)
    # Предварительно очистим лог
    : > "${PROCESSING_LOG}"
    echo "${NEWFILE}" >> "${PROCESSING_LOG}"


    echo '===========  Sources extractions (config_setting.py) ============'
    # Теперь вызываем скрипт config_setting.py, передавая ему путь к новому файлу
    # предполагаем, что python скрипты находятся в той же директории, что и этот bash скрипт
    python main/config_setting.py "${NEWFILE}"


    echo '===========  Region File Generation (WriteOutRegionFile.py) ====='
    # Вызываем скрипт WriteOutRegionFile.py
    # Убрали префикс SRC/, предполагая, что скрипты теперь лежат рядом
    python main/region.py


    echo '===========  Astrometry (WFT_19072024.py or similar) ============='
    # Вызываем скрипт для астрометрии
    # Убрали префикс SRC/, предполагая, что скрипты теперь лежат рядом
    # Примечание: в оригинале был вызов WFTforSatsimDATA.py, здесь используем WFT_19072024.py
    # как был предоставлен, но нужно убедиться, какой из них нужен
    python main/astrometry.py "${NEWFILE}"


    echo '===========  RADEC Conversion and Analysis (radec_without_mode.py) ='
    # Вызываем скрипт radec_without_mode.py
    # Убрали префикс SRC/, предполагая, что скрипты теперь лежат рядом
    python main/radec_without_mode.py


    # Проверка существования файла с XY координатами после обработки
    if [ ! -s "${XY_FITS_FILE}" ]; then
        echo "Error: ${XY_FITS_FILE} does not exist or is empty after processing."
        # В оригинале был exit 1, но это остановит монитор.
        # Возможно, лучше просто пропустить этот файл и перейти к следующему.
        # Для сохранения оригинального поведения оставим exit 1, но будьте внимательны.
        exit 1
    fi

    # Удаление исходного файла после успешной обработки
    echo "Removing processed file: ${NEWFILE}"
    rm -f "${NEWFILE}"

    # Очистка лога обработки после успешного цикла для этого файла
    # (оригинальный скрипт делал это дважды, оставим одно очищение в конце)
    : > "${PROCESSING_LOG}"

    echo '==================  DONE  =============================='
    echo "" # Пустая строка для разделения вывода
}

# --- Основная логика ---

# Очистка TMP перед запуском мониторинга
clear_tmp_dir

echo "--- Starting monitor on ${MONITOR_DIR} ---"
echo "Using temporary directory ${TMP_DIR}"
echo "Using lock file ${LOCK_FILE}"

# Используем inotifywait для мониторинга создания файлов в MONITOR_DIR
# Запускаем цикл while для каждого нового файла
inotifywait -m -e create --format '%w%f' "${MONITOR_DIR}" | while read NEWFILE
do
    echo "Detected new file: ${NEWFILE}"
    # Используем flock для создания блокировки, чтобы одновременно обрабатывался только один файл
    {
        exec 200>"${LOCK_FILE}"
        flock -x 200
        echo "Lock acquired for ${NEWFILE}"

        # Вызываем функцию обработки файла
        process_file "${NEWFILE}"

        # Очистка TMP после обработки каждого файла (если нужно)
        # clear_tmp_dir # Закомментировано, т.к. оригинал чистил в начале и конце process_file

        echo "Lock released for ${NEWFILE}"
    } # Завершаем блок с блокировкой
done

# Если монитор по какой-то причине остановится (например, из-за exit 1 в process_file)
echo "Monitor script stopped."