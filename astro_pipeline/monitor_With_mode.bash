#!/bin/bash
MONITORDIR="LOAD_FILE/"
LOCKFILE="/tmp/process.lock"
TMPDIR="TMP"
LOGFILE="${TMPDIR}/accuracy_log.txt"

process_file() {
    NEWFILE="$1"
    echo "${NEWFILE} uploaded"

    echo "${NEWFILE}" >> ${TMPDIR}/processing_log.txt

    sleep 1
    echo '===========  Sources extractions ======================='
    python confic_setting.py "${NEWFILE}"
    # python config_setting_without_focallen.py "${NEWFILE}"
    

    sleep 1
    echo '===========  Sources extractions ======================='
    # sex "${NEWFILE}" -c CONFIGS/default.sex
    # sex "${NEWFILE}" -c CONFIGS/defaultTEST.sex

    echo '========================================================'
    python SRC/WriteOutRegionFile_18072024.py

    echo '===========  astrometry up TMP directory  =============='
    python SRC/WFT_19072024.py "${NEWFILE}"

    echo '===========  astrometry up TMP directory  =============='
    python SRC/xy2radec.py "${NEWFILE}"

    echo '===========  astrometry up TMP directory  =============='
    python radec_StarObservation.py

    if [ ! -s TMP/XY.fits ]; then
        echo "Error: TMP/XY.fits does not exist or is empty"
        exit 1
    fi

    # rm -rf ${TMPDIR}/*

    echo "${NEWFILE}" > ${TMPDIR}/processing_log.txt

    rm -f "${NEWFILE}"
    : > ${TMPDIR}/processing_log.txt


    # echo "Corect: ${NEWFILE}:"
    # read correct
    # echo "Incorrect: ${NEWFILE}:"
    # read incorrect

    # echo "Правильный объект был обнаружен в ${NEWFILE} (y/n)?"
    # read found_correct_object
    # if [ "$found_correct_object" == "y" ]; then
    #     correct_files=1
    # else
    #     correct_files=0
    # fi

    # echo "${NEWFILE},Correct:${correct},Incorrect:${incorrect},CorrectFile:${correct_files}" >> ${LOGFILE}

    echo '==================  APEX  =============================='
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate apex_env
    # echo "current conda env: $(conda info --envs | grep \* | awk '{print $1}')"
    # python GSS_Apex_TURBO.py
    # conda deactivate


    echo '==================  DONE  =============================='
}

# # Запуск обработки файлов последовательно
while true; do
    for NEWFILE in "${MONITORDIR}"*; do
        if [ -f "$NEWFILE" ]; then
            {
                # Ожидаем, пока не получим блокировку
                exec 200>$LOCKFILE
                flock -x 200
                process_file "${NEWFILE}"
            }
        fi
    done

    # Добавляем задержку перед повторной проверкой
    sleep 1
done

# Запускаем обработку файлов с блокировкой
# inotifywait -m -e create --format '%w%f' "${MONITORDIR}" | while read NEWFILE
# do
#     {
#         # Ожидаем, пока не получим блокировку
#         exec 200>$LOCKFILE
#         flock -x 200
#         process_file "${NEWFILE}"
#     }
# done