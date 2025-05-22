#!/bin/sh
MONITORDIR="LOAD_FILE/"
inotifywait -m -e create --format '%w%f' "${MONITORDIR}" | while read NEWFILE
do
        echo "${NEWFILE} uploaded" 
	echo '========================================================'
	python WFT_15052023.py
	
	echo '========================================================'
        mv ${NEWFILE} PROCESS_FILE/
        python Search4Trails_22May2023_Assy.py
        
	echo '===========  preparation for Apex  ====================='
	python GSS_Apex_ASTRiDE.py
	apex_geo.py -c:apex.conf_ATO_cluster object_area=0.8 deblend=0 trail_len_tol=0 auto_postfilter=0 PROCESS_FILE/c*new 
	mv *RES PROCESS_FILE/
	mv *xml PROCESS_FILE/
	apex_geo_postprocess.py PROCESS_FILE/*.RES -c:apex.conf_ATO_cluster
	
	#echo '================  clean up PROCESS directory for the next file ============================'
        mv PROCESS_FILE/* PROCESSED

	echo '==================  DONE  =============================='
done

#inotifywait -m LOAD_FILE/ -e modify | 
#    while read dir action file; do
#        echo "The file '$file' appeared in directory '$dir' via '$action'"
#        # do something with the file
#        if [ -f "$dir$file" ]; then
#	   python WFT_26102022.py
#           rm $dir$file
#	fi   
#    done
