#!/bin/bash

usage="$(basename "$0") Map_FILE T1_FILE reg_MATRIX Roi_FOLDER DTD_METRIC SEQUENCE APPROACH -- Registration and segmentation of dtd-parameter maps

where:
    Map_FILE    dtd-parameter map
    T1_FILE     masked T1 image
    reg_MATRIX  registration matrix from diffusion to T1 space
    Roi_FOLDER  folder which contains binary masks for interested ROIs
    DTD_METRIC  dtd-metric from parameter map (MD, FA, uFA, MKi, OP)
    SEQUENCE    used diffusion sequence
    APPROACH    used tensor valued encoding approach (DIVIDE, QTI)"

# option
mag_flag=false
while getopts "ht:m:d" opt; do
    case $opt in
        h) echo "$usage"
        exit
        ;;
    esac
done
shift "$((OPTIND-1))"

# input
if [ "$#" -lt 1 ]; then
    echo "Map-file missing"
    exit 1
elif [ "$#" -lt 2 ]; then
    echo "T1-file missing"
    exit 1
elif [ "$#" -lt 3 ]; then
    echo "registration-matrix missing"
    exit 1
elif [ "$#" -lt 4 ]; then
    echo "roi-folder missing"
    exit 1
elif [ "$#" -lt 5 ]; then
    echo "dtd-metric missing"
    exit 1
elif [ "$#" -lt 6 ]; then
    echo "sequence missing"
    exit 1
elif [ "$#" -lt 7 ]; then
    echo "approach missing"
    exit 1
else
    MAP_FILE="$1"
    T1_FILE="$2"
    REG_MATRIX="$3"
    ROI_FOLDER="$4"
    DTD_METRIC="$5"
    SEQUENCE="$6"
    APPROACH="$7"
fi

MAP_FILE_PREFIX=${MAP_FILE%%.*}
MAP_FILE_PATH=$(dirname $MAP_FILE)
T1_FILE_PREFIX=${T1_FILE%%.*}

# parameter maps to T1 space
flirt -in ${MAP_FILE} -ref ${T1_FILE} -out "${MAP_FILE_PREFIX}_reg.nii.gz" -applyxfm -init ${REG_MATRIX}

# apply ROI-segmentation on registered parameter maps
fslmaths "${MAP_FILE_PREFIX}_reg" -mul "${ROI_FOLDER}/jhu-labels_label_Spleniumofcorpuscallosum_T1_thresh" "${MAP_FILE_PREFIX}_reg_Spleniumofcorpuscallosum.nii.gz"
fslmaths "${MAP_FILE_PREFIX}_reg" -mul "${ROI_FOLDER}/harvardoxford-subcortical_prob_LateralVentricle_T1_thresh" "${MAP_FILE_PREFIX}_reg_LateralVentricle.nii.gz"
fslmaths "${MAP_FILE_PREFIX}_reg" -mul "${ROI_FOLDER}/jhu-labels_label_Anteriorcoronaradiata_T1_thresh" "${MAP_FILE_PREFIX}_reg_Anteriorcoronaradiata.nii.gz"

# save parameter map values into Pandas Dataframe for comparison
python "into_dataframe.py" -map_file "${MAP_FILE_PREFIX}_reg_Spleniumofcorpuscallosum.nii.gz" -region Splenium -dtd ${DTD_METRIC} -sequence ${SEQUENCE} -approach ${APPROACH}
python "into_dataframe.py" -map_file "${MAP_FILE_PREFIX}_reg_LateralVentricle.nii.gz" -region LateralVentricle -dtd ${DTD_METRIC} -sequence ${SEQUENCE} -approach ${APPROACH}
python "into_dataframe.py" -map_file "${MAP_FILE_PREFIX}_reg_Anteriorcoronaradiata.nii.gz" -region AnteriorCR -dtd ${DTD_METRIC} -sequence ${SEQUENCE} -approach ${APPROACH}
