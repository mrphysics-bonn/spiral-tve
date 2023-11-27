#!/bin/bash

usage="$(basename "$0") [-h] [-t T1_FILE] [-m MASK_FILE] [-d] DIFF_FILE SLSPEC -- Preprocessing of spiral diffusion data

where:
    DIFF_FILE    Spiral diffusion images
    SLSPEC       slspec text file specifying the acquisition order for eddy
    -h           show this help text
    -t T1_FILE   register diffusion data to T1/MPRAGE data provided
    -m MASK_FILE Use provided brain mask for diffusion data
    -d           do denoising on magnitude instead of complex data (not recommended)"

# option
mag_flag=false
while getopts "ht:m:d" opt; do
    case $opt in
        h) echo "$usage"
        exit
        ;;
        t) T1_FILE=${OPTARG} ;;
        m) MASK_FILE=${OPTARG} ;;
        d) mag_flag=true ;;
        :) printf "missing argument for -%s\n" "$OPTARG" >&2
        echo "$usage" >&2
        exit 1
        ;;
        \?) printf "illegal option: -%s\n" "$OPTARG" >&2
        echo "$usage" >&2
        exit 1
        ;;
    esac
done
shift "$((OPTIND-1))"

# input
if [ "$#" -lt 1 ]; then
    echo "Input file missing"
    exit 1
elif [ "$#" -lt 2 ]; then
    echo "slspec file missing"
    exit 1
else
    IN_FILE="$1"
    slspec="$2"
fi

IN_FILE_PREFIX=${IN_FILE%%.*}
IN_FILE_PATH=$(dirname $IN_FILE)
SCRIPTPATH="/home/niesens/reco/preprocessing"

if $mag_flag; then
    echo "Use magnitude data for denoising."

    # Convert to magnitude
    python "${SCRIPTPATH}/nifti2mag.py" ${IN_FILE} "${IN_FILE_PREFIX}_mag.nii.gz"
    # Denoising
    dwidenoise -force "${IN_FILE_PREFIX}_mag.nii.gz" "${IN_FILE_PREFIX}_denoised_mag.nii.gz"
else
    # Denoising
    echo "Use complex data for denoising."
    dwidenoise -force ${IN_FILE} "${IN_FILE_PREFIX}_denoised.nii.gz"
    # Convert to magnitude
    python "${SCRIPTPATH}/nifti2mag.py" "${IN_FILE_PREFIX}_denoised.nii.gz" "${IN_FILE_PREFIX}_denoised_mag.nii.gz"
fi

# convert nii to mif
mrconvert -force "${IN_FILE_PREFIX}_denoised_mag.nii.gz" -fslgrad "${IN_FILE_PREFIX}.bvec" "${IN_FILE_PREFIX}.bval" "${IN_FILE_PREFIX}_denoised_mag.mif"

# Gibbs-Ringing removal
mrdegibbs -force "${IN_FILE_PREFIX}_denoised_mag.mif" "${IN_FILE_PREFIX}_denoised_mag_gr.mif"

# Calculate mp_order parameter for eddy - use n_excitations / 3
mporder=`expr  $(wc -l < $slspec) / 3`

# Mask for eddy
if test -f "$MASK_FILE" ; then
    echo "Use mask file for eddy mask."
    eddy_mask="$MASK_FILE"
    if [[ $eddy_mask == *.gz ]]; then
        gunzip -f -k $eddy_mask # unzip as otherwise the mask gets corrupted by mrconvert in dwifslpreproc
        eddy_mask=${eddy_mask%.*}
    fi
else
    echo "Generating mask for eddy from mean b0 image"
    dwiextract -force "${IN_FILE_PREFIX}_denoised_mag_gr.mif" - -bzero | mrmath -force - mean "${IN_FILE_PREFIX}_meanb0_forEddyMask.mif" -axis 3
    mrconvert -force "${IN_FILE_PREFIX}_meanb0_forEddyMask.mif" "${IN_FILE_PREFIX}_meanb0_forEddyMask.nii.gz"
    bet "${IN_FILE_PREFIX}_meanb0_forEddyMask.nii.gz" "${IN_FILE_PREFIX}_meanb0_forEddyMask_bet.nii.gz" -f 0.4 -m
    gunzip -f "${IN_FILE_PREFIX}_meanb0_forEddyMask_bet_mask.nii.gz" # unzip as otherwise the mask gets corrupted by mrconvert in dwifslpreproc
    eddy_mask="${IN_FILE_PREFIX}_meanb0_forEddyMask_bet_mask.nii"
fi

# Motion correction with eddy
# first level model (flm) is set to movement to avoid eddy current correction
# mporder is recommended to be somewhere between N/4 and N/2, where N is the number of excitations
dwifslpreproc -force "${IN_FILE_PREFIX}_denoised_mag_gr.mif" "${IN_FILE_PREFIX}_moco.mif" -rpe_none -pe_dir ap -eddy_mask $eddy_mask -eddy_slspec $slspec -eddyqc_all "$IN_FILE_PATH/eddy_params" -eddy_options " --flm=movement --repol --data_is_shelled --mporder=$mporder --ol_type=both "

# Convert mrtrix output to nii and bvec/bval
mrconvert -force "${IN_FILE_PREFIX}_moco.mif" -export_grad_fsl "${IN_FILE_PREFIX}_moco_unwarped_bet.bvec" "${IN_FILE_PREFIX}_moco_unwarped_bet.bval" "${IN_FILE_PREFIX}_moco.nii.gz"

# Gradient nonlinearity correction
${SCRIPTPATH}/GradientDistortionUnwarp.sh --workingdir="$IN_FILE_PATH/unwarp_wd" --in="${IN_FILE_PREFIX}_moco" --out="${IN_FILE_PREFIX}_moco_unwarped" --coeffs="${SCRIPTPATH}/coeff_SC72CD.grad" --owarp="${IN_FILE_PREFIX}_owarp"

# Calculate mean b0
mrconvert -force "${IN_FILE_PREFIX}_moco_unwarped.nii.gz" -fslgrad "${IN_FILE_PREFIX}_moco_unwarped_bet.bvec" "${IN_FILE_PREFIX}_moco_unwarped_bet.bval" "${IN_FILE_PREFIX}_moco_unwarped.mif"
dwiextract -force "${IN_FILE_PREFIX}_moco_unwarped.mif" - -bzero | mrmath -force - mean "${IN_FILE_PREFIX}_meanb0.mif" -axis 3
mrconvert -force "${IN_FILE_PREFIX}_meanb0.mif" "${IN_FILE_PREFIX}_meanb0.nii.gz"

if test -f "$MASK_FILE" ; then
    echo "Use mask file for masking diffusion data."
    # Use mask file
    fslmaths "${IN_FILE_PREFIX}_moco_unwarped.nii.gz" -mul $MASK_FILE "${IN_FILE_PREFIX}_moco_unwarped_bet.nii.gz"
    fslmaths "${IN_FILE_PREFIX}_meanb0.nii.gz" -mul $MASK_FILE "${IN_FILE_PREFIX}_meanb0_bet.nii.gz"
else
    echo "No mask file given - mask with mean b0"
    # Use BET mask from mean b0
    bet "${IN_FILE_PREFIX}_meanb0.nii.gz" "${IN_FILE_PREFIX}_meanb0_bet.nii.gz" -m -f 0.4
    fslmaths "${IN_FILE_PREFIX}_moco_unwarped.nii.gz" -mul "${IN_FILE_PREFIX}_meanb0_bet_mask.nii.gz" "${IN_FILE_PREFIX}_moco_unwarped_bet.nii.gz"
fi

# Register to T1 (optional)
if test -f "$T1_FILE" ; then
    # Do preprocessing and masking on T1 data
    T1_FILE_PREFIX=${T1_FILE%%.*}
    python "${SCRIPTPATH}/t1_processing.py" -m $T1_FILE "${T1_FILE_PREFIX}_bet" #-mni "/home/niesens/fsl/data/standard/MNI152_T1_1mm.nii.gz" -mni_out "${IN_FILE_PREFIX}_MNI152_T1_1mm_T1" -roi "/scratch/niesens/20230914/ReconstructedImages/rois/harvardoxford-subcortical_prob_LeftLateralVentricle.nii.gz,/scratch/niesens/20230914/ReconstructedImages/rois/jhu-labels_label_AnteriorcoronaradiataR.nii.gz,/scratch/niesens/20230914/ReconstructedImages/rois/jhu-labels_label_AnteriorcoronaradiataL.nii.gz"

    # Register mean b0 to MPRAGE using the epi_reg script
    epi_reg --epi="${IN_FILE_PREFIX}_meanb0.nii.gz" --t1=$T1_FILE --t1brain="${T1_FILE_PREFIX}_bet.nii.gz" --out="${IN_FILE_PREFIX}_meanb0_reg.nii.gz"

    # Apply registration
    flirt -in "${IN_FILE_PREFIX}_moco_unwarped.nii.gz" -ref "${T1_FILE_PREFIX}_bet.nii.gz" -out "${IN_FILE_PREFIX}_moco_unwarped_reg.nii.gz" -applyxfm -init "${IN_FILE_PREFIX}_meanb0_reg.mat"

    # Apply T1 brain mask (usually works better than mean-b0 mask for sTx 7T images)
    fslmaths "${IN_FILE_PREFIX}_moco_unwarped_reg.nii.gz" -mul "${T1_FILE_PREFIX}_bet_mask.nii.gz" "${IN_FILE_PREFIX}_moco_unwarped_reg_bet.nii.gz"
    fslmaths "${IN_FILE_PREFIX}_meanb0_reg.nii.gz" -mul "${T1_FILE_PREFIX}_bet_mask.nii.gz" "${IN_FILE_PREFIX}_meanb0_reg_bet.nii.gz"

    # Rotate b-vectors
    python "${SCRIPTPATH}/rotate_bvecs.py" -i "${IN_FILE_PREFIX}_moco_unwarped_bet.bvec" -t "${IN_FILE_PREFIX}_meanb0_reg.mat" -o "${IN_FILE_PREFIX}_moco_unwarped_reg.bvec"
    /bin/cp "${IN_FILE_PREFIX}_moco_unwarped_bet.bval" "${IN_FILE_PREFIX}_moco_unwarped_reg.bval"
    
    # Register T1 image to mean b0
    convert_xfm -omat "${IN_FILE_PREFIX}_T1toDiff.mat" -inverse "${IN_FILE_PREFIX}_meanb0_reg.mat"
    flirt -in "${T1_FILE_PREFIX}_bet_mask.nii.gz" -ref "${IN_FILE_PREFIX}_meanb0.nii.gz" -out "${T1_FILE_PREFIX}_bet_mask_diff.nii.gz" -applyxfm -init "${IN_FILE_PREFIX}_T1toDiff.mat"
    
    # FAST segmentation
    #fast -S 1 -t 1 -n 3 -o "${T1_FILE_PREFIX}_bet_segmented" -b -B "${T1_FILE_PREFIX}_bet"
    #fslmaths "${T1_FILE_PREFIX}_bet_segmented_pve_0.nii.gz" -thr 0.9 -bin "${T1_FILE_PREFIX}_bet_segmented_CSF.nii.gz"
    #fslmaths "${T1_FILE_PREFIX}_bet_segmented_pve_1.nii.gz" -thr 0.9 -bin "${T1_FILE_PREFIX}_bet_segmented_GM.nii.gz"
    #fslmaths "${T1_FILE_PREFIX}_bet_segmented_pve_2.nii.gz" -thr 0.9 -bin "${T1_FILE_PREFIX}_bet_segmented_WM.nii.gz"
    
    # Register T1 to MNI152 (T1-weighted)
    #flirt -in "${T1_FILE_PREFIX}_bet" -ref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -dof 12 -out "${T1_FILE_PREFIX}_bet_T1toMNIlin" -omat "${T1_FILE_PREFIX}_bet_T1toMNIlin.mat" # if you just do flirt use: MNI152_T1_1mm_brain.nii.gz
    #fnirt --in="${T1_FILE}" --aff="${T1_FILE_PREFIX}_bet_T1toMNIlin.mat" --config=T1_2_MNI152_2mm.cnf --iout="${T1_FILE_PREFIX}_bet_T1toMNInonlin" --cout="${T1_FILE_PREFIX}_bet_T1toMNI_coef" --lambda=400,200,150,75,60,45
fi
