#!/bin/bash

echo "DIVIDE"

echo "qMASmod"
echo "MD"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/output/dtd_gamma_MD.nii.gz /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/Reco_qMasLte2000_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MD qMASmod DIVIDE
echo "MKi"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/output/dtd_gamma_MKi.nii.gz /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/Reco_qMasLte2000_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MKi qMASmod DIVIDE
echo "FA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/output/Reco_qMasLte2000_Lte_fa.nii.gz /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/Reco_qMasLte2000_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois FA qMASmod DIVIDE
echo "uFA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/output/dtd_gamma_ufa.nii.gz /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/Reco_qMasLte2000_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois uFA qMASmod DIVIDE

echo "NowLteTuned"
echo "MD"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/output/dtd_gamma_MD.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/Reco_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MD NowLteTuned DIVIDE
echo "MKi"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/output/dtd_gamma_MKi.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/Reco_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MKi NowLteTuned DIVIDE
echo "FA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/output/Reco_NowLteTuned_Lte_fa.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/Reco_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois FA NowLteTuned DIVIDE
echo "uFA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/output/dtd_gamma_ufa.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteTuned/Reco_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois uFA NowLteTuned DIVIDE

echo "NowLteDetuned"
echo "MD"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/output/dtd_gamma_MD.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/Reco_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MD NowLteDetuned DIVIDE
echo "MKi"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/output/dtd_gamma_MKi.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/Reco_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MKi NowLteDetuned DIVIDE
echo "FA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/output/Reco_NowLteDetuned_Lte_fa.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/Reco_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois FA NowLteDetuned DIVIDE
echo "uFA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/output/dtd_gamma_ufa.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteDetuned/Reco_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois uFA NowLteDetuned DIVIDE

echo "NowLteComp"
echo "MD"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteComp/output/dtd_gamma_MD.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteComp/Reco_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MD NowLteComp DIVIDE
echo "MKi"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteComp/output/dtd_gamma_MKi.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteComp/Reco_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MKi NowLteComp DIVIDE
echo "FA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteComp/output/Reco_NowLteComp_Lte_fa.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteComp/Reco_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois FA NowLteComp DIVIDE
echo "uFA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/NowLteComp/output/dtd_gamma_ufa.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/NowLteComp/Reco_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois uFA NowLteComp DIVIDE

echo "QTI"

echo "qMASmod"
echo "MD"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_qMas/output/Reco_qti_qMas_md.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/Reco_qti_qMas_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MD qMASmod QTI
echo "OP"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_qMas/output/Reco_qti_qMas_c_c.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/Reco_qti_qMas_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois OP qMASmod QTI
echo "MKi"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_qMas/output/Reco_qti_qMas_c_md.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/Reco_qti_qMas_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MKi qMASmod QTI
echo "FA"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_qMas/output/Reco_qti_qMas_fa.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/Reco_qti_qMas_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois FA qMASmod QTI
echo "uFA"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_qMas/output/Reco_qti_qMas_ufa.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_qMas/Reco_qti_qMas_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois uFA qMASmod QTI

echo "NowLteTuned"
echo "MD"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/output/Reco_qti_NowLteTuned_md.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/Reco_qti_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MD NowLteTuned QTI
echo "OP"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/output/Reco_qti_NowLteTuned_c_c.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/Reco_qti_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois OP NowLteTuned QTI
echo "MKi"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/output/Reco_qti_NowLteTuned_c_md.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/Reco_qti_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MKi NowLteTuned QTI
echo "FA"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/output/Reco_qti_NowLteTuned_fa.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/Reco_qti_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois FA NowLteTuned QTI
echo "uFA"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/output/Reco_qti_NowLteTuned_ufa.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteTuned/Reco_qti_NowLteTuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois uFA NowLteTuned QTI

echo "NowLteDetuned"
echo "MD"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/output_Dipy/qti_NowLteDetuned_cls_real_wslice_wb0/Reco_qti_NowLteDetuned_md.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/Reco_qti_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MD NowLteDetuned QTI
echo "OP"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/output_Dipy/qti_NowLteDetuned_cls_real_wslice_wb0/Reco_qti_NowLteDetuned_c_c.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/Reco_qti_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois OP NowLteDetuned QTI
echo "MKi"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/output_Dipy/qti_NowLteDetuned_cls_real_wslice_wb0/Reco_qti_NowLteDetuned_c_md.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/Reco_qti_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MKi NowLteDetuned QTI
echo "FA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/output_Dipy/qti_NowLteDetuned_cls_real_wslice_wb0/Reco_qti_NowLteDetuned_fa.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/Reco_qti_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois FA NowLteDetuned QTI
echo "uFA"
./map_segmentation_T1.sh /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/output_Dipy/qti_NowLteDetuned_cls_real_wslice_wb0/Reco_qti_NowLteDetuned_ufa.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/Reco_qti_NowLteDetuned_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois uFA NowLteDetuned QTI

echo "NowLteComp"
echo "MD"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/output/Reco_qti_NowLteComp_md.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/Reco_qti_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MD NowLteComp QTI
echo "OP"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/output/Reco_qti_NowLteComp_c_c.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/Reco_qti_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois OP NowLteComp QTI
echo "MKi"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/output/Reco_qti_NowLteComp_c_md.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/Reco_qti_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois MKi NowLteComp QTI
echo "FA"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/output/Reco_qti_NowLteComp_fa.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/Reco_qti_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois FA NowLteComp QTI
echo "uFA"
./map_segmentation_T1.sh /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/output/Reco_qti_NowLteComp_ufa.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/008-dzne-bn_MPRAGE_0_bet.nii.gz /scratch/niesens/20231020/ReconstructedImages/qti_NowLteComp/Reco_qti_NowLteComp_meanb0_reg.mat /scratch/niesens/20230914/ReconstructedImages/rois uFA NowLteComp QTI


