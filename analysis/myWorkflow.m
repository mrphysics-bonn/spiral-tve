clear all
close all
clc

%% Create xps-file (correct sequence path)

% Define path to nii file 
filepath = '/scratch/niesens/20230914/ReconstructedImages/qMasLte2000/';
fn_STE_zip = strcat(filepath,'Reco_qMasLte2000_moco_unwarped_STE.nii.gz');
fn_LTE_zip = strcat(filepath,'Reco_qMasLte2000_moco_unwarped_LTE.nii.gz');
gunzip(fn_STE_zip)
gunzip(fn_LTE_zip)
fn_STE = strcat(filepath,'Reco_qMasLte2000_moco_unwarped_STE.nii');
fn_LTE = strcat(filepath,'Reco_qMasLte2000_moco_unwarped_LTE.nii');

% Create a cell array of s structures
s{1}.nii_fn = fn_STE;
s{2}.nii_fn = fn_LTE;

% Get path to corresponding bval/bvec of STE and LTE respectively
bval_fn = strings(1,2);
bval_fn(1) = strcat(filepath,'Reco_qMasLte2000_moco_unwarped_STE.bval');
bval_fn(2) = strcat(filepath,'Reco_qMasLte2000_moco_unwarped_LTE.bval');

bvec_fn = strings(1,2);
bvec_fn(1) = strcat(filepath,'Reco_qMasLte2000_moco_unwarped_STE.bvec');
bvec_fn(2) = strcat(filepath,'Reco_qMasLte2000_moco_unwarped_LTE.bvec');

% Corresponding b-tensor shapes. In this case: spherical, linear
b_deltas  = [0 1];

% Loop over nii files to create partial xps structures, and store them in the cell array.
for i = 1:length(s)
    s{i}.xps = mdm_xps_from_bval_bvec(bval_fn(i), bvec_fn(i), b_deltas(i));
end

% Define a name for the merged nii (output)
merged_nii_path = filepath;
merged_nii_name = 'Reco_qMasLte2000_moco_unwarped_merged';

% Merge the s structure, and save the merged nii along with its corresponding xps.mat file.
s_merged = mdm_s_merge(s, merged_nii_path, merged_nii_name);

%% calculate divide-method

outputpath = strcat(filepath,'output/');
mkdir(outputpath)

parpool;
mdm_fit --data /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/Reco_qMasLte2000_moco_unwarped_merged.nii.gz --mask /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/008-dzne-bn_MPRAGE_0_bet_mask_diff.nii.gz --method dtd_gamma --out /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/output/ --xps /scratch/niesens/20230914/ReconstructedImages/qMasLte2000/Reco_qMasLte2000_moco_unwarped_merged_xps.mat;
p = gcp('nocreate');
delete(p)
