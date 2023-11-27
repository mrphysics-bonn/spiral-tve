"""
dtd workflow for divide and qti approach
created: 22.09.2023
"""
import numpy as np
import os
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import matplotlib.pyplot as plt
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
import matplotlib.pyplot as plt
import numpy as np
import dipy.reconst.qti as qti
from scipy.io import savemat
from scipy import stats
from multiprocessing import Pool
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
import pandas as pd
import seaborn as sns
import nibabel as nib

#%% First step - preprocessing

# after mrd2nifti.py run process_spiral_Fast.sh

#%% Second step - prepare datasets for Matlab and Dipy dtd-analysis

Divide = 0
Qti = 1

# configure filenames
seq = 'qti_NowLteDetuned'
data_dir = '/scratch/niesens/20230914/ReconstructedImages/'+seq+'/'
filename_data = 'Reco_'+seq+'_moco_unwarped' # filename without .nii.gz
filename_mask = data_dir+'008-dzne-bn_MPRAGE_0_bet_mask_diff'
filename_bvec = 'Reco_'+seq+'_moco_unwarped_bet'
filename_bval = 'Reco_'+seq+'_moco_unwarped_bet'
filename_bvec_orig = 'Reco_'+seq
filename_bval_orig = 'Reco_'+seq
filename_bDelta = 'Reco_'+seq
outfile = 'Reco_'+seq

# Read data
filepath_data = data_dir + filename_data + '.nii.gz'
data, affine = load_nifti(filepath_data)
filepath_mask = filename_mask + '.nii.gz'
mask = load_nifti(filepath_mask)[0][...,np.newaxis]
data = mask * data
print('data.shape (%d, %d, %d, %d)' % data.shape)

# Read bvec and bval after motion correction
filepath_bvec = data_dir + os.path.splitext(filename_bvec)[0] + '.bvec'
filepath_bval = data_dir + os.path.splitext(filename_bval)[0] + '.bval'
bvals, bvecs = read_bvals_bvecs(filepath_bval, filepath_bvec)

# Read bvec before motion correction to rotate calculated b-tensor
filepath_bvec_orig = data_dir + os.path.splitext(filename_bvec_orig)[0] + '.bvec'
filepath_bval_orig = data_dir + os.path.splitext(filename_bval_orig)[0] + '.bval'
_, bvecs_orig = read_bvals_bvecs(filepath_bval_orig, filepath_bvec_orig)

# Read bDelta file
filepath_bDelta = data_dir + os.path.splitext(filename_bDelta)[0] + '.bDelta'
if os.path.exists(filepath_bDelta):
    f = open(filepath_bDelta, 'r')
    bDeltas = f.read().split()
    f.close()
for k,elem in enumerate(bDeltas):
    if float(elem) == -1:
        elem = -0.5
    bDeltas[k] = float(elem)
bDeltas = np.asarray(bDeltas)

if Divide:
    # seperate STE, LTE for divide-approach
    data_Ste = data[:,:,:,bDeltas==0]
    save_nifti(data_dir + filename_data + '_STE.nii.gz', data_Ste.astype(np.float32), affine) # np.complex64 ; np.float32
    bvals_Ste = bvals[bDeltas==0]
    bvecs_Ste = bvecs[bDeltas==0]
    np.savetxt(data_dir + filename_data + '_STE.bval', np.atleast_2d(bvals_Ste), fmt='%i')
    np.savetxt(data_dir + filename_data + '_STE.bvec', bvecs_Ste.T, fmt='%1.6f')

    data_Lte = data[:,:,:,bDeltas==1]
    save_nifti(data_dir + filename_data + '_LTE.nii.gz', data_Lte.astype(np.float32), affine) # np.complex64 ; np.float32
    bvals_Lte = bvals[bDeltas==1]
    bvecs_Lte = bvecs[bDeltas==1]
    np.savetxt(data_dir + filename_data + '_LTE.bval', np.atleast_2d(bvals_Lte), fmt='%i')
    np.savetxt(data_dir + filename_data + '_LTE.bvec', bvecs_Lte.T, fmt='%1.6f')

    # calculate DT from LTE signal for FA-map and RGB-map
    gtab_Lte = gradient_table(bvals_Lte, bvecs_Lte)
    tenmodel_Lte = dti.TensorModel(gtab_Lte)
    tenfit_Lte = tenmodel_Lte.fit(data_Lte)
    FA_Lte = fractional_anisotropy(tenfit_Lte.evals)
    FA_Lte[np.isnan(FA_Lte)] = 0
    save_nifti(data_dir+outfile+'_Lte_fa.nii.gz', FA_Lte.astype(np.float32), affine)
    FA_Lte = np.clip(FA_Lte, 0, 1)
    RGB_Lte = color_fa(FA_Lte, tenfit_Lte.evecs)
    save_nifti(data_dir+outfile+'_Lte_rgb.nii.gz',np.array(255 * RGB_Lte, 'uint8'), affine)

if Qti:
    # prepare and save btens for qti-approach
    btens = np.load(data_dir+'btens.npy')
    A = np.array([[1,0,0],[0,1,0],[0,0,-1]])
    btens = np.matmul(np.matmul(A,btens),A.transpose()) # from mrd2nifti.py
    B_orig = bvecs_orig.T
    B = bvecs.T
    A2 = np.dot(B, np.linalg.pinv(B_orig))
    btens = np.matmul(np.matmul(A2,btens),A2.transpose()) # from preprocessing
    savemat(data_dir + filename_data + "_btens.mat", {'btens': btens*1e-3}) # save for Matlab
    np.save('btens',btens) # save for DIPY

#%%  Third step - QTI with DIPY (see Matlab script 'myWorkflow.m' (divide) and 'myWorkflow_btens.m' (qti))

btens_real = 1 # use btens calculated from sequence
with_b0 = 1 # include b0 images in fitting
fit_method = 'SDPdc' # 'WLS'; qti-fitting method; recommended: SDPdc

if btens_real:
    btens = np.load('btens.npy')
else:
    btens = []
    for i in bDeltas:
        if i==-0.5:
            btens.append('PTE')
        elif i==0:
            btens.append('STE')
        elif i==1:
            btens.append('LTE')
    btens = np.asarray(btens)

# Gradient table
if with_b0:
    gtab = gradient_table(bvals, bvecs, btens=btens)
else:
    gtab = gradient_table(bvals[bvals!=0], bvecs[bvals!=0], btens=btens[bvals!=0], b0_threshold=102)
    data = data[:,:,:,bvals!=0]
ranks = np.array([np.linalg.matrix_rank(b) for b in gtab.btens])
print('%s volumes with b=0' % (np.sum(ranks == 0)))
print('%s volumes with PTE' % (np.sum(ranks == 2)))
print('%s volumes with STE' % (np.sum(ranks == 3)))
print('%s volumes with LTE' % (np.sum(ranks == 1)))

# Fit for parameter maps
qtimodel = qti.QtiModel(gtab,fit_method)
data = np.where(data < 0 , 0, data)

keys = ['fa', 'ufa', 'md', 'v_md', 'v_shear', 'v_iso', 'c_md', 'c_mu', 'c_m', 'c_c', 'mk', 'k_bulk', 'k_shear', 'k_mu', 'fa_rgb']
results_metrics = {key: np.zeros_like(load_nifti(filepath_mask)[0]) for key in keys}
fa_rgb = np.zeros((load_nifti(filepath_mask)[0].shape[0],load_nifti(filepath_mask)[0].shape[1],load_nifti(filepath_mask)[0].shape[2],3))

if fit_method == 'SDPdc':
    def qtifit_slices(slice):
        return qtimodel.fit(data[:,:,slice,:], mask = load_nifti(filepath_mask)[0][:,:,slice])

    pool = Pool(processes=8)
    qtifit_list = pool.map(qtifit_slices, [slice for slice in range(data.shape[2])])
    pool.close()
    pool.join()
    # Compute parameter maps
    for slice, qtifit in enumerate(qtifit_list):
        results_metrics['fa'][:,:,slice] = qtifit.fa
        results_metrics['ufa'][:,:,slice] = qtifit.ufa
        results_metrics['md'][:,:,slice] = qtifit.md
        results_metrics['v_md'][:,:,slice] = qtifit.v_md
        results_metrics['v_shear'][:,:,slice] = qtifit.v_shear
        results_metrics['v_iso'][:,:,slice] = qtifit.v_iso
        results_metrics['c_md'][:,:,slice] = qtifit.c_md
        results_metrics['c_mu'][:,:,slice] = qtifit.c_mu
        results_metrics['c_m'][:,:,slice] = qtifit.c_m
        results_metrics['c_c'][:,:,slice] = qtifit.c_c
        results_metrics['mk'][:,:,slice] = qtifit.mk
        results_metrics['k_bulk'][:,:,slice] = qtifit.k_bulk
        results_metrics['k_shear'][:,:,slice] = qtifit.k_shear
        results_metrics['k_mu'][:,:,slice] = qtifit.k_mu
        fa_rgb[:,:,slice,:] = qtifit.fa_rgb
else:
    qtifit = qtimodel.fit(data, mask = load_nifti(filepath_mask)[0])
    # Compute parameter maps
    results_metrics['fa'] = qtifit.fa
    results_metrics['ufa'] = qtifit.ufa
    results_metrics['md'] = qtifit.md
    results_metrics['v_md'] = qtifit.v_md
    results_metrics['v_shear'] = qtifit.v_shear
    results_metrics['v_iso'] = qtifit.v_iso
    results_metrics['c_md'] = qtifit.c_md
    results_metrics['c_mu'] = qtifit.c_mu
    results_metrics['c_m'] = qtifit.c_m
    results_metrics['c_c'] = qtifit.c_c
    results_metrics['mk'] = qtifit.mk
    results_metrics['k_bulk'] = qtifit.k_bulk
    results_metrics['k_shear'] = qtifit.k_shear
    results_metrics['k_mu'] = qtifit.k_mu
    results_metrics['fa_rgb'] = qtifit.fa_rgb

# Plotting of dtd-metrics maps
z = 36

fig, ax = plt.subplots(3, 4, figsize=(12, 9))

background = np.zeros(data.shape[0:2])  # Black background for figures
for i in range(3):
    for j in range(4):
        ax[i, j].imshow(background, cmap='gray')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

ax[0, 0].imshow(np.rot90(results_metrics['md'][:, :,z]), cmap='gray', vmin=0, vmax=3e-3)
ax[0, 0].set_title('MD')
ax[0, 1].imshow(np.rot90(results_metrics['v_md'][:, :,z]),
                cmap='gray', vmin=0, vmax=.1e-5)
ax[0, 1].set_title('V_MD')
ax[0, 2].imshow(np.rot90(results_metrics['v_shear'][:, :,z]), cmap='gray', vmin=0,
                vmax=.9e-6)
ax[0, 2].set_title('V_shear')
ax[0, 3].imshow(np.rot90(results_metrics['v_iso'][:, :,z]),
                cmap='gray', vmin=0, vmax=0.2e-5)
ax[0, 3].set_title('V_iso')

ax[1, 0].imshow(np.rot90(results_metrics['c_md'][:, :,z]), cmap='gray', vmin=0, vmax=0.5)
ax[1, 0].set_title('C_MD')
ax[1, 1].imshow(np.rot90(results_metrics['c_mu'][:, :,z]), cmap='gray', vmin=0, vmax=1)
ax[1, 1].set_title('C_μ = μFA^2')
ax[1, 2].imshow(np.rot90(results_metrics['c_m'][:, :,z]), cmap='gray', vmin=0, vmax=1)
ax[1, 2].set_title('C_M = FA^2')
ax[1, 3].imshow(np.rot90(results_metrics['c_c'][:, :,z]), cmap='gray', vmin=0, vmax=1)
ax[1, 3].set_title('C_c')

ax[2, 0].imshow(np.rot90(results_metrics['mk'][:, :,z]), cmap='gray', vmin=0, vmax=1.5)
ax[2, 0].set_title('MK')
ax[2, 1].imshow(np.rot90(results_metrics['k_bulk'][:, :,z]),
                cmap='gray', vmin=0, vmax=1.5)
ax[2, 1].set_title('K_bulk')
ax[2, 2].imshow(np.rot90(results_metrics['k_shear'][:, :,z]), cmap='gray', vmin=0,
                vmax=1.5)
ax[2, 2].set_title('K_shear')
ax[2, 3].imshow(np.rot90(results_metrics['k_mu'][:, :,z]), cmap='gray', vmin=0, vmax=1.5)
ax[2, 3].set_title('K_μ')

fig.tight_layout()
plt.show()

# save parameter maps
image_list = [img for img in results_metrics.values()]
results_4dimage = np.stack(image_list, axis=0).transpose(1, 2, 3, 0)

save_nifti(data_dir+'output/'+outfile+'_results.nii.gz', results_4dimage.astype(np.float32), affine)
save_nifti(data_dir+'output/'+outfile+'_md.nii.gz', results_metrics['md'].astype(np.float32), affine)
save_nifti(data_dir+'output/'+outfile+'_fa.nii.gz', results_metrics['fa'].astype(np.float32), affine)
save_nifti(data_dir+'output/'+outfile+'_ufa.nii.gz', results_metrics['ufa'].astype(np.float32), affine)
save_nifti(data_dir+'output/'+outfile+'_c_md.nii.gz', results_metrics['c_md'].astype(np.float32), affine)
save_nifti(data_dir+'output/'+outfile+'_c_c.nii.gz', results_metrics['c_c'].astype(np.float32), affine)
save_nifti(data_dir+'output/'+outfile+'_rgb.nii.gz', fa_rgb.astype(np.float32), affine)

# import dill

# # Save object with dill
# with open('qtifit_list.pkl', 'wb') as f:
#     dill.dump(qtifit_list, f)

# # Load object with dill
# with open('qtifit_list.pkl', 'rb') as f:
#     loaded_object = dill.load(f)

#%% Fourth step - parameter maps to T1 and segmentation to CSF, GM, WM -> parameter maps to T1 space and ROI analysis

# run map_segmentation.sh (or use segmentation_pipe.sh to process dtd metrics together)
# run map_segmentation_T1.sh -> for segmentation in T1 space (or use segmentation_pipe_T1.sh to process dtd metrics together)

#%% Fifth step - Statistical analysis of the dtd-metrics - imported into map_segmentation.sh; here: only for plotting

# load dtd-metrics maps
filepath_mapFull = '/scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/output_Dipy/qti_real_cls_noslice/Reco_qti_NowLteDetuned_qti_c_md_real_c_full_MNI.nii.gz'
mapFull = load_nifti(filepath_mapFull)[0]

filepath_map = '/scratch/niesens/20230914/ReconstructedImages/qti_NowLteDetuned/output_Dipy/qti_real_cls_noslice_T1/Reco_qti_NowLteDetuned_qti_md_real_c_full_reg_Spleniumofcorpuscallosum_T1.nii.gz'
map, map_affine = load_nifti(filepath_map)

region = 'Splenium' # Splenium, LateralVentricle, AnteriorCR
dtd = 'MD' # MD, FA, uFA, MKi, OP
sequence = 'NowLteDetuned'
approach = 'QTI' # DIVIDE, QTI

# transform 3D vectors into 1D vectors:
if approach == 'QTI' and dtd == 'MD':
    map = map*1e3
if dtd == 'MKi':
    if approach == 'DIVIDE':
        mapFull=mapFull[mapFull<10]
        map = map[map<10]
    #map = map/mapFull.max()
map_flat = map.flatten()
map_flat = map_flat[~np.isnan(map_flat)]
map_flat = map_flat[map_flat!=0]

# Sample Size
N1 = len(map_flat)

# Degrees of freedom  
dof = N1 - 1
sample_standard_deviation = np.std(map_flat, ddof=1)

# Plot of histogram
plt.hist(map_flat, bins=100, density=True, color='gray', alpha=1)
plt.xlabel('MD [µm²/s]')
plt.ylabel('density distribution')
plt.title('histogram')
plt.grid(True)
#plt.xlim([0,4])
#plt.ylim([0,0.12])
plt.plot([map_flat.mean(),map_flat.mean()],[0,10])
plt.text(map_flat.mean(),3, r'${:.5f}$'.format(map_flat.mean()), horizontalalignment='center',verticalalignment='center',color = 'b',fontsize=20)
plt.show()

print("mean = " + str(map_flat.mean()))
print("std = " + str(sample_standard_deviation))
print("interval = [" + str(map_flat.mean()-sample_standard_deviation) + ',' + str(map_flat.mean()+sample_standard_deviation) + ']')

#%% Sixth step - pandas dataframe for better visualization and comparison

region = 'AnteriorCR' # Splenium, LateralVentricle, AnteriorCR
dtd = 'MD' # MD, FA, uFA, MKi, OP
dti_val = 1 # use dti sequence for comparison, else: use literature values

file_path = "dataframe_"+dtd+"_abstract.csv"
if os.path.exists(file_path):
    df_read = pd.read_csv(file_path)
    print(df_read)
else:
    print(f"File {file_path} does not exist!")

if dti_val:
    file_path_dti = "dataframe_"+dtd+"_dti1.csv"
    if os.path.exists(file_path_dti):
        df_read_dti = pd.read_csv(file_path_dti)
        grouped_dti = df_read_dti.loc[df_read_dti['ROI'] == region].groupby(['Sequence'])[dtd]
        lit_mean = grouped_dti.mean().values[0]
        lit_std = grouped_dti.std().values[0]
        if dtd == 'MD':
            lit_mean = lit_mean * 1e3 # [µm²/ms]
            lit_std = lit_std * 1e3 # [µm²/ms]
    else:
        print(f"File {file_path_dti} does not exist!")
else:
    if region == 'Splenium':
        if dtd == 'MD': # [µm²/ms]
            lit_mean = 0.739082
            lit_std = 3.36351e-02
        elif dtd == 'FA':
            lit_mean = 0.788438
            lit_std = 0.0225405
    elif region == 'AnteriorCR':
        if dtd == 'MD': # [µm²/ms]
            lit_mean = 0.7732705
            lit_std = 4.4731950000000005e-02
        elif dtd == 'FA':
            lit_mean = 0.45581499999999997
            lit_std = 0.0319375
    elif region == 'LateralVentricle':
        if dtd == 'MD': # [µm²/ms]
            lit_mean = 2.844
            lit_std = 0.00071
        elif dtd == 'FA':
            lit_mean = 0.091
            lit_std = 0.012

# Make violin plot
sns.violinplot(x=df_read.loc[df_read['ROI'] == region]["Sequence"], y=df_read[dtd]) # hue=df_read["Approach"], split=True,inner=None ;*1e3
ax = plt.gca()
ax.legend().set_visible(False)
# legend = ax.legend(fontsize=24,framealpha=1)
# for line in legend.get_lines():
#     line.set_linewidth(2.0)
ax.set_ylim(bottom=0.4,top=1.1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=24)
ax.set_xlabel('Sequence', fontsize=26)
if dtd == 'MD':
    ax.set_ylabel(dtd+'[µm²/ms]', fontsize=26)
else:
    ax.set_ylabel(dtd, fontsize=26)
#ax.set_yticks(np.arange(2, 4.5, 0.5))
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=24)
#ax.set_title(region, fontsize=14)
if dtd == 'MD' or dtd == 'FA':
    ax.axhline(lit_mean, color='k', linewidth=1)
    ax.axhline(lit_mean+lit_std, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(lit_mean-lit_std, color='k', linestyle='--', linewidth=0.5)
plt.show()

# calculate mean and standard deviation based on sequence and approach
grouped = df_read.loc[df_read['ROI'] == region].groupby(['Sequence', 'Approach'])[dtd]
means = grouped.mean()
stds = grouped.std()

# DataFrame for means and standard deviations
df_stats = pd.DataFrame({
    'ROI': [region]*len(means.values),
    'DTD': [dtd]*len(means.values),
    'Sequence': [index[0] for index in means.index],
    'Approach': [index[1] for index in means.index],
    'Mean': means.values, # *1e3
    'Standard Deviation': stds.values # *1e3
})

print(df_stats)

fig, ax = plt.subplots()
for approach in df_stats['Approach'].unique():
    subset = df_stats[df_stats['Approach'] == approach]
    plt.scatter(subset['Sequence'], subset['Mean'], marker='x', s=400, linewidth=4, edgecolor='black')
sns.lineplot(data=df_stats, x='Sequence', y='Mean', hue='Approach', lw=5)
#sns.lineplot(data=df_stats, x='Sequence', y='Mean', hue='Approach', marker='x', lw=5)
ax = plt.gca()
legend = ax.legend(fontsize=26)
for line in legend.get_lines():
    line.set_linewidth(6.0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=24)
ax.set_xlabel(r'$\hat{B}-\mathrm{tensor}$', fontsize=26)
if dtd == 'MD':
    ax.set_ylabel(dtd+'[µm²/ms]', fontsize=26)
else:
    ax.set_ylabel(dtd, fontsize=26)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=24)
#ax.set_title(region, fontsize=16)
# if dtd == 'MD' or dtd == 'FA':
#     ax.axhline(lit_mean, color='k', linewidth=1)
#     ax.axhline(lit_mean+lit_std, color='k', linestyle='--', linewidth=0.5)
#     ax.axhline(lit_mean-lit_std, color='k', linestyle='--', linewidth=0.5)
plt.show()

#%% Optional step: Plot noise map as histogram

fn = '/groups/ag-stoecker/user/niesens/ReconstructedImages/20230914/ReconstructedImages/NowLteDetunedSNR/Reco_NowLteDetuned_SNR_snr_reg_Spleniumofcorpuscallosum.nii.gz'
img_nifti = nib.load(fn)
img_data = img_nifti.get_fdata()

# Flatten of 2D array
img_data_flat = img_data.flatten()
img_data_flat = img_data_flat[img_data_flat != 0]

# Plot of histogram
histo = plt.hist(img_data_flat, bins=1000, density=True, color='gray', alpha=1)
density_maximum = np.amax(histo[0])
SNR_maximum = histo[1][np.argmax(histo[0])]
plt.xlabel('SNR', fontsize=26)
plt.ylabel('Density distribution', fontsize=26)
plt.grid(True)
plt.xlim([0, 100])
plt.ylim([0, 0.08])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.plot([img_data_flat.mean(), img_data_flat.mean()],
         [0, 0.2],lw=5)
plt.plot([SNR_maximum, SNR_maximum],
        [0, density_maximum],lw=2,color='g')
plt.plot([0, SNR_maximum],
        [density_maximum, density_maximum],lw=2,color='g')
plt.text(50, 0.01, r'${}$'.format(round(img_data_flat.mean(), 3)),
         horizontalalignment='center', verticalalignment='center', color='b', fontsize=24)
plt.text(54, 0.045, f'({round(SNR_maximum,3)},{round(density_maximum,3)})',
         horizontalalignment='center', verticalalignment='center', color='g', fontsize=24)
plt.show()

print('density_maximum = '+str(density_maximum))
print('SNR_maximum = '+str(SNR_maximum))