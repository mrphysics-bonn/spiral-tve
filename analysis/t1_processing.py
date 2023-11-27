# Imports

import ants
import antspynet
import argparse
import nibabel as nib

def main(args):

    nifti = nib.load(args.in_file)
    ants_img = ants.image_read(args.in_file)

    mask = antspynet.brain_extraction(ants_img, modality="t1")
    denoise = ants.denoise_image(ants_img,mask=mask)
    n4 = ants.n4_bias_field_correction(denoise,mask=mask)

    thresh = 0.2
    mask = mask.numpy()
    mask[mask>thresh] = 1
    mask[mask<=thresh] = 0
    t1_preprocessed = n4.numpy()*mask

    nifti_img = nib.Nifti1Image(t1_preprocessed, affine=nifti.affine, header=nifti.header)
    nib.save(nifti_img, args.out_file+'.nii.gz')

    if args.mask:
        nifti_img = nib.Nifti1Image(mask, affine=nifti.affine, header=nifti.header)
        nib.save(nifti_img, args.out_file+'_mask.nii.gz')
    
    if args.mni_file:
        mni = ants.image_read(args.mni_file)

        registration_result = ants.registration(fixed=n4, moving=mni, type_of_transform='SyN')
        mni_to_T1 = registration_result['warpedmovout']
        
        mni_to_T1.to_file(args.mni_output+'.nii.gz')

        if args.roi_images:
            roiList = args.roi_images.split(',')
            for roi in roiList:
                roi_mni = ants.image_read(roi)
                roi_to_T1 = ants.apply_transforms(fixed=n4, moving=roi_mni, transformlist=registration_result['fwdtransforms'])
                roi_to_T1.to_file(roi.rsplit(".nii.gz", 1)[0]+'_T1.nii.gz')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='T1 image preprocessing (brain extraction denoising and n4 bias correction)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_file', type=str, help='Input T1 file')
    parser.add_argument('out_file', type=str, help='Output filename for preprocessed T1 data (without .nii.gz ending).')
    parser.add_argument('-m', '--mask', action='store_true', help='Output the brain mask.')
    parser.add_argument('-mni', '--mni_file', type=str, default=None, help='Input MNI file')
    parser.add_argument('-mni_out', '--mni_output', type=str, default=None, help='Output registered MNI file (without .nii.gz ending)')
    parser.add_argument('-roi', '--roi_images', type=str, default=None, help='List of ROIs in MNI space (e.g.: roi1,roi2,roi3,...). Use probability maps and do thresholding after registration.')

    args = parser.parse_args()

    main(args)
