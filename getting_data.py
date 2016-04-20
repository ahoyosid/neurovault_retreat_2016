import numpy as np
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import glob
from nilearn.input_data import NiftiMasker
from scipy.ndimage import binary_dilation, binary_erosion
from nilearn.plotting import plot_stat_map, plot_roi

plt.close('all')

def _my_mkdirs(paths):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def _dilatea_and_erode(img, iterations=1):
    dilated_img = binary_dilation(img, iterations=iterations)
    eroded_img = binary_erosion(dilated_img, iterations=iterations)
    return eroded_img

root_dir = "/media/ahoyosid/Seagate Backup Plus Drive/neurovault_analysis/data"
mask = pjoin(root_dir, 'MNI152_T1_3mm_brain_mask.nii.gz')
# mask = "/usr/share/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"
memory = 'cache'

meta_data = pd.read_csv(pjoin(root_dir, 'metadata.csv'))

labels = meta_data[['image_id', 'map_type']].sort('image_id')
manual_labels = pd.read_csv('manual_label.csv')

labels['is_yannick_brain'] = 1
labels.at[labels.image_id.isin(manual_labels.image_id), 'is_yannick_brain'] = 0
labels.to_csv('data_labels.csv')

# directory = 'original'
# # directory = 'resampled_3mm'

# original_mask = "masks"
# report_dir = pjoin('report', directory)

# mask_dir = pjoin(report_dir, 'masks')
# mask_niimg_dir = pjoin(mask_dir, 'niimg')
# mask_plot_dir = pjoin(mask_dir, 'plots')
# masked_plot_dir = pjoin(mask_dir, 'masked_niimg')

# _my_mkdirs([report_dir, mask_dir, mask_niimg_dir, mask_plot_dir,
#             masked_plot_dir])

# niimgs = glob.glob(pjoin(root_dir, directory, '*.nii.gz'))
# masker = NiftiMasker(mask_img=mask, memory=memory)
# niimgs.remove(pjoin(root_dir, directory, '0407.nii.gz'))
# import pdb; pdb.set_trace()  # XXX BREAKPOINT

# for niimg in sorted(niimgs):
#     plt.close('all')
#     niimg_name = niimg.split(os.sep)[-1].split('.')
#     mask_name = niimg_name[0] + '_mask'

#     img = nb.load(niimg)
#     # affine = img.get_affine()
#     data = img.get_data()

#     if data.sum() != 0:
#         display_niimg = plot_stat_map(img) #, colorbar=True)
#         display_niimg.savefig(pjoin(masked_plot_dir, mask_name + 'png'))
#     # else:
#     #     print "Niimg %s is cero" % mask_name
#     # bg = np.logical_not(np.logical_or(data == 0, np.isnan(data)))
#     # mask_img = _dilatea_and_erode(bg).astype(np.float)

#     # mask_niimg = nb.Nifti1Image(mask_img, affine)
#     # masked_niimg = nb.Nifti1Image(data * mask_img, affine)

#     # display_mask = plot_roi(mask_niimg)
#     # display_niimg = plot_stat_map(masked_niimg) #, colorbar=True)

#     # # mask_niimg.to_filename(pjoin(mask_niimg_dir, mask_name + '.nii.gz'))
#     # display_niimg.savefig(pjoin(masked_plot_dir, mask_name + 'png'))
#     # display_mask.savefig(pjoin(mask_plot_dir, mask_name + 'png'))



