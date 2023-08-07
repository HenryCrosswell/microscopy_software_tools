from preprocessing import load_image_and_apply_gaussian, z_score_normalize
import napari
import nibabel as nib

image_filepath = 'files/BIDS dataset/sub-01/anat/sub-01_T1w.nii.gz'
brain_image = nib.load(image_filepath)
brain_image_array = brain_image.get_fdata()

gaussian_brain_image = load_image_and_apply_gaussian(image_filepath, 1)
norm_gaussian_brain_image = z_score_normalize(gaussian_brain_image)

with napari.gui_qt():
    viewer = napari.view_image(gaussian_brain_image, colormap='gray', name='Blurred Image')
    viewer.add_image(brain_image_array, colormap='gray', name='Unblurred Image')
    viewer.add_image(norm_gaussian_brain_image, colormap='gray', name='normalised&blurred Image')
