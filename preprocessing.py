import nibabel as nib 
from pathlib import Path
import cv2
import numpy as np
import napari
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_image_and_apply_gaussian(file_path, sigma):
    if file_path.endswith('.nii.gz'):
        brain_image = nib.load(file_path)
        brain_image_array = brain_image.get_fdata()
    else:
        brain_image_array = cv2.imread(file_path)

    # Apply Gaussian blur only if the input is a NumPy array (not a NIfTI object)
    if isinstance(brain_image_array, np.ndarray):
        blurred_image = cv2.GaussianBlur(brain_image_array, (5, 5), sigma)
    else:
        blurred_image = None

    return blurred_image

def z_score_normalize(image):
    mean_value = np.mean(image)
    std_value = np.std(image)
    normalized_image = (image - mean_value) / std_value
    return normalized_image
