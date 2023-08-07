import cv2
import numpy as np
import pytest
from preprocessing import load_image_and_apply_gaussian, z_score_normalize 
 
def test_load_image_and_apply_gaussian():
    file_path_nifti = 'tests/test_data/downsampled_nifti.nii.gz'
    file_path_other = 'tests/test_data/brain anatomy teaser.jpg'
    sigma = 1.0

    # Test loading and applying to NIfTI
    blurred_nifti = load_image_and_apply_gaussian(file_path_nifti, sigma)
    assert blurred_nifti is not None

    # Test loading and applying to other image format
    blurred_other = load_image_and_apply_gaussian(file_path_other, sigma)
    assert blurred_other is not None


def test_z_score_normalize():
    # Create a sample image
    image = np.array([[10, 20, 30],
                      [40, 50, 60],
                      [70, 80, 90]])

    # Calculate mean and standard deviation
    mean_value = np.mean(image)
    std_value = np.std(image)

    # Normalize the image using the function
    normalized_image = z_score_normalize(image)

    # Calculate the expected normalized image using the formula
    expected_normalized_image = (image - mean_value) / std_value

    # Check if the normalized image matches the expected result
    assert np.allclose(normalized_image, expected_normalized_image)

