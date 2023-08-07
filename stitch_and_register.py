import logging
import cv2
import numpy as np
import SimpleITK as sitk

def register_images(images):
    registered_images = [images[0]]  # The first image is the reference

    for i in range(1, len(images)):
        logging.info(f"Registering image {i + 1}...")
        image = images[i]
        
        # Perform image registration using feature matching, homography, etc.
        registered_image = image_registration_sitk(registered_images[i - 1], image)
        
        registered_images.append(registered_image)

    return registered_images

def image_registration_sitk(reference, image_to_register):
    # Convert images to SimpleITK format
    reference_sitk = sitk.GetImageFromArray(reference)
    image_to_register_sitk = sitk.GetImageFromArray(image_to_register)
    
    # Set up registration parameters
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()  # You can use other metrics as needed
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=sitk.ImageRegistrationMethod.EachIteration)
    registration_method.SetInitialTransform(sitk.TranslationTransform(reference_sitk.GetDimension()))
    
    # Perform image registration
    final_transform = registration_method.Execute(reference_sitk, image_to_register_sitk)
    
    # Apply the transformation to the image
    registered_image_sitk = sitk.Resample(image_to_register_sitk, reference_sitk, final_transform, sitk.sitkLinear, 0.0)
    
    # Convert the registered image back to a NumPy array
    registered_image = sitk.GetArrayFromImage(registered_image_sitk)
    
    return registered_image


def stitch_images(registered_images):
    logging.info("Stitching images together...")
    
    # Combine registered images using a stitching algorithm
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(registered_images)
    
    if status == cv2.Stitcher_OK:
        return stitched_image
    else:
        logging.error("Image stitching failed!")
        return None