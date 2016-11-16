"""
Module for texture features calculation
It calculates entropy, homogenity and energy.
These features are used in people number estimation
"""
import os
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix

def texture_features_calculation(input_image=os.path.join('vidf1_33_000_f123.png'),
                                 no_of_gray_levels=4,
                                 distance=1,
                                 teta=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """"
    Texture features calculation
    Parameters:
    image to process
    number of grey scale levels
    distance for neighborhood checking
    angle
    Returns:
    List of features: for eeach angle 3 features are calculated 
    Output looks like: [homogenity for angle1, energy for angle1, entropy1, etc]
    """
#   Input image reading in grayscale
    img = input_image
#   Pixel association with gray levels
    interval = 256/no_of_gray_levels    
    img_copy = np.copy(img)
    
    
    for pixel in np.nditer(img_copy, op_flags=['readwrite']):
        for level in range(0, no_of_gray_levels):
            if pixel >= level*interval and pixel <= interval+level*interval:
                pixel[...] = level

#   GLCM calculation using greycom1atrix function from skimage
    P = greycomatrix(img_copy,
                     distances = map(int, str(distance)),
                     angles = teta,
                     levels = no_of_gray_levels,
                     symmetric = True,
                     normed = True)

#   TODO: avoid matriList by iteration over P
    matrix_list = []
    for angle in range(0, len(teta)):
        matrix_list.append(P[:,:,0,angle])
#   For each angle 3 features are calculated
    feature_list = []

    for matrix in matrix_list:
        energy = 0
        homogenity = 0
        entropy = 0
        for index, value in np.ndenumerate(matrix):

            homogenity += np.divide(value, (1+np.square(index[0]-index[1])))
            energy += np.square(value)
            # Avoid ln(0)
            if value == 0:
                pass
            else:
                entropy += np.multiply(value, np.log(value))

        feature_list.append(homogenity)
        feature_list.append(energy)
        feature_list.append(-entropy)

    print feature_list
    return feature_list

if __name__ == "__main__":
    start_time = time.time()
    texture_features_calculation()
    print("--- %s seconds ---" % (time.time() - start_time))
