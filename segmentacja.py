import scipy.io
import cv2
import numpy as np


img = cv2.imread('vidf1_33_002_f026_prep.png')
img = cv2.imread('vidf1_33_002_f200.png')

def segmentacja(path = "C:/Users/Praca/Desktop/Praca_inzynierska/ucsdpeds/vidf/vidf1_33_000.y/vidf1_33_000_f001.png"):

    # opening files
    dmap_file = open('dmap.txt', 'w')
    img_maped_file = open('img_maped.txt', 'w')
    binarized_file = open('binarized.txt', 'w')
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #loading roi
    mat  = scipy.io.loadmat('vidf1_33_roi_mainwalkway.mat')
    roi = mat['roi'][0]
    roi = np.array(roi[0][2])

    #binarization
    ret, binarized = cv2.threshold(img,90,255,cv2.THRESH_BINARY_INV)



    #binarized roi
    binarized_roi = cv2.bitwise_and(binarized,binarized,mask = roi)
    binarized_roi = binarized_roi/255
    binarized_roi = binarized_roi.astype(np.uint8)
    img_masked= cv2.multiply(img,binarized_roi)
    #showing images
    cv2.imshow('img',img)
    #cv2.imshow('img_roi',img_roi)
    #cv2.imshow('img_masked',img_masked)
    #cv2.imshow('map',dmap_uint8)
    #cv2.imshow('binarized',binarized)
    #cv2.imshow('binarized_roi',binarized_roi)
    #cv2.imshow('img_binarized',img_maped)
    #cv2.waitKey(1)
    return(img_masked)


if __name__ == "__main__":
    segmentacja()