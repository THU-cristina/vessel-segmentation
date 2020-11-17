import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
from PIL import Image

import functions


sourceListTIF = ["/IMAGES/images_training/", "/IMAGES/images_test/"]
#targetListTIFF = ["/IMAGES/training/05_training_prep_TIFF/", "/IMAGES/test/06_test_prep_TIFF/"]

maskTempPath = ["/IMAGES/mask_training_temp/", "/IMAGES/mask_test_temp/"]
maskTargetPath = ["/IMAGES/mask_training/", "/IMAGES/mask_test/"]
datasetTargetPath = ["/IMAGES/img_training/", "/IMAGES/img_test/"]



for i in range(len(sourceListTIF)):    # length of both lists are equl

    # putting files in list
    mypath =  __file__ + sourceListTIF[i]

    filenames = []
    
    for file in os.listdir(mypath):
        filenames.append(file)
     
    imgReadList = functions.readImages(filenames, sourceListTIF[i])
   
    grayList = functions.rgb2grayscale(imgReadList)
    
    # Blur using 3 * 3 kernel. 
    grayBlurredList = functions.blur(grayList) 

    maskList = functions.generateMask(grayBlurredList)
    functions.writeImagesMask(maskList, maskTempPath[i], i)
    functions.writeTIFinGIF(maskList, maskTempPath[i], maskTargetPath[i], i)

    datasetList = functions.generateImgageWithBackground(grayBlurredList, imgReadList)
    functions.writeImagesDataset(datasetList, datasetTargetPath[i], i)
    
