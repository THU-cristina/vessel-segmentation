#######
# Rohdaten werden ausgelesen, verarbeitet (Kontrast, etc.),
# in Graustufe umgewandelt und abgespeichert
#######

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import scipy
import scipy.ndimage

# ++++++ Definitions +++++++ #

### READ IMAGES IN ###
def readImages(img):
    list = []
    for i in range(len(img)):
        #print(__file__ + "/../Retina_sortiert/" + img[i])
        list.append(cv2.imread(__file__ + "/../Retina_sortiert/" + img[i]))

    return list


### LAB to RGB ###
def convertToRGB(img):
    list = []
    for i in range(len(img)):
        
        #images werden als parameter übergeben
        
        #convert image zu lab
        imgLab = cv2.cvtColor(img[i], cv2.COLOR_BGR2LAB)

        #splitting in channals
        l, a, b = cv2.split(imgLab)

        #appling CLAHE zu l channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        #merge the CLAHE enhanded l channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        #convert lab into rgb
        list.append(cv2.cvtColor(limg, cv2.COLOR_LAB2BGR))

    return list


### SHARPEN IMAGE ###
def increaseSharpness(img, kernal_sharpening):
    list = []
    for i in range(len(img)):
        # Another method
        #list.append(cv2.filter2D(img[i], -1, kernal_sharpening))
        # Blur the image
        gauss = cv2.GaussianBlur(img[i], (3,3),0)
        # Apply Unsharp masking
        list.append(cv2.addWeighted(img[i], 6, gauss, -5, 2))

    return list


### LIMITED CONTRAST ### ???????
def increaseContrast(img):
    list = []
    #create a CLAHE object
    for i in range(len(img)):
        #equ = cv2.equalizeHist(img[i])
        #list.append(equ)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #rauschen unbedingt unterdrücken TODO
        # median filter
        median_filtered = scipy.ndimage.median_filter(clahe, size=3)
        list.append(clahe.apply(img[i]))
        cv2.imshow("img", img[0])
        cv2.waitKey()
        exit()

    return list


### GRAY-SCALE CONVERSION ###  ---> input is RGB IMAGES !
def convertToGrayscale(img):
    list = []
    for i in range(len(img)):
        # Rauschunterdrückung?
        #dst = cv2.fastNlMeansDenoisingColored(img[i],None,4,4,9,21)
        list.append(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY))

    return list


### Variablen

# kernal for sharping operation
kernal_sharpening = np.array([[-1/8, -1/8, -1/8],  [-1/8, 8, -1/8],  [-1/8, -1/8, -1/8]])


# TODO
imgGammaList = []


### Functions

# putting files in list
mypath =  __file__ + "/../Retina_sortiert"
files = []
for file in os.listdir(mypath):
    files.append(file)


# Operatoren (Reihenfolge: Schräfe, Kontrast,Graustufe)
#1
imgReadList = readImages(files)
exit()
#2
#imgSharpList = increaseSharpness(imgReadList, kernal_sharpening) #soll übersprungen werden
#3
imgRGBList = convertToRGB(imgReadList)
#4
imgGrayList = convertToGrayscale(imgRGBList)
#5
imgContList = increaseContrast(imgGrayList) #viel zu hoch

# change directory
os.chdir(__file__ + "/../Schwarz_Weiss/")


# safe images
for i in range(len(imgContList)):
    cv2.imwrite("SchwarzWeiss_" + str(i+20) + ".png", imgContList[i]) # in eigenständige Funktion packen
    cv2.imshow("graustufe", imgContList[i])
    cv2.waitKey(0)
#TODO skalierung, auflösung, Drehung
