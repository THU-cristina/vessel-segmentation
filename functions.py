import numpy as np
import cv2
import os
from PIL import Image


# ++++++ Function definitions +++++++ #

### READ IMAGES ###
def readImages(imgList, pathVar):
    list = []
    for i in range(len(imgList)):
        list.append(cv2.imread(__file__ + pathVar + imgList[i]))

    return list


### WRITE IMAGES (safe) ###
def writeImagesDataset(imgList, pathVar, mode): #mode: 0 = training, 1 = test
    # change directory
    os.chdir(__file__ + pathVar)
    for i in range(len(imgList)):
        if mode == 0:
            if i < 9:
                cv2.imwrite("0" + str(i+1) + "_training.tif", imgList[i])
            else:
                cv2.imwrite(str(i+1) + "_training.tif", imgList[i])
        elif mode == 1:
            if i < 9:
                cv2.imwrite("0" + str(i+1) + "_test.tif", imgList[i])
            else:
                cv2.imwrite(str(i+1) + "_test.tif", imgList[i])
        else:
            print("hier ist was schief gelaufen - writeImagesDataset. mode = " + str(mode))

### WRITE IMAGES (safe) ###
def writeImagesMask(imgList, pathVar, mode):
    # change directory
    os.chdir(__file__ + pathVar)
    for i in range(len(imgList)):
        if mode == 0:
            if i < 9:
                cv2.imwrite("0" + str(i+1) + "_training_mask.tif", imgList[i])
            else:
                cv2.imwrite(str(i+1) + "_training_mask.tif", imgList[i])
        elif mode == 1:
            if i < 9:
                cv2.imwrite("0" + str(i+1) + "_test_mask.tif", imgList[i])
            else:
                cv2.imwrite(str(i+1) + "_test_mask.tif", imgList[i])
        else:
            print("hier ist was schief gelaufen - Abspeichern von .tif Dateien ist nicht möglich, siehe mode; mode ist " + str(mode))
            exit()


def writeTIFinGIF(imgList, pathVarTemp, pathVar, mode):
    for i in range(len(imgList)):
        if mode == 0:
            if i < 9:
                img = Image.open(__file__ + pathVarTemp + "0" + str(i+1) + "_training_mask.tif")
                img.save(__file__ + pathVar + "0" + str(i+1) + "_training_mask.gif")
            else:
                img = Image.open(__file__ + pathVarTemp + str(i+1) + "_training_mask.tif")
                img.save(__file__ + pathVar + str(i+1) + "_training_mask.gif")
        elif mode == 1:
            if i < 9:
                img = Image.open(__file__ + pathVarTemp + "0" + str(i+1) + "_test_mask.tif")
                img.save(__file__ + pathVar + "0" + str(i+1) + "_test_mask.gif")
            else:
                img = Image.open(__file__ + pathVarTemp + str(i+1) + "_test_mask.tif")
                img.save(__file__ + pathVar + str(i+1) + "_test_mask.gif")
        else:
            print("hier ist was schief gelaufen - Abspeichern von .gif Dateien ist nicht möglich, siehe mode. mode ist = " + str(mode))
            exit()


def rgb2grayscale(imgList): #1
    list = []
    for i in range(len(imgList)):
        list.append(cv2.cvtColor(imgList[i], cv2.COLOR_BGR2GRAY))

    return list


def histogramEqualization(imgList): #mit dem Ergebnis unzufrieden
    list = []
    for i in range(len(imgList)):
        list.append(cv2.equalizeHist(imgList[i]))

    return list


def clahe_contrast(imgList): #3
    list = []
    for i in range(len(imgList)):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        list.append(clahe.apply(imgList[i]))

    return list


def normalization(imgList):
    list = []
    for i in range(len(imgList)):
        #1. Variante
        zeroArray = np.zeros((1600,1200))
        list.append(cv2.normalize(imgList[i], zeroArray, 0 , 255, cv2.NORM_MINMAX))

    return list
#2. Variante
# y = (x - min) / (min - max)

def adjustGamma(imgList, gamma):
    list = []
    for i in range(len(imgList)):
        invGamma = 1.0 / gamma
        table = np.array( [ ((i/255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        list.append(cv2.LUT(imgList[i], table))
      
    return list

def blur(imgList):
    list = []
    for i in range(len(imgList)):
        list.append(cv2.blur(imgList[i], (3, 3)))
    
    return list

### SINUS COSINUS FILTER: remove noise from histogram ###
def scfilter(imgList, iterations, kernel):
    list = []
    for i in range(len(imgList)):
        
        #Sine‐cosine filter.
        #kernel can be tuple or single value.
        #Returns filtered image.
        
        for j in range(iterations):
            image = np.arctan2(
            scipy.ndimage.filters.uniform_filter(np.sin(imgList[i]), size=kernel),
            scipy.ndimage.filters.uniform_filter(np.cos(imgList[i]), size=kernel))
            list.append(image)
            # hier funktionert was nicht
            # https://stackoverflow.com/questions/36691020/how-to-remove-noise-from-a-histogram-equalized-image      
    return list

def generateImgageWithBackground(imgListGray, imgListRAW):
    result = []
    for i in range(len(imgListRAW)):
        # Apply Hough transform on the blurred image. 
        detected_circles = cv2.HoughCircles(imgListGray[i],  
                        cv2.HOUGH_GRADIENT, 1, 600, param1 = 50, 
                    param2 = 30, minRadius = 700, maxRadius = 800) 

        # Draw circles that are detected. 
        if detected_circles is not None: 
        
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
        
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 

                # Draw the circumference of the circle
                background = np.zeros_like(imgListRAW[i])
                mask = cv2.circle(background, (a, b), r, (255, 255, 255), thickness=cv2.FILLED)
                # Draw a small circle (of radius 1) to show the center. 
                #circle = cv2.circle(imgListRAW[i], (a, b), 1, (0, 0, 255), 3)
                result.append(cv2.bitwise_and(imgListRAW[i], mask))

    return result 
            

def generateMask(imgList):
    masks = []
    for i in range(len(imgList)):
        # Apply Hough transform on the blurred image. 
        detected_circles = cv2.HoughCircles(imgList[i],  
                        cv2.HOUGH_GRADIENT, 1, 600, param1 = 50, 
                    param2 = 30, minRadius = 700, maxRadius = 800) 

        # Draw circles that are detected. 
        if detected_circles is not None: 
        
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles))
        
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 

                # Draw the circumference of the circle
                background = np.zeros_like(imgList[i])
                masks.append(cv2.circle(background, (a, b), r, (255, 255, 255), thickness=cv2.FILLED))

    return masks 
