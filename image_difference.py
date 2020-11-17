import cv2
from matplotlib import pyplot as plt
import numpy as np

#read images
image1 = cv2.imread("./image1.tif")
image2 = cv2.imread("./image2.tif")

#show images with matplotlib (opencv.imshow does not work in jupyter)
plt.imshow(image1)
plt.show()
plt.imshow(image2)
plt.show()

#calculate difference and show
difference = cv2.absdiff(image1, image2)
plt.imshow(difference)
plt.show()


#copy original image (this is the image we want to show the difference in)
colorchange = image1.copy()

#get dimensions and iterate over difference image
height, width, depth = difference.shape

for h in range(height):
    for w in range(width):
        if difference[h][w][0] != 0 or difference[h][w][1] != 0 or difference[h][w][2] != 0:
            colorchange[h][w][0] = 0
            colorchange[h][w][1] = 255
            colorchange[h][w][2] = 0

plt.imshow(colorchange)
plt.show()