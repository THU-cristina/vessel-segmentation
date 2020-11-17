import cv2
from matplotlib import pyplot as plt
import numpy as np

#bild zur segmentierung einlesen
img = cv2.imread("test_img_non_plus.png")

#nur grünen channel des bildes nehmen, da beste kontraste
img_green = img[:,:,1]

#gaus blur verwenden um rauschen zu verringern
img_blur = cv2.GaussianBlur(img_green, (3, 3), 0)
cv2.imshow("blur", img_blur)

#adaptiven threshold anwenden (kein threshold über ganzes bild, sondern je nach belichtung in einzelnen bereichen des bildes)
img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 1.6)
cv2.imshow("thresh", img_thresh)

#bild invertieren (schwarz -> weiß, weiß -> schwarz)
img_inverted = cv2.bitwise_not(img_thresh)
cv2.imshow("inverted", img_inverted)

#opening verwenden: dadurch werden kleine objekte gelöscht
img_opened = cv2.morphologyEx(img_inverted, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
cv2.imshow("opening", img_opened)

#objekte im bild erkennen
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_opened, connectivity=4)
sizes = stats[:, -1]


#min size ist die anzahl der pixel die ein objekt haben muss, um "erkannt" zu werden
#dh nur objekte mit dieser mindestgröße werden später im ergebnis auftauchen
min_size = 100 

#liste in die die zutreffenden objekte gespeichert wird
objects_list = []

#über liste mit erkannten objekten iterieren
#wenn mindestgröße passt wird objekt in die liste der "erkannten" objekte gespeichert (also in objects_list)
for i in range(2, nb_components):
    if sizes[i] > min_size:
        objects_list.append(i)

#neues leeres bild mit selben ausmaßen wie das originalbild erstellen
img_objects = np.zeros(output.shape)

#objekte aus object_list in leeres bild einfügen
for i in range(len(objects_list)):
    img_objects[output == objects_list[i]] = 255

cv2.imshow("objects", img_objects)

#closing verwenden: dadurch werden löcher geschlossen; die erkannten objekte werden etwas besser aneinander"geklebt"
img_closed = cv2.morphologyEx(img_objects, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
cv2.imshow("closing", img_closed)
cv2.waitKey()

cv2.imwrite("v1_automatic_segmentation_test_img_non_plus" + ".png", img_closed)
