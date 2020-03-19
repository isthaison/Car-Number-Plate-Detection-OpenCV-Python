import shutil
import numpy as np
import cv2
import imutils
from datetime import datetime
import pickle
from PIL import Image
from imgsearch.searcher import Searcher
import os
import glob
import pytesseract
from skimage.transform import resize
from imgsearch.rgbhistogram import RGBHistogram


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    print("width: {}, height: {},angle: {} ".format(width, height, angle))

    M = cv2.getRotationMatrix2D(center, angle, 1)

    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def segmentation(searcher, img0):
    plates = ""
    img = img0.copy()

    files = glob.glob('tmp/*')
    for f in files:
        os.remove(f)
    files = glob.glob('valid/*')
    for f in files:
        os.remove(f)
    # grayscale

    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binarize
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # find contours
    ctrs, hier = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop over our contours to find the best possible approximate contour of number plate
    now = datetime.now()
    print("now =", now)

    dt_string = now.strftime("%Y%m%d%H%M%S")


    

    ctrs.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))
    platesmatrix= []
    for i, ctr in enumerate(ctrs):
        # Get bounding box

        x, y, w, h = cv2.boundingRect(ctr)
        if (h > 20 and w > 4 and h < height * 0.9 ):
            # Getting ROI
            roi = gray[y:y+h, x:x+w]

            # show ROI
            newpath = 'tmp/'+dt_string+str(i)+'.png'



            roi = cv2.resize(roi, (40, 60), interpolation=cv2.INTER_AREA)
            # roi = cv2.resize(roi , width)
            cv2.imwrite(newpath, roi)
            # cv2.imshow('charachter'+str(i), roi)
            cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 1)

            desc = RGBHistogram([8, 8, 8])
            queryFeatures = desc.describe(roi)

            # load the index perform the search
            results = searcher.search(queryFeatures)
            (score, imageName) = results[0]
            print ("\t%d. %s : %.3f" % (1, imageName, score))
            if imageName[0] == "+":
                plates = plates +  "."
            else:
                plates =plates + imageName[0]
         
          
            (score, imageName) = results[1]
            print ("\t%d. %s : %.3f" % (1, imageName, score))
    print ("plates: " +plates)
    cv2.imshow('marked areas', img)
    print(platesmatrix)
    cv2.waitKey(0)
    return


def detect_chart(img, searcher):
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("valid/"+dt_string + "_" + str(count) + '.png', img)

    segmentation(searcher, img)
    return img


# Read the image file
image = cv2.imread('images/89931926_206247237324541_4638874939527528448_n.jpg')

# Resize the image - change width to 500
image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_AREA)


# Display the original image
# cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("1 - Grayscale Conversion", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
# cv2.imshow("2 - Bilateral Filter", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
# cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges
(cnts, _) = cv2.findContours(edged.copy(),
                             cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # pyyhon3

# cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #python2
cv2.imshow("22",gray)
cv2.waitKey(0)  # Wait for user input before closing the images displayed

# sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt = None  # we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate

count = 0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    rect = cv2.minAreaRect(c)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 1)

    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx  # This is our approx Number Plate Contour

        # Extract subregion
        img_crop, img_rot = crop_rect(image, rect)
        print("Loading model")
        filename = './index.cpickle'
        model = pickle.load(open(filename, 'rb'))
        searcher = Searcher(model)
        print('Model loaded. Predicting characters of number plate')
        # cv2.imshow("Final Image With Number Plate Detected", img_crop)
        img_crop = detect_chart(img_crop, searcher)
        img_crop = detect_chart(img_crop, searcher)
        img_crop = detect_chart(img_crop, searcher)
        img_crop = detect_chart(img_crop, searcher)


        break


# Drawing the selected contour on the original image
print(NumberPlateCnt)


cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)


cv2.imshow("Final Image With Number Plate Detected", image)


cv2.waitKey(0)  # Wait for user input before closing the images displayed
cv2.destroyAllWindows()
