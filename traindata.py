import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
import cv2
from skimage.filters import threshold_otsu
from imgsearch.rgbhistogram import RGBHistogram
import glob

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z','+' ,'-'
        ]

# initialize the index dictionary to store our our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
index = {}

# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = RGBHistogram([8, 8, 8])






def read_training_data(training_directory):
    for each_letter in letters:
        c = 0
        for imagePath in glob.glob(training_directory+each_letter + "/*.png"):
            print(imagePath)
            

            # extract our unique image ID (i.e. the filename)
            k = each_letter + '_' + str(c)
            # print( str(each))
            print(k)
            # load the image, describe it using our RGB histogram
            # descriptor, and update the index
            image = cv2.imread(imagePath)
           
            try:
                features = desc.describe(image)
                index[k] = features
                # print k, imagePath
            except:
                print("something")
            c=c+1


     
           
    return

# current_dir = os.path.dirname(os.path.realpath(__file__))
#
# training_dataset_dir = os.path.join(current_dir, 'train')
print('reading data')
training_dataset_dir = 'train20X20/'
read_training_data(training_dataset_dir)
print('reading data completed')





# we are now done indexing our image -- now we can write our
# index to disk
f = open("index.cpickle" ,"wb")
f.write(pickle.dumps(index))
f.close()

# show how many images we indexed
print ("done...indexed %d images" % (len(index)))