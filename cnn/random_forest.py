import os
import cv2 #library for photos
import matplotlib.pyplot as plt
import numpy as np #mathematical library
from sklearn.model_selection import train_test_split #sklearn is a AI library
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential #AI library, most used for CNN
import joblib
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


CATEGORIES = ['Maltese_dog','Chihuahua','Japanese_spaniel']

img_height = 375
img_width = 500

data = []
images = []
labels = []

data_dir = r'images'

def create_data():
    for i in range(len(CATEGORIES)):
        category = CATEGORIES[i]
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) #if you use an image, it is in the bgr format, so we need to convert it to rgb
            img_reshape = cv2.resize(img_rgb, (img_width, img_height))
            data.append([img_reshape, i])
            images.append(img_reshape)
            labels.append(i)
    return data, images, labels
    
data, images, labels = create_data()

####################################################

#We prepare the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(images, labels, test_size=0.2, random_state=1)

#We convert the photos so that we can use that in the random forest

def makematrix(data):
        numImages = len(data)
        sz = (img_width, img_height, 3)

        datamatrix = np.zeros((numImages, sz[0] * sz[1]), dtype=np.float32)

        for i in range(0, numImages):
            data1 = np.float32(data[i])                       # This conversion is needed for conversion to grayscale
            image1 = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)  # We convert to gray images
            image = image1.flatten()                          # We convert the 100x100 gray image to one dimension 10000
            image = image/255
            datamatrix[i,:] = image

        return datamatrix
        
X_train_flat = makematrix(X_train)
X_test_flat = makematrix(X_test)

###############################################

#We train a RandomForest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=20, n_estimators=100)
clf.fit(X_train_flat,y_train)
RandomForestClassifier(...)

# save
joblib.dump(clf, "./random_forest.joblib")

###############################################

#now we check the accuracy

predicted = clf.predict(X_test_flat)
print(accuracy_score(y_test, predicted))
print(recall_score(y_test, predicted, average='weighted'))
print(f1_score(y_test, predicted, average='weighted'))