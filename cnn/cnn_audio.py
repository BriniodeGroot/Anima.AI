from gc import callbacks
import os
import cv2 #library for photos
import matplotlib.pyplot as plt
import numpy as np #mathematical library
from sklearn.model_selection import train_test_split #sklearn is a AI library
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential #AI library, most used for CNN
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#os.system('cmd /c ".\keras_env\Scripts\activate"')


#we import the data from the different folders and store them according to their label

#CATEGORIES_EXT = ['Border_collie', 'Chihuahua', 'Maltese_dog', 'Eskimo_dog', 'French_bulldog', 'Golden_retriever', 'Irish_terrier', 'Norwich_terrier',  'Norfolk_terrier', 'Rottweiler']
CATEGORIES = ['Bark','Bow-wow','Growling','Howl','Whimper','Yip']

img_height = 300
img_width = 300

data = []
images = []
labels = []

data_dir = r'C:\school\3de_jaar\ai_applications\AnimaI\cnn\dog_sounds'

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

#visualisation 

print(data)
#plt.imshow(images[-1]) #last image
#plt.show()

#################################################

#We make train, validate and test data
#Traindata is the the data we use for the training
#Validata is the validation data for the train data
#Test data is the data to test our model


X_train, X_testval, y_train, y_testval  = train_test_split(images, labels, test_size=0.2, random_state=1)
X_val, X_test, y_val, y_test  = train_test_split(X_testval, y_testval, test_size=0.5, random_state=1)
X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

print(y_val)

#################################################

#We build a CNN with SELU activation functions.

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(32, 3, padding='valid', activation='relu'), 
  layers.MaxPooling2D(), #2x2 kernel size
  layers.Conv2D(64, 3, padding='valid', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='valid', activation='relu'),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(16, activation='relu'),
  layers.Dense(6, activation = 'softmax')
])

model.summary()

#opt = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#######################################################

epochs= 10
batch_size = 32

#We train the model 

history = model.fit(
  X_train,
  y_train,  
  validation_data=(X_val, y_val),
  epochs=epochs,
  batch_size = batch_size,
  verbose = True,
  callbacks = callbacks
)

######################################################

model.save('cnn_model_sounds_dogs')

predicted = model.predict(X_test)

predictedlabels = []
for i in range(0,len(predicted)):
  label = np.argmax(predicted[i])
  predictedlabels.append(label)

#We check the accuracy metrics
print(accuracy_score(y_test, predictedlabels))
print(recall_score(y_test, predictedlabels, average='weighted'))
print(f1_score(y_test, predictedlabels, average='weighted'))

# show an example 

print(predicted[1])
print(y_test[1])
print(CATEGORIES[y_test[1]])
plt.imshow(X_test[1])
plt.show()