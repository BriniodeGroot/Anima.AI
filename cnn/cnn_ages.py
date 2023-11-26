import os
import cv2 #library for photos
import matplotlib.pyplot as plt
import numpy as np #mathematical library
from sklearn.model_selection import train_test_split #sklearn is a AI library
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential #AI library, most used for CNN
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
#os.system('cmd /c ".\keras_env\Scripts\activate"')


#we import the data from the different folders and store them according to their label

AGES = ['Young','Adult','Senior']

img_height = 375
img_width = 500

data = []
images = []
labels = []

data_dir = r'images_small'

def create_data():
    for i in range(len(AGES)):
        category = AGES[i]
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            if img_array is None:
                print(f"Failed to read image: {os.path.join(path, img)}")
                continue

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
  layers.Dense(3, activation = 'softmax')
])

model.summary()

opt = keras.optimizers.legacy.Adam(learning_rate=0.000001)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#######################################################

epochs= 5
batch_size = 32

#We train the model 

history = model.fit(
  X_train,
  y_train,  
  validation_data=(X_val, y_val),
  epochs=epochs,
  batch_size = batch_size,
  verbose = True,
  #callbacks = callback
)

######################################################

model.save('cnn_model_ages')

predicted = model.predict(X_test)

predictedlabels = []
for i in range(0,len(predicted)):
  label = np.argmax(predicted[i])
  predictedlabels.append(label)

#We check the accuracy metrics
print(accuracy_score(y_test, predictedlabels))
print(recall_score(y_test, predictedlabels, average='weighted'))
print(f1_score(y_test, predictedlabels, average='weighted'))

#condusion matrix
cm = confusion_matrix(y_test, predictedlabels)

#plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=AGES, yticklabels=AGES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
save_path = 'confusion_matrix/ages.png'
plt.savefig(save_path)
plt.show()


# show an example 
# print(predicted[1])
# print(y_test[1])
# print(AGES[y_test[1]])
# plt.imshow(X_test[1])
# plt.show()

# plot an graph of the accuracy in function of the epoch

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)