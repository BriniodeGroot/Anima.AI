import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from kerastuner.tuners import RandomSearch

# os.system('cmd /c ".\keras_env\Scripts\activate"')


# we import the data from the different folders and store them according to their label

# CATEGORIES_EXT = ['Border_collie', 'Chihuahua', 'Maltese_dog', 'Eskimo_dog', 'French_bulldog', 'Golden_retriever', 'Irish_terrier', 'Norwich_terrier',  'Norfolk_terrier', 'Rottweiler']
CATEGORIES = ['Maltese_dog', 'Eskimo_dog', 'Rottweiler']

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
            img_array = cv2.imread(os.path.join(path, img))
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_reshape = cv2.resize(img_rgb, (img_width, img_height))
            data.append([img_reshape, i])
            images.append(img_reshape)
            labels.append(i)
    return data, images, labels


data, images, labels = create_data()

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

best_model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(64, 3, padding='valid', activation='relu'),
  layers.MaxPooling2D(), #2x2 kernel size
  layers.Conv2D(256, 3, padding='valid', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(160, 3, padding='valid', activation='relu'),
  layers.Flatten(),
  layers.Dropout(0.5),
  layers.Dense(128, activation='relu'),
  layers.Dense(96, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(3, activation = 'softmax')
])

best_model.summary()

opt = keras.optimizers.legacy.Adam(learning_rate=0.00001)
best_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#######################################################

epochs= 5
batch_size = 32

#We train the model

history = best_model.fit(
  X_train,
  y_train,
  validation_data=(X_val, y_val),
  epochs=epochs,
  batch_size = batch_size,
  verbose = True,
  #callbacks = callback
)

######################################################

# best_model.save('cnn_model_ages')
predicted = best_model.predict(X_test)

predictedlabels = []
for i in range(0, len(predicted)):
    label = np.argmax(predicted[i])
    predictedlabels.append(label)

print(accuracy_score(y_test, predictedlabels))
print(recall_score(y_test, predictedlabels, average='weighted'))
print(f1_score(y_test, predictedlabels, average='weighted'))

cm = confusion_matrix(y_test, predictedlabels)

# plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
save_path = 'confusion_matrix/breed.png'
plt.savefig(save_path)
plt.show()

# # show an example

print(predicted[1])
print(y_test[1])
print(CATEGORIES[y_test[1]])
plt.imshow(X_test[1])
plt.show()

# plot an graph of the accuracy in function of the epoch

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
