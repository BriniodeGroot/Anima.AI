import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from keras_tuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Define categories and image dimensions
CATEGORIES = ['Maltese_dog', 'Eskimo_dog', 'Rottweiler']
img_height = 375
img_width = 500
data_dir = r'images'

# Function to create data
def create_data():
    data = []
    images = []
    labels = []
    for i in range(len(CATEGORIES)):
        category = CATEGORIES[i]
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_reshape = cv2.resize(img_rgb, (img_width, img_height))
                images.append(img_reshape)
                labels.append(i)
            except Exception as e:
                print(e)
    return np.array(images), np.array(labels)

# Create data
images, labels = create_data()

# Split data
X_train, X_testval, y_train, y_testval = train_test_split(images, labels, test_size=0.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, test_size=0.5, random_state=1)

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Model builder function
def build_model(hp):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(hp.Int('conv_1_units', min_value=16, max_value=256, step=32), 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp.Int('conv_2_units', min_value=16, max_value=256, step=32), 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(hp.Int('conv_3_units', min_value=16, max_value=256, step=32), 3, padding='valid', activation='relu'),
        layers.Flatten(),
        layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, default=0.25, step=0.05)),
        layers.Dense(hp.Int('dense_1_units', min_value=16, max_value=128, step=16), activation='relu'),
        layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, default=0.25, step=0.05)),
        layers.Dense(3, activation='softmax')  # Output layer for 3 categories
    ])

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Tuner configuration
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='random_search',
    project_name='cnn_model'
)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train generator with data augmentation
train_datagen = ImageDataGenerator(preprocessing_function=data_augmentation)
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# Start the search
tuner.search(train_generator, epochs=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model
best_model.save('cnn_model')

# Model evaluation on test data
predicted = best_model.predict(X_test)
predicted_labels = np.argmax(predicted, axis=1)

# Calculate metrics
print("Accuracy:", accuracy_score(y_test, predicted_labels))
print("Recall:", recall_score(y_test, predicted_labels, average='weighted'))
print("F1 Score:", f1_score(y_test, predicted_labels, average='weighted'))

cm = confusion_matrix(y_test, predicted_labels)

# # plot confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion matrix")
# plt.show()

# # show an example

# print(predicted[1])
# print(y_test[1])
# print(CATEGORIES[y_test[1]])
# plt.imshow(X_test[1])
# plt.show()

# plot an graph of the accuracy in function of the epoch

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show

# Model evaluation
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)