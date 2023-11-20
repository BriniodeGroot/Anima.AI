from flask import Flask, request, jsonify, render_template # flask is a web app framework written in python
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)

img_height = 375
img_width = 500

CATEGORIES = ['Maltese dog','Chihuahua','Japanese spaniel']
AGES = ['young','adult','senior']

# load cnn model
model = load_model('cnn_model')
model_ages = load_model('cnn_model_ages')

def preprocess_image(image):
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_reshape = cv2.resize(img_rgb, (img_width, img_height))

    #image_array = img_reshape / 255.0
    
    return img_reshape


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/help.html')
def outcome():
    return render_template('help.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file:

        image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)

    # Adding the batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)  # Shape becomes (1, 375, 500, 3)

    ##############################################################
    #prediction of breed

    predicted = model.predict(processed_image)
    label = np.argmax(predicted[0])

    print(predicted)
    print(label)
    print(CATEGORIES[label])

    dog_breed = CATEGORIES[label]

    ##############################################################
    #prediction of age category

    predicted_ages = model_ages.predict(processed_image)
    label_ages = np.argmax(predicted_ages[0])

    print(predicted_ages)
    print(label_ages)
    print(AGES[label_ages])

    dog_age = AGES[label_ages]

    #############################################################

    return jsonify(breed=dog_breed, age=dog_age)

if __name__ == '__main__':
    app.run(debug=True)
