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

# load cnn model
model = load_model('cnn_model')

def preprocess_image(image):
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_reshape = cv2.resize(img_rgb, (img_width, img_height))

    #image_array = img_reshape / 255.0
    
    return img_reshape


@app.route('/')
def index():
    return render_template('index.html')

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

    predicted = model.predict(processed_image)
    label = np.argmax(predicted[0])

    print(predicted)
    print(label)
    print(CATEGORIES[label])

    dog_breed = CATEGORIES[label]

    return jsonify(breed=dog_breed)

if __name__ == '__main__':
    app.run(debug=True)
