from flask import Flask, request, jsonify, render_template # flask is a web app framework written in python
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import cv2

app = Flask(__name__)

img_height = 375
img_width = 500

CATEGORIES = ['Maltese_dog', 'Eskimo_dog','Golden_retriever']

# load cnn model
model = load_model('cnn_model')

def preprocess_image(image):
    # Resize the image
    image = image.resize((500, 375))  # Width x Height

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Normalize or scale the pixel values if necessary
    # For example, if your model expects pixel values in [0, 1]
    image_array = image_array / 255.0

    return image_array


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
    predictedlabels = []
    for i in range(0,len(predicted)):
        label = np.argmax(predicted[i])
        predictedlabels.append(label)
    print(predicted)
    print(predictedlabels)
    print(CATEGORIES[predictedlabels[0]])

    dog_breed = CATEGORIES[predictedlabels[0]]

    return jsonify(breed=dog_breed)

if __name__ == '__main__':
    app.run(debug=True)
