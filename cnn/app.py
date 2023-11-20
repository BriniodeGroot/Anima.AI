from flask import Flask, request, jsonify, render_template # flask is a web app framework written in python
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

# load cnn model
model = load_model('cnn_model')

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

        predicted = model.predict(X_test)
        return jsonify(breed=prediction)

if __name__ == '__main__':
    app.run(debug=True)
