from flask import Flask, request, jsonify, render_template # flask is a web app framework written in python
import numpy as np
from PIL import Image
import io
import os
import wave
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import librosa
import librosa.display

app = Flask(__name__)

img_height = 375
img_width = 500

img_height_sound = 300
img_width_sound = 300

CATEGORIES = ['Maltese dog','Chihuahua','Japanese spaniel']
AGES = ['young','adult','senior']
SOUNDS = ['Bark','Bow-wow','Growling','Howl','Whimper','Yip']

# load cnn model
model = load_model('cnn_model')
model_ages = load_model('cnn_model_ages')
model_sound = load_model('cnn_model_sound_dogs')

def preprocess_image(image):
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_reshape = cv2.resize(img_rgb, (img_width, img_height))

    #image_array = img_reshape / 255.0
    
    return img_reshape

def preprocess_audio(audio):
    y, sr = librosa.load(audio)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_S = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')  # no axis
    save_path = './predict_images/test.png'
    plt.savefig(save_path)
    plt.close()
    img_audio = cv2.imread(save_path)
    img_audio_rgb = cv2.cvtColor(img_audio, cv2.COLOR_BGR2RGB)
    img_audio_reshape = cv2.resize(img_audio_rgb, (img_width_sound, img_height_sound))
    plt.imshow(img_audio_reshape)
    # plt.show()
    os.remove(save_path)
    # plt.close()

    return img_audio_reshape


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/help.html')
def help():
    return render_template('help.html')

@app.route('/dog_breed.html')
def breed():
    return render_template('dog_breed.html')

@app.route('/dog_sound.html')
def sound():
    return render_template('dog_sound.html')

@app.route('/predict_breed', methods=['POST'])
def predict_breed():
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

@app.route('/predict_sound', methods=['POST'])
def predict_sound():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    # if file:
    #     audio = io.BytesIO(file.read())
    processed_audio = preprocess_audio(file)

    # Adding the batch dimension
    processed_audio = np.expand_dims(processed_audio, axis=0)  # Shape becomes (1, 375, 500, 3)

    ##############################################################
    #prediction of breed

    predicted = model_sound.predict(processed_audio)
    label = np.argmax(predicted[0])

    print(predicted)
    print(label)
    print(SOUNDS[label])

    dog_sound = SOUNDS[label]

    return jsonify(sound=dog_sound)

if __name__ == '__main__':
    app.run(debug=True)
