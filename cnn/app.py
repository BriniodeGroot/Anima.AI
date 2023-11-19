import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('cnn_model')

def predict_dog_breed(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make a prediction
    predictions = model.predict(img_array)

    # Assuming you have a list of class names corresponding to the indices
    class_names = ['breed1', 'breed2', 'breed3', ...]  # Replace with actual breed names

    # Find the index of the class with the highest probability
    predicted_class = class_names[np.argmax(predictions)]

    return predicted_class

# Example usage
img_path = 'path/to/your/dog/photo.jpg'  # Replace with the path to your image
breed = predict_dog_breed(img_path)
print(f"The predicted breed is: {breed}")
