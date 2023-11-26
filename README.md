# Anima.AI

This web application allows users to upload images or audio files related to dogs. The core functionalities include:

1. **Image Upload**: Users can upload an image of a dog. The application's AI, powered by a self-trained Convolutional Neural Network (CNN), will analyze the image to recognize the dog's breed and estimate its age category.

1. **Audio Upload**: Users also have the option to upload an audio file containing barking sounds. The AI will process these sounds to determine the dog's current situation or emotional state.

## Setting Up the Application:

To effectively run and interact with the application, follow these steps:

1. **Create a Python Environment:**

* Set up a dedicated Python environment for running the application. This ensures that all necessary libraries and dependencies are isolated and managed correctly.

2. **Install Required Libraries:**

* Import all required libraries within this environment. The specific libraries and their versions will depend on the application's requirements. (It's helpful to provide a requirements.txt file for this purpose.)

3. **Model Training:**

* Download all the datasets and adjust the path to your own correct directory.
* Before launching the application, execute the training scripts for the CNN models. This step is crucial as it prepares the models to accurately analyze and classify the data (images and audio) uploaded by users.
* Ensure that the trained models are saved on your local system for the application to use.

or use our pretrained models (expires on 27-12-2023): https://filesender.belnet.be/?s=download&token=6314f5c6-898f-448a-ad59-a31978994302

4. **Running the Flask Application:**

* With the models trained and saved, you can now run the web application using Flask. Start the Flask server to host the application locally.

5. **Interaction and Exploration:**

* Once the application is running, explore its features by uploading dog images and audio files to test the AI's capabilities.

## Enjoy the Experience:

This application showcases the power of AI in recognizing and interpreting complex patterns in visual and auditory data. Enjoy the experience and the insights it provides about man's best friend!
