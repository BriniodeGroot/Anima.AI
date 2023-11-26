import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np #mathematical library

source_directory = 'sounds'
target_directory = 'sounds_dogs_img'

# function to convert audio to spectrogram

def audio_to_spectrogram(file_path, save_path):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_S = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')  # no axis
    plt.savefig(save_path)
    plt.close()

# Create target directory structure
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

for category in ['bark', 'bow-wow', 'growling', 'howl', 'whimper', 'yip']:
    category_path = os.path.join(source_directory, category)
    target_category_path = os.path.join(target_directory, category)
    
    if not os.path.exists(target_category_path):
        os.makedirs(target_category_path)

    for filename in os.listdir(category_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(category_path, filename)
            target_path = os.path.join(target_category_path, filename.replace('.wav', '.png'))
            audio_to_spectrogram(file_path, target_path)

print("Conversion complete.")