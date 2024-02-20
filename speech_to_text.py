import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
import tarfile
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO

def download_and_unzip(url, extract_to="Datasets", chunk_size=1024*1024):
    http_response = urlopen(url)

    data = b""
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    tarFile = tarfile.open(fileobj=BytesIO(data), mode="r|bz2")
    tarFile.extractall(path=extract_to)
    tarFile.close()

def preprocess_data(data_path, duration=3, sr=16000):
    X = []
    y = []
    label_encoder = LabelEncoder()
    
    for folder in os.listdir(data_path):
        label = folder
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                audio, sr = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                X.append(mfccs)
                y.append(label)
            except Exception as e:
                print(f"Error encountered while processing {file_path}: {e}")
    
    X = np.array(X)
    y = label_encoder.fit_transform(y)
    
    return X, y, label_encoder

dataset_path = os.path.join("Datasets", "LJSpeech-1.1")
if not os.path.exists(dataset_path):
    print("Downloading and unzipping dataset...")
    download_and_unzip("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", extract_to="Datasets")
    print("Dataset downloaded and unzipped successfully!")

X, y, label_encoder = preprocess_data(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
