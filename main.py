import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extract_mfcc(audio_file):
    audio, sr = librosa.load(audio_file)
    mfcc = np.array(librosa.feature.mfcc(y=audio, sr=sr))
    return mfcc


def load_dataset(dataset_path):
    X = []
    y = []

    classes = os.listdir(dataset_path)
    print(classes)
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        print(class_path)
        if os.path.isdir(class_path):
            audio_files = os.listdir(class_path)
            print(audio_files)
            for audio_file in audio_files:
                audio_path = os.path.join(class_path, audio_file)
                print(audio_path)
                mfcc = extract_mfcc(audio_path)
                X.append(mfcc)
                y.append(class_name)

    return X, y


def predict_audio(audio_file):
    mfcc = extract_mfcc(audio_file)
    mfcc = mfcc.reshape((1, -1))

    # Predict the class label
    label = svm.predict(mfcc)[0]

    if label == "music":
        print("The provided audio file contains music.")
    elif label == "speech":
        print("The provided audio file contains speech.")
    else:
        print("Unable to classify the provided audio file.")


# Define the path to the dataset
dataset_path = 'dataset'

# Load the dataset
X, y = load_dataset(dataset_path)

# Convert the MFCC data to a 2D array
X = np.array(X)
X = X.reshape((X.shape[0], -1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm = SVC(C=1000)
svm.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example usage: classify a provided audio file
audio_file = 'dataset/speech/acomic.wav'
predict_audio(audio_file)
