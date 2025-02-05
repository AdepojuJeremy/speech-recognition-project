import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_mfccs

# Load preprocessed data
def load_data(processed_audio_dir, transcriptions_dir):
    X = []
    y = []
    for filename in os.listdir(processed_audio_dir):
        if filename.endswith(".npy"):
            mfccs = np.load(os.path.join(processed_audio_dir, filename))
            X.append(mfccs)
            # Load corresponding transcription (dummy labels for now)
            y.append(0)  # Replace with actual labels
    return np.array(X), np.array(y)

# Build the model
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(64, activation='relu')),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Train the model
def train():
    X, y = load_data("data/processed_audio/", "data/transcriptions/")
    model = build_model((13, 100, 1), 29)  # Adjust input_shape and num_classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save("models/speech_recognition_model.h5")

if _name_ == "_main_":
    train()
