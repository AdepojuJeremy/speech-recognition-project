import numpy as np
import tensorflow as tf
from utils import extract_mfcc

def transcribe(audio_path, model_path="models/speech_recognition_model.h5"):
    model = tf.keras.models.load_model(model_path)
    mfccs = extract_mfcc(audio_path)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    prediction = model.predict(mfccs)
    # Convert prediction to text (dummy implementation)
    return " ".join([str(np.argmax(p)) for p in prediction[0]])

if _name_ == "_main_":
    audio_path = "data/raw_audio/test.wav"
    transcription = transcribe(audio_path)
    print(f"Transcription: {transcription}")
