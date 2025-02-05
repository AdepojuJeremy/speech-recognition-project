import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)  # Normalize
    return mfccs

def save_mfcc(mfccs, save_path):
    """Save MFCC features to a file."""
    np.save(save_path, mfccs)
