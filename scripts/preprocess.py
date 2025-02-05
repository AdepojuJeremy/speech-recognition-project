import os
from utils import extract_mfcc, save_mfcc

# Paths
raw_audio_dir = "data/raw_audio/"
processed_audio_dir = "data/processed_audio/"

# Create processed_audio directory if it doesn't exist
os.makedirs(processed_audio_dir, exist_ok=True)

# Preprocess all audio files
for filename in os.listdir(raw_audio_dir):
    if filename.endswith(".wav"):
        audio_path = os.path.join(raw_audio_dir, filename)
        mfccs = extract_mfcc(audio_path)
        save_path = os.path.join(processed_audio_dir, filename.replace(".wav", ".npy"))
        save_mfcc(mfccs, save_path)
        print(f"Processed {filename}")
