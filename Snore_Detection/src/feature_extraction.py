import os
import numpy as np
import librosa
from utils import load_audio_files
from config import (
    TRAIN_DATA_DIR, SR, N_MFCC, N_MELS, HOP_LENGTH, BATCH_SIZE, 
    AUGMENTATION, TIME_STRETCH_RATES, PITCH_SHIFT_STEPS, NOISE_FACTOR, NORMALIZE
)

def augment_audio(y, sr):
    """Apply augmentation techniques like time-stretching, pitch-shifting, and adding noise."""
    if AUGMENTATION:
        if TIME_STRETCH_RATES:
            y = librosa.effects.time_stretch(y, np.random.choice(TIME_STRETCH_RATES))
        if PITCH_SHIFT_STEPS:
            y = librosa.effects.pitch_shift(y, sr, np.random.choice(PITCH_SHIFT_STEPS))
        if NOISE_FACTOR:
            y = y + NOISE_FACTOR * np.random.randn(len(y))
    return y

def extract_features(file_path, sr, n_mfcc, n_mels, hop_length):
    y, _ = librosa.load(file_path, sr=sr)

    y = augment_audio(y, sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if NORMALIZE:
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)

    combined_features = np.hstack((np.mean(mfcc.T, axis=0), np.mean(mel_spectrogram.T, axis=0)))

    if combined_features.shape[0] != TIME_STEPS:
        if combined_features.shape[0] < TIME_STEPS:
            pad_width = TIME_STEPS - combined_features.shape[0]
            combined_features = np.pad(combined_features, (0, pad_width), mode='constant')
        else:
            combined_features = combined_features[:TIME_STEPS]

    return combined_features

def extract_features_in_batches(files, batch_size, sr, n_mfcc, n_mels, hop_length):
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        features = np.array([
            extract_features(file, sr, n_mfcc, n_mels, hop_length) 
            for file in batch_files
        ])
        yield features

def process_large_dataset():
    files = load_audio_files(TRAIN_DATA_DIR)
    labels = [1 if 'snore' in os.path.basename(file).lower() else 0 for file in files]

    for batch_features in extract_features_in_batches(files, BATCH_SIZE, SR, N_MFCC, N_MELS, HOP_LENGTH):
        # Process or save the batch features as needed
        pass

if __name__ == "__main__":
    process_large_dataset()
