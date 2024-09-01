import os
import librosa
import numpy as np

def load_audio_files(directory, file_extension='.wav'):
    """
    Load audio files from a directory.

    Parameters:
    - directory: str, path to the directory containing audio files.
    - file_extension: str, file extension to filter by (default is '.wav').

    Returns:
    - List of file paths to the audio files.
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                audio_files.append(os.path.join(root, file))
    return audio_files

def extract_mfcc(file_path=None, sr=16000, n_mfcc=13, y=None):
    if y is not None:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    elif file_path is not None:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    else:
        raise ValueError("Either file_path or y must be provided")
    return mfcc


# Example update to extract_mel_spectrogram
def extract_mel_spectrogram(file_path=None, sr=16000, y=None):
    if y is not None:
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    elif file_path is not None:
        y, sr = librosa.load(file_path, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    else:
        raise ValueError("Either file_path or y must be provided")
    return mel_spectrogram


def normalize_features(features):
    """
    Normalize features to have zero mean and unit variance (z-score normalization).

    Parameters:
    - features: numpy array, the features to normalize.

    Returns:
    - numpy array, normalized features.
    """
    return (features - np.mean(features)) / np.std(features)
