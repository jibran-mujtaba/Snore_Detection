import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from utils import extract_mfcc, extract_mel_spectrogram, normalize_features
from config import TRAIN_DATA_DIR, TEST_DATA_DIR, SR, N_MFCC, N_MELS, HOP_LENGTH, AUGMENTATION, TIME_STEPS

def preprocess_file(file, label):
    mfcc_features = extract_mfcc(file, SR, N_MFCC)
    mel_features = extract_mel_spectrogram(file, SR)

    if mfcc_features is None or mel_features is None:
        print(f"Failed to extract features from file: {file}")
        return None, None

    combined_features = np.concatenate((mfcc_features, mel_features), axis=0)
    normalized_features = normalize_features(combined_features)

    # Ensure correct shape by padding or truncating
    if normalized_features.shape[1] != TIME_STEPS:
        if normalized_features.shape[1] < TIME_STEPS:
            pad_width = TIME_STEPS - normalized_features.shape[1]
            normalized_features = np.pad(normalized_features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            normalized_features = normalized_features[:, :TIME_STEPS]

    return normalized_features, label

def preprocess_data():
    features = []
    labels = []

    for label, subdir in enumerate(['snoring', 'non-snoring']):
        subdir_path = os.path.join(TRAIN_DATA_DIR, subdir)
        for file in os.listdir(subdir_path):
            if file.endswith('.wav'):
                file_path = os.path.join(subdir_path, file)
                feature, label = preprocess_file(file_path, label)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(label)

                if AUGMENTATION:
                    y, _ = librosa.load(file_path, sr=SR)
                    
                    augmentations = [
                        librosa.effects.time_stretch(y, rate=0.8), 
                        y + 0.005 * np.random.randn(len(y)), 
                        librosa.effects.pitch_shift(y, sr=SR, n_steps=2)
                    ]
                    
                    for augmented_y in augmentations:
                        mfcc_aug = extract_mfcc(file_path=None, sr=SR, n_mfcc=N_MFCC, y=augmented_y)
                        mel_aug = extract_mel_spectrogram(file_path=None, sr=SR, y=augmented_y)
                        combined_aug = np.concatenate((mfcc_aug, mel_aug), axis=0)
                        normalized_aug = normalize_features(combined_aug)
            
                        if normalized_aug.shape[1] != TIME_STEPS:
                            if normalized_aug.shape[1] < TIME_STEPS:
                                pad_width = TIME_STEPS - normalized_aug.shape[1]
                                normalized_aug = np.pad(normalized_aug, ((0, 0), (0, pad_width)), mode='constant')
                            else:
                                normalized_aug = normalized_aug[:, :TIME_STEPS]
                
                        features.append(normalized_aug)
                        labels.append(label)
                        

    features = np.array(features)
    labels = np.array(labels)


    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    

    np.savez_compressed(os.path.join(TRAIN_DATA_DIR, 'train_data.npz'), X_train=X_train, y_train=y_train)
    np.savez_compressed(os.path.join(TRAIN_DATA_DIR, 'val_data.npz'), X_val=X_val, y_val=y_val)
    np.savez_compressed(os.path.join(TEST_DATA_DIR, 'test_data.npz'), X_test=X_test, y_test=y_test)
    
    print("Data preprocessing and splitting completed successfully.")

if __name__ == "__main__":
    preprocess_data()