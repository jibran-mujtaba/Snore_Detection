import os
import numpy as np
from config import TRAIN_DATA_DIR, EPOCHS, BATCH_SIZE, MODEL_DIR
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from model import create_model  

def load_data():
    train_data = np.load(os.path.join(TRAIN_DATA_DIR, 'train_data.npz'))
    val_data = np.load(os.path.join(TRAIN_DATA_DIR, 'val_data.npz'))

    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_val = val_data['X_val']
    y_val = val_data['y_val']
    
    return X_train, y_train, X_val, y_val

def train_model():
    X_train, y_train, X_val, y_val = load_data()
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original X_val shape: {X_val.shape}")
    

    time_steps = X_train.shape[1]  # 141
    features = X_train.shape[2]  # 1280
    

    num_samples_train = X_train.shape[0]
    num_samples_val = X_val.shape[0]
    total_elements_train = X_train.size
    total_elements_val = X_val.size

    expected_shape_train = (num_samples_train, time_steps, features)
    expected_shape_val = (num_samples_val, time_steps, features)

    print(f"Total elements in X_train: {total_elements_train}")
    print(f"Total elements in X_val: {total_elements_val}")
    print(f"Expected shape for X_train: {expected_shape_train}")
    print(f"Expected shape for X_val: {expected_shape_val}")
    

    if total_elements_train != np.prod(expected_shape_train):
        print(f"Error: Cannot reshape X_train to {expected_shape_train} because the total number of elements does not match.")
    if total_elements_val != np.prod(expected_shape_val):
        print(f"Error: Cannot reshape X_val to {expected_shape_val} because the total number of elements does not match.")
    

    X_train = X_train.reshape(expected_shape_train)
    X_val = X_val.reshape(expected_shape_val)
    
    model = create_model(input_shape=(time_steps, features))
    

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'snore_detection_model.weights.h5'), 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min',
        save_weights_only=True 
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)


    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    )

    model.save(os.path.join(MODEL_DIR, 'final_snore_detection_model.h5'), save_format='h5')
    
if __name__ == "__main__":
    train_model()
