import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from config import TEST_DATA_DIR, MODEL_DIR

def load_test_data():
    data = np.load(os.path.join(TEST_DATA_DIR, 'test_data.npz'))
    return data['X_test'], data['y_test']

def evaluate_model():

    model = load_model('best_model.keras')
    
    X_test, y_test = load_test_data()
    
    predictions = (model.predict(X_test) > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
