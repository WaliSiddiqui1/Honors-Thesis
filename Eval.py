import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load test dataset
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load trained models
resnet_model = tf.keras.models.load_model("resnet_cnn_model.h5")
custom_model = tf.keras.models.load_model("custom_cnn_model.h5")

# Make predictions
resnet_preds = np.argmax(resnet_model.predict(X_test), axis=1)
custom_preds = np.argmax(custom_model.predict(X_test), axis=1)

# Generate evaluation metrics
def evaluate_model(predictions, y_true, model_name):
    print(f"\nEvaluation Report for {model_name}:\n")
    print(classification_report(y_true, predictions, target_names=["Clear Sky", "Thin Cloud", "Opaque Cloud"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Clear Sky", "Thin Cloud", "Opaque Cloud"], yticklabels=["Clear Sky", "Thin Cloud", "Opaque Cloud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Evaluate both models
evaluate_model(resnet_preds, y_test, "ResNet-50 CNN")
evaluate_model(custom_preds, y_test, "Custom CNN")
