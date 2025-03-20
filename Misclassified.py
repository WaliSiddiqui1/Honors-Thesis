import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pandas as pd

# Load test dataset
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load trained models
resnet_model = tf.keras.models.load_model("resnet_cnn_model.h5")
custom_model = tf.keras.models.load_model("custom_cnn_model.h5")

# Make predictions
resnet_preds = np.argmax(resnet_model.predict(X_test), axis=1)
custom_preds = np.argmax(custom_model.predict(X_test), axis=1)

# Identify misclassified samples
def get_misclassified_indices(predictions, y_true):
    return np.where(predictions != y_true)[0]

resnet_misclassified = get_misclassified_indices(resnet_preds, y_test)
custom_misclassified = get_misclassified_indices(custom_preds, y_test)

# Function to log misclassified samples
def log_misclassified_samples(y_true, y_pred, misclassified_indices, model_name):
    log_data = []
    class_labels = ["Clear Sky", "Thin Cloud", "Opaque Cloud"]
    
    for idx in misclassified_indices:
        log_data.append({
            "Model": model_name,
            "Index": idx,
            "True Label": class_labels[y_true[idx]],
            "Predicted Label": class_labels[y_pred[idx]]
        })
    
    return pd.DataFrame(log_data)

# Create logs for misclassifications
resnet_log = log_misclassified_samples(y_test, resnet_preds, resnet_misclassified, "ResNet-50 CNN")
custom_log = log_misclassified_samples(y_test, custom_preds, custom_misclassified, "Custom CNN")

# Save logs to CSV
resnet_log.to_csv("resnet_misclassified_log.csv", index=False)
custom_log.to_csv("custom_misclassified_log.csv", index=False)

print("Misclassification logs saved as CSV files!")

# Function to analyze misclassification patterns
def analyze_misclassification_patterns():
    resnet_counts = resnet_log.groupby(["True Label", "Predicted Label"]).size().unstack().fillna(0)
    custom_counts = custom_log.groupby(["True Label", "Predicted Label"]).size().unstack().fillna(0)
    
    print("\nResNet-50 Misclassification Patterns:\n", resnet_counts)
    print("\nCustom CNN Misclassification Patterns:\n", custom_counts)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(resnet_counts, annot=True, cmap='Blues', fmt='g')
    plt.title("ResNet-50 Misclassification Patterns")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(custom_counts, annot=True, cmap='Reds', fmt='g')
    plt.title("Custom CNN Misclassification Patterns")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plt.show()

# Run analysis
analyze_misclassification_patterns()

# Function to display misclassified images
def plot_misclassified_samples(X, y_true, y_pred, misclassified_indices, model_name):
    num_samples = min(9, len(misclassified_indices))
    if num_samples == 0:
        print(f"No misclassified samples found for {model_name}!")
        return
    
    indices = random.sample(list(misclassified_indices), num_samples)
    class_labels = ["Clear Sky", "Thin Cloud", "Opaque Cloud"]
    
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[idx, :, :, 0], cmap='gray')  # Assuming first band is most informative
        plt.title(f"True: {class_labels[y_true[idx]]}\nPred: {class_labels[y_pred[idx]]}")
        plt.axis("off")
    plt.suptitle(f"Misclassified Samples - {model_name}")
    plt.show()

# Display misclassified samples for both models
plot_misclassified_samples(X_test, y_test, resnet_preds, resnet_misclassified, "ResNet-50 CNN")
plot_misclassified_samples(X_test, y_test, custom_preds, custom_misclassified, "Custom CNN")
