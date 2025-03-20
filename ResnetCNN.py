import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input

# Load preprocessed dataset
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# ResNet expects 3-channel images, so we may need to adjust bands
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

# Load pre-trained ResNet50 with ImageNet weights, excluding top layers
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

# Freeze base model layers to retain pre-trained features
base_model.trainable = False

# Add custom classification head
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
out = Dense(3, activation='softmax')(x)  # 3 classes: clear sky, thin cloud, opaque cloud

# Define model
model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save model
model.save("resnet_cnn_model.h5")
print("ResNet-based CNN model saved!")