import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

# Check if the model is already saved
MODEL_PATH = "cifar10_model.h5"

if os.path.exists(MODEL_PATH):
    print("Loading the pre-trained model...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print("Training a new model...")
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert labels to categorical (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Define the CNN model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=1)

    # Save the trained model
    model.save(MODEL_PATH)
    print("Model saved successfully!")

# Evaluate the model
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test / 255.0
y_test = keras.utils.to_categorical(y_test, 10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Function to predict a custom image
def predict_custom_image(image_path):
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Load and preprocess the image
    img = Image.open(image_path).resize((32, 32))
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    print(f"Predicted class: {class_names[predicted_class]}")
    return class_names[predicted_class]

# Example usage:
predict_custom_image('../images/dog.jpg')
