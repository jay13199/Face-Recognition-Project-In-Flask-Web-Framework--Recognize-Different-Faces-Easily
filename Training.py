import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the collected data
data = pd.read_csv('gesture_data.csv')

# Separate features and labels
X = data.iloc[:, 1:].values  # Landmarks
y = data.iloc[:, 0].values  # Gesture labels

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save the label encoder classes
np.save('gesture_classes.npy', label_encoder.classes_)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save the model
model.save('gesture_classifier.h5')
