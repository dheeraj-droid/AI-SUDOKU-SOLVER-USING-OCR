import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from sklearn.metrics import classification_report

# Load MNIST dataset

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Convert labels/categories to one hot encoding

test_labels = to_categorical(test_labels, 10)
train_labels = to_categorical(train_labels, 10)

# Normalize test and train data in range [0, 1]

train_data = train_data / 255
test_data = test_data / 255

# Reshape to include grayscale color channel

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

# Create model
model = Sequential()

# First set of Convolution layer
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

# Pooling layer
model.add(MaxPool2D((2, 2)))

# Second set of Convolution layer
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))

# Third set of Convolution layer
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))

# Pooling layer
model.add(MaxPool2D((2, 2)))

# Flat layer: 2 Dimension --> 1 Dimension
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

# Output layer/classifer
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary

model.summary()

# Train model
model.fit(train_data, train_labels, epochs=20)

# Evaluate model performance on test data and labels

model.evaluate(test_data, test_labels)

# Predict classes on test images

predictions = model.predict(test_data)

# Classification report

print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1)))

# Save model

model.save('./model/model-4.h5')