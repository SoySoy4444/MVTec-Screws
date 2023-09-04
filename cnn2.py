# instead of using the training set for just training & validation,
# uses the set for training, validation & testing in 75-15-15 split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Rescaling
from keras.losses import binary_crossentropy
from keras.optimizers.legacy import SGD # for M1/M2 macs
from keras.utils import image_dataset_from_directory#, plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
# from PIL import Image
import tensorflow as tf
import warnings
from matplotlib import MatplotlibDeprecationWarning

image_size = (224, 224)
batch_size = 32

DIR = "augmented/"
# DIR = "preprocessed/"
# DIR = "archive/train/"
train_ds, val_ds = image_dataset_from_directory(
    DIR,
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
plt.figure(figsize=(8, 8))
for images, labels in val_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

# -----------

BATCH_SIZE = 10

model = Sequential()
# model.add(Rescaling(scale=1./255))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 1), padding="same"))

model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss=binary_crossentropy, optimizer=SGD(learning_rate=0.0001), metrics=["accuracy"])

model.summary()

print(len(train_ds))
print(len(val_ds))

# plot_model(model, show_shapes=True)

NUM_EPOCHS = 10
hist = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=val_ds
)

# TODO: fix accuracy not increasing
""" with augmented/
Epoch 1/10
23/23 [==============================] - 26s 1s/step - loss: 135.0311 - accuracy: 0.4986 - val_loss: 0.6926 - val_accuracy: 0.5611
Epoch 2/10
23/23 [==============================] - 27s 1s/step - loss: 0.6911 - accuracy: 0.5542 - val_loss: 0.6923 - val_accuracy: 0.5611
Epoch 3/10
23/23 [==============================] - 27s 1s/step - loss: 0.6916 - accuracy: 0.5542 - val_loss: 0.6906 - val_accuracy: 0.5611
Epoch 4/10
23/23 [==============================] - 27s 1s/step - loss: 0.6889 - accuracy: 0.5542 - val_loss: 0.6909 - val_accuracy: 0.5611
Epoch 5/10
23/23 [==============================] - 28s 1s/step - loss: 0.6900 - accuracy: 0.5542 - val_loss: 0.6897 - val_accuracy: 0.5611
Epoch 6/10
23/23 [==============================] - 29s 1s/step - loss: 0.6925 - accuracy: 0.5542 - val_loss: 0.6895 - val_accuracy: 0.5611
Epoch 7/10
23/23 [==============================] - 34s 1s/step - loss: 0.6903 - accuracy: 0.5542 - val_loss: 0.6889 - val_accuracy: 0.5611
Epoch 8/10
23/23 [==============================] - 39s 2s/step - loss: 0.6870 - accuracy: 0.5542 - val_loss: 0.6880 - val_accuracy: 0.5611
Epoch 9/10
23/23 [==============================] - 40s 2s/step - loss: 0.6883 - accuracy: 0.5542 - val_loss: 0.6874 - val_accuracy: 0.5611
Epoch 10/10
23/23 [==============================] - 43s 2s/step - loss: 0.6877 - accuracy: 0.5542 - val_loss: 0.6872 - val_accuracy: 0.5611
"""
# with preprocessed/ or train/, accuracy stuck at 0.83333