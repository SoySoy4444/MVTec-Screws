# a copy of the covid xray model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image

TRAIN_PATH = "archive/train/"
TRAIN_PATH = "preprocessed/"
BATCH_SIZE = 30

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss=binary_crossentropy, optimizer=SGD(learning_rate=0.0002), metrics=["accuracy"])

print(model.summary())

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    save_format="jpeg",
    save_to_dir="generated/train"
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,  # same directory as training data
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)  # set as validation data

print(f"Len of train: {len(train_generator)}")
print(len(validation_generator))

print(train_generator.class_indices)
print(validation_generator.class_indices)

print(type(train_generator))

print(f"Traing generator samples = {train_generator.samples}")
print(f"Validation generator samples = {validation_generator.samples}")

NUM_EPOCHS = 10
hist = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//BATCH_SIZE,
)

if not os.path.exists("models"):
    print("Created new directory: models")
    os.makedirs("models")

from datetime import datetime
now = datetime.now()
model.save(f"models/{now.strftime('%Y/%m/%d_%H%M')}_{NUM_EPOCHS}_epochs.keras")

# need to put test_generator
print(model.evaluate(train_generator)) # loss value and metrics values
