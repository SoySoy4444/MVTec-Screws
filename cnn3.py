# instead of using the training set for just training & validation,
# uses the set for training, validation & testing in 75-15-15 split
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers.legacy import SGD # for M1/M2 macs
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf

def loadDir(root: str):
    dataset = []
    label = []
    for path, subdirs, files in os.walk(root):
        # print(path, subdirs)
        for name in files:
            img_path = os.path.join(path, name)
            if (img_path.split('.')[1] == 'png'):  # necessary due to .DS_Store
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                # image = Image.fromarray(image, 'RGB')
                # image = image.resize((224, 224))
                dataset.append(image)
                # print(path)
                label.append(int("not-good" in path))
    return dataset, label


physical_devices = tf.config.experimental.list_physical_devices("GPU")
print(f"Num GPUs available = {len(physical_devices)}")

X, y = loadDir("archive/train")
print(f"Size of X = {len(X)}, size of y = {len(y)}")
print(y)

# Split into 70% training set and 30% temporary set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split temporary set into validation and testing sets (50% of 30% is 15% each)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(len(X_train), len(X_test), len(X_val)) # 215, 45, 45
print(len(y_train), len(y_test), len(y_val))  # 215, 45, 45

# -----------

BATCH_SIZE = 30

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)))

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

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow(np.array(X_train, dtype=object), y=np.array(y_train))
val_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow(np.array(X_val, dtype=object), y=np.array(y_val))
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow(np.array(X_test, dtype=object), y=np.array(y_test))


NUM_EPOCHS = 10
hist = model.fit(
    train_batches,
    batch_size=BATCH_SIZE,
    steps_per_epoch=len(train_batches)//BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=val_batches,
    validation_steps=len(val_batches)//BATCH_SIZE,
)
