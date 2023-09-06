from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers.legacy import SGD, Adam # for M1/M2 macs, legacy needed
from keras.metrics import Precision, Recall, AUC, F1Score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from preprocessor import loadDir
from image_augmentation import augmentImages
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
import random
from keras import backend as K
import visualkeras
import zipfile

np.random.seed(42)
BATCH_SIZE = 20
# MODE can be "augmented", "preprocessed" or "archive"
MODE = "augmented"
TRAIN_PATH = f"{MODE}/train/"
VALIDATION_PATH = f"{MODE}/validation"
TESTING_PATH = f"{MODE}/testing"
NUM_EPOCHS = 10
LR = 0.0002

# clear the directory
if os.path.exists(MODE):
    shutil.rmtree(MODE)

# add it back
if MODE == "archive":
    with zipfile.ZipFile("screws.zip", 'r') as dir:
        dir.extractall()
elif MODE == "preprocessed":
    loadDir("archive/train")
else:
    augmentImages()

# create necessary directories
directories = [f"{MODE}/validation/good", f"{MODE}/validation/not-good", f"{MODE}/testing/good", f"{MODE}/testing/not-good"]
for dir in directories:
    os.makedirs(dir)

# move 10% of the data from training into each of testing and validation
for c in random.sample(glob.glob(f"{MODE}/train/good/*.png"), 25):
    shutil.move(c, f"{MODE}/validation/good")

for c in random.sample(glob.glob(f"{MODE}/train/not-good/*.png"), 5):
    shutil.move(c, f"{MODE}/validation/not-good")

for c in random.sample(glob.glob(f"{MODE}/train/good/*.png"), 25):
    shutil.move(c, f"{MODE}/testing/good")

for c in random.sample(glob.glob(f"{MODE}/train/not-good/*.png"), 5):
    shutil.move(c, f"{MODE}/testing/not-good")

# greyscale image, so only 1 channel
img_shape = (224, 224, 1) if K.image_data_format() == "channels_last" else (1, 224, 224)

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation="relu", input_shape=img_shape, padding="same")) # 224x224 grey-scale images

model.add(Conv2D(16, (3, 3), activation="relu", padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(20, (3, 3), activation="relu", padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid")) # binary classification hence 1 class & sigmoid

# model.compile(loss=binary_crossentropy, optimizer=SGD(learning_rate=LR), metrics=["accuracy"])
model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=LR), metrics=["accuracy", Precision(), Recall(), F1Score(), AUC()])

# Get summary of model and visualise the layers
model.summary()
visualkeras.layered_view(model, to_file="architecture.png") # .show()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="grayscale",
    # save_format="jpeg",
    # save_to_dir="generated/train"
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale"
)

testing_generator = validation_datagen.flow_from_directory(
    TESTING_PATH,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="grayscale"
)

print(f"Len of train: {len(train_generator)}") # should be train_generator.samples // BATCH_SIZE

print(f"Traing generator samples = {train_generator.samples}")
print(f"Validation generator samples = {validation_generator.samples}")

mapping = train_generator.class_indices  # {'good': 0, 'not-good': 1}
# 250 good samples and 50 not-good samples so ratio is 0.83:0.17
class_weights = {mapping["not-good"] : 0.17, mapping["good"] : 0.83}
hist = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//BATCH_SIZE,
    # class_weight=class_weights # imbalanced dataset
)

# Save the model
if not os.path.exists("models"):
    print("Created new directory: models")
    os.makedirs("models")
from datetime import datetime
now = datetime.now()
model.save(f"models/{now.strftime('%Y_%m_%d_%H%M')}_{NUM_EPOCHS}_epochs.keras")

# Evaluation stuff
categories = ["good", "not-good"] # order determined by .class_indices
print(model.evaluate(testing_generator)) # loss value and metrics values
y_pred = model.predict(testing_generator)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(testing_generator.classes, y_pred, normalize='pred')
print(cm)
print(classification_report(validation_generator.classes, y_pred, target_names=categories))

matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
matrix.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix.png")
plt.show()
