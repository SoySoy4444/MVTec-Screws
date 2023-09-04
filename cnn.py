from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
import os
from keras import backend as K

TRAIN_PATH = "archive/train/"
# TRAIN_PATH = "preprocessed/"
BATCH_SIZE = 20

# greyscale image, so only 1 channel
img_shape = (224, 224, 1) if K.image_data_format() == "channels_last" else (1, 224, 224)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=img_shape, padding="same")) # 224x224 grey-scale images

model.add(Conv2D(32, (3, 3), activation="relu", padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation="relu", padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu", padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid")) # binary classification hence 1 & sigmoid

model.compile(loss=binary_crossentropy, optimizer=SGD(learning_rate=0.0002), metrics=["accuracy"])

model.summary()

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
    save_to_dir="generated/train",
    color_mode="grayscale"
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    color_mode="grayscale"
)

print(f"Len of train: {len(train_generator)}") # should be train_generator.samples // BATCH_SIZE

print(f"Traing generator samples = {train_generator.samples}")
print(f"Validation generator samples = {validation_generator.samples}")

mapping = train_generator.class_indices  # {'good': 0, 'not-good': 1}
# 250 good samples and 50 not-good samples so ratio is 0.83:0.17
class_weights = {mapping["not-good"] : 0.17, mapping["good"] : 0.83}
NUM_EPOCHS = 10
hist = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//BATCH_SIZE,
    class_weight=class_weights # imbalanced dataset
)

# Save the model
if not os.path.exists("models"):
    print("Created new directory: models")
    os.makedirs("models")
from datetime import datetime
now = datetime.now()
model.save(f"models/{now.strftime('%Y_%m_%d_%H%M')}_{NUM_EPOCHS}_epochs.keras")

# Evaluation stuff
# need to put test_generator
print(model.evaluate(train_generator)) # loss value and metrics values
