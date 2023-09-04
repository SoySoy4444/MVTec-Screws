import os
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0,  # % shift
    height_shift_range=0,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest', # constant, nearest, reflect or warp
) 

if not os.path.exists("augmented"):
    print("Created new directory: augmented")
    os.makedirs("augmented")
if not os.path.exists("augmented/good"):
    print("Created new directory: augmented/good")
    os.makedirs("augmented/good")
if not os.path.exists("augmented/not-good"):
    print("Created new directory: augmented/not-good")
    os.makedirs("augmented/not-good")

for category in ["good", "not-good"]:
    i = 0
    print(f'archive/train/{category}/')
    for batch in datagen.flow_from_directory(directory=f'archive/train/',
                                            batch_size=16,
                                            target_size=(256, 256),
                                            color_mode="grayscale",
                                            classes=[category],
                                            save_to_dir=f'augmented/{category}/',
                                            save_prefix='aug',
                                            save_format='png'):
        i += 1
        if i > 31:
            break
