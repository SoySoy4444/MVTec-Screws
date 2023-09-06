import os
from keras.preprocessing.image import ImageDataGenerator

def augmentImages():
    datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0,  # % shift
        height_shift_range=0,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='reflect', # constant, nearest, reflect or warp
    ) 
    if not os.path.exists("augmented/train/good"):
        print("Created new directory: augmented/train/good")
        os.makedirs("augmented/train/good")
    if not os.path.exists("augmented/train/not-good"):
        print("Created new directory: augmented/train/not-good")
        os.makedirs("augmented/train/not-good")

    for category in ["good", "not-good"]:
        i = 0
        print(f'archive/train/{category}/')
        for batch in datagen.flow_from_directory(directory=f'archive/train/',
                                                batch_size=16,
                                                target_size=(256, 256),
                                                color_mode="grayscale",
                                                classes=[category],
                                                save_to_dir=f'augmented/train/{category}/',
                                                save_prefix='aug',
                                                save_format='png'):
            i += 1
            if i > 31:
                break

if __name__ == "__main__":
    augmentImages()