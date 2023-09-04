# MVTec Screws Image Anomaly Detection

A Convolutional Neural Network (CNN) is employed to detect anomalous screws in the MVTec AD Screws dataset.

## How to setup

Run `bash setup.sh` in order to install all required Python modules and download the dataset from GDrive.

## Explanation of files

- `cnn.py`: uses `ImageDataGenerator` to split into training and validation set
- `cnn2.py`: uses `image_dataset_from_directory` to split into training and validation set.
- `cnn3.py`: uses `train_test_split` to split into training, validation and testing set. DOES NOT CURRENTLY WORK.
- `preprocessor.py`: creates binary, Otsu, adaptive or morph thresholded versions of the images and saves them inside /preprocessed
- `image_augmentation.py`: image data augmentation. Stores them inside augmented/train/good and augmented/train/not-good
- 