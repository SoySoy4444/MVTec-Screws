# MVTec Screws Image Anomaly Detection

A Convolutional Neural Network (CNN) is employed to detect anomalous screws in the MVTec AD Screws dataset.

## How to setup

1. If desired, create a new virtual environment and activate.
2. Run `pip install -r requirements.txt` to install all required Python modules
3. Run `python3 setup.py` to  download the dataset from GDrive.

## How to run

1. Set the `MODE` variable inside cnn.py to be either `archive` (default), `preprocessed` or `augmented`
   - `archive`: default settings
   - `preprocessed`: some thresholding pre-processing is first applied to the images. Select which type by going to `preprocessor.py` and changing the function used in line 18.
   - `augmented`: the image is first augmented and the minority class is upsampled.
2. Change any other variables as desired
   -learning rate, batch size, optimiser, number of epochs
3. Run with `python3 cnn.py`

## Explanation of files

- `svm.py`: an SVM-based approach. Run on its own. Note: the archive folder must be in its original, unzipped state.
- `cnn.py`: uses `ImageDataGenerator` to split into training and validation set
- `preprocessor.py`: creates binary, adaptive, morph or canny thresholded versions of the images and saves them inside preprocessed/train/good and preprocessed/train/not-good
- `image_augmentation.py`: image data augmentation. Stores them inside augmented/train/good and augmented/train/not-good
- `setup.py`: used for downloading the dataset
- `image_hist.py`: used to visualise the histogram of an image
