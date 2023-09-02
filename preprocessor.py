import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def loadDir(root: str):
    if not os.path.exists("preprocessed"):
        print("Created new directory: preprocessed")
        os.makedirs("preprocessed")
    if not os.path.exists("preprocessed/not-good"):
        print("Created new directory: preprocessed/not-good")
        os.makedirs("preprocessed/not-good")
    if not os.path.exists("preprocessed/good"):
        print("Created new directory: preprocessed/good")
        os.makedirs("preprocessed/good")
    
    for path, subdirs, files in os.walk(root):
        # print(path, subdirs)
        for name in files:
            img_path = os.path.join(path, name)
            if (img_path.split('.')[1] == 'png'):  # necessary due to .DS_Store
                image = getProcessedBinary(img_path)
                print(path)
                
                if "not-good" in path:
                    cv2.imwrite(f"preprocessed/not-good/{name}", image)
                else:
                    cv2.imwrite(f"preprocessed/good/{name}", image)

def getProcessedBinary(imgpath: str):
    blur_r = 3
    pic1 = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    pic1 = cv2.GaussianBlur(pic1, (blur_r, blur_r), 9)

    THRESH = 110
    _, thresh1 = cv2.threshold(pic1, THRESH, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
    # cv2.imshow(f"original vs thres blur r = {blur_r}, thresh={THRESH}", np.hstack((pic1, thresh1)))
    # cv2.waitKey(0)
    return thresh1

def getProcessedAdaptive(imgpath: str):
    blur_r = 3
    pic1 = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    pic1 = cv2.GaussianBlur(pic1, (blur_r, blur_r), 9)

    thresh3 = cv2.adaptiveThreshold(pic1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3) 
    # cv2.imshow("adaptiveThresh", np.hstack((pic1, thresh3)))
    # cv2.waitKey(0)
    return thresh3


def getProcessedMorph(imgpath: str):
    morph = cv2.imread(imgpath)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # take morphological gradient
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

    # split the gradient image into channels
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

    # merge the channels
    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
    return image_channels

if __name__ == "__main__":
    loadDir("archive/train")

