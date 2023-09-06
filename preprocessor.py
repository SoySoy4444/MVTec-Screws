import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def loadDir(root: str):
    dirs = ["preprocessed/train/not-good", "preprocessed/train/good"]
    for dir in dirs:
        if not os.path.exists(dir):
            print(f"Created new directory: {dir}")
            os.makedirs(dir)

    # path = archive/train/good and archive/train/not-good
    for path, _, files in os.walk(root):
        for name in files:
            img_path = os.path.join(path, name)
            if (img_path.split('.')[1] == 'png'):  # necessary due to .DS_Store
                image = getProcessedCanny(img_path, True)                
                cv2.imwrite(f"preprocessed/train/{path.split('/')[-1]}/{name}", image)

def getProcessedBinary(imgpath: str, THRESH = 110):
    blur_r = 3
    pic1 = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    pic1 = cv2.GaussianBlur(pic1, (blur_r, blur_r), 9)

    _, thresh1 = cv2.threshold(pic1, THRESH, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
    cv2.imshow(f"original vs thres blur r = {blur_r}, thresh={THRESH}", np.hstack((pic1, thresh1)))
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

    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
    return image_channels

# If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge. 
# If a pixel gradient value is below the lower threshold, then it is rejected.
def getProcessedCanny(imgpath: str, thresholdFirst = False):
    if thresholdFirst:
        img = getProcessedBinary(imgpath, 120)
        edges = cv2.Canny(img, threshold1=90, threshold2=180, apertureSize=3)
    else:
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 5)
        edges = cv2.Canny(img, threshold1=70, threshold2=165, apertureSize=3)

    # plt.subplot(121)
    # plt.imshow(img, cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122)
    # plt.imshow(edges, cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return edges

if __name__ == "__main__":
    # loadDir("archive/train")
    # quit()
    cat = "not-good"
    filename = "scratch_head004.png"
    image = getProcessedCanny("archive/train/" + cat + "/" + filename)
    cv2.imshow(f"preprocessed/{cat}/{filename}", image)
    cv2.waitKey(0)
