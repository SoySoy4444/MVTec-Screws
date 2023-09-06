import matplotlib.pyplot as plt

img = plt.imread("archive/train/not-good/scratch_head003.png")
plt.subplot(211)
plt.imshow(img, cmap="gray")
plt.subplot(212)
plt.ylabel("Frequency")
plt.xlabel("Pixel intensity")
plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

plt.show()