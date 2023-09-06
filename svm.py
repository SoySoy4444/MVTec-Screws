from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

categories = ["good","not-good"]
flat_data_arr = []
target_arr = []
TRAIN_PATH = "archive/train" 

for i, cat in enumerate(categories):
    path = os.path.join(TRAIN_PATH, cat)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img), as_gray=True)
        img_resized = resize(img_array, (224, 224, 1))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(i) # label is 0 or 1
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(flat_data)
df['Target'] = target
df.shape

x = df.iloc[:,:-1] 
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

print(f"Split into x_train {len(x_train)} and x_test {len(x_test)}")

# Defining the parameters grid for GridSearchCV
param_grid = {"C" : [0.001, 0.1, 1, 10, 100],
            "gamma" : [0.0001, 0.001, 0.1, 1, 10],
            "kernel" : ["rbf", "sigmoid"], # rbf or sigmoid good for images
            "verbose" : [True]} 

svc = svm.SVC(probability=False)  # probability=True is too slow
model = GridSearchCV(svc, param_grid)  # Hyperparameter tuning with GridSearch


print("Beginning training...")
model.fit(x_train,y_train)

print("Training finished")
print(f"Best indicator was {model.best_estimator_} with the parameters {model.best_params_} with a score of {model.best_score_}")

y_pred = model.predict(x_test)

print("Prediction finished)")
accuracy = accuracy_score(y_pred, y_test)

print(f"Accuracy: {accuracy}")

print(classification_report(y_test, y_pred, target_names=categories))

cm = confusion_matrix(y_test, y_pred, normalize='pred')
print(cm)

matrix = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=categories)
matrix.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix_svm.png")
plt.show()
