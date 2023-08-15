import cv2
import os
import sys
import numpy as np
from sklearn import svm
from skimage import feature
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


categories = ['male', 'female']
# Input direc from database
dirTrain="C:\\Users\\arina\\train"
dirValid= "C:\\Users\\arina\\valid"
dirTest=  "C:\\Users\\arina\\test"
class LocalBinaryPatterns:
    def __init__(self, numofPoints, radius):
        # Save the number of points and radius
        self.numofPoints = numofPoints
        self.radius = radius

    # Part 2 - Feature extraction
    def describe(self, image, eps=1e-7):
    # Compute the Local Binary Pattern representation of the image,then use the LBP representation
    # To build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numofPoints,self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, self.numofPoints+ 3),range=(0, self.numofPoints + 2))
        # normalize of histogram
        histogram = hist.astype("float")
        histogram /= (hist.sum() + eps)
        return histogram

# Loading images with labels from directory
def loading(direc, desc):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(direc, category)
        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            im = cv2.imread(imgpath)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            labels.append(categories.index(category))
            data.append(hist)
    return data, labels


# Find the :max accuracy, model with this max accuracy
def find_max_accuracy(data_train, labels_train, data_valid, labels_valid, max_accuracy, good_model):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid)
    model.fit(data_train, labels_train)
    if max_accuracy < model.score(data_valid, labels_valid):
        max_accuracy = model.score(data_valid, labels_valid)
        good_model=model
    model = svm.SVC(kernel='linear', C=1)
    model.fit(data_train, labels_train)
    if max_accuracy < model.score(data_valid, labels_valid):
        max_accuracy = model.score(data_valid, labels_valid)
        good_model=model
    return max_accuracy, good_model


model = svm.SVC(kernel='linear', C=1)
desc = LocalBinaryPatterns(8, 1)
arrTrain=loading(dirTrain, desc)
arrValid = loading(dirValid, desc)
max_accuracy_8, model= find_max_accuracy(arrTrain[0], arrTrain[1], arrValid[0], arrValid[1], 0, model)
desc1 = LocalBinaryPatterns(24, 3)
arrTrain=loading(dirTrain, desc1)
arrValid = loading(dirValid, desc1)
max_accuracy_24, model= find_max_accuracy(arrTrain[0], arrTrain[1], arrValid[0], arrValid[1], max_accuracy_8, model)
if max_accuracy_24>max_accuracy_8:
    desc=desc1

# Actual and predicted values for confusion matrix
actual = []
predicted = []

# Loading the testing data from directory to list
for category in categories:
    path = os.path.join(dirTest, category)
    for img in os.listdir(path):
        im = cv2.imread(os.path.join(path, img))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        prediction = model.predict(hist.reshape(1, -1))
        actual.append(category)
        predicted.append(categories[prediction[0]])

mat= confusion_matrix(actual, predicted, labels=['male', 'female'])

f = open("Results.txt", "w")
f.write(" Svm with kernel : {}".format(model.kernel))
f.write("\n Number points : {}".format(desc.numofPoints))
f.write("\n Radius : {}".format(desc.radius))
f.write("\n Accuracy : {:.2f}%".format(max_accuracy_24*100))
f.write("\n\n Confusion Matrix : ")
f.write(f"\n \t male \t female \n male \t {mat[0][0]} \t {mat[1][0]} \n female \t {mat[0][1]} \t {mat[1][1]}")
f.close()