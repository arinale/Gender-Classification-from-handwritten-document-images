
Description:
 The code cross over all the images and determine which of them are in a female handwriting and which are in the handwriting of a male.
Environment :
 : The code is written using python,you have to install these Libraries
* cv2
*os
* sys
*numpy 
* pandas 
 sklearn 
sklearn.metrics 
 sklearn.model_selection 
 sklearn.metrics 
import cv2
import os
import sys 
import numpy as np
import pandas as pdfrom sklearn import svm
from skimage import feature
from sklearn.metrics import confusion_matrix
 : to run the code from the command line using this command
> python classifier.py path_train path_val path_test
path_train ------ input folder that contain the images to train
path_val ------ input folder that contain the images to valid
path_test ------ input folder that contain the images to test