# -*- coding: utf-8 -*-
#COMPARE THIS FILE WITH train_model.py file to check their similarties

import os
from keras.models import load_model
import pandas as pd
from utility_functions import pre_process_image
from skimage import io
import numpy as np

X_test = []
Y_test = []
image_size = (48,48)
#Open and see GT-final_test.csv file
#Using pandas lib we read csv data
test = pd.read_csv('E:\\traffic_sign_recognition\\GTSRB\\GT-final_test.csv',sep=';')
#sep is the parameter pass to split so here we slipt data from the .csv file by ; 
for file_name, class_id in zip(list(test['Filename']),list(test['ClassId'])):
    #zip is used to iterate two or more list at a time
    image_path = os.path.join('E:\\traffic_sign_recognition\\GTSRB\\Final_Test\\Images',file_name)
    image = io.imread(image_path)
    #EVEN during testng we need to feed the input data in pre process form
    processed_image = pre_process_image(image , image_size)
    X_test.append(processed_image)
    Y_test.append(class_id)
    
X_test = np.array(X_test,dtype='float32')
Y_test = np.array(Y_test)

model = load_model('model_train.h5')

Y_pred = model.predict_classes(X_test)
#read on google about classifying and accuracy

accuracy = np.sum(Y_pred == Y_test)/np.size(Y_pred)
#right predictions is divided by total predictions

