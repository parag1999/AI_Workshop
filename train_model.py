# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from utility_functions import pre_process_image, get_class_label, learning_rate_schedule
from skimage import io
from models import cnn_model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import h5py

#Since there are 43 folders in Images file of Final_Training
number_of_classes = 43
def read_training_images(image_size, root_dir):
    try:
        with h5py.File('Input.h5') as hf:
            X, Y = hf['images'][:],hf['labels'][:]
    
    except Exception as e: 
        print(e)
        images = []
        labels = []
        #glob is used to filter out and select files only with .ppm format using regex
        all_images_path = glob.glob(os.path.join(root_dir,'*/*.ppm'))
        # it is used to shuffle the path so that it recieves img frm all different classes or folders
        np.random.shuffle(all_images_path)
        
        for image_path in all_images_path:
            try:
                image = io.imread(image_path)
                #check utility_function.py
                image = pre_process_image(image, image_size)
                images.append(image)
                #check utility_function.py
                label = get_class_label(image_path)
                labels.append(label)
            #In case a file is corrupted it won't start the process again instead continue with the next img
            except (IOError, OSError):
                pass
    #AT this point we have converted all the data in list format it will be converted into numpy arrays bcoz keras accepts numpy array
    #IN CNN the convention is to use X as input and Y as output            
        #float32 indicates the type is float and array is of 32 bits
        X = np.array(images, dtype='float32')
        #eye function creates diagonal matrix where the labels are added as the diagonal element
        Y = np.eye(number_of_classes, dtype='uint8')[labels]
        #This is used to save data in .h5 format so that for every epoch we don't need to load the whole database
        with h5py.File('Input.h5','w') as hf:
            hf.create_dataset('images',data=X)
            hf.create_dataset('labels',data=Y)
        
    return X, Y


def train_model():
    root_dir = 'GTSRB\\Final_Training\\Images\\'
    image_size = (48,48)
    X , Y = read_training_images(image_size, root_dir)
    model = cnn_model(image_size, number_of_classes)
    learning_rate = 0.01
    sgd = SGD(lr=learning_rate)
    #Just like perceptron SGD is a type of optimizer google it
    model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])
    #When we have many class we use categorical_crossentropy
    #metrics define the purpose or the paramter on which it should work
    batch_size = 16
    #it takes data in a group of 16
    epochs = 3
    # In 1 epoch it will take all 39000 images in batches of 16 in order to train it
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[LearningRateScheduler(learning_rate_schedule),
                                                                                     ModelCheckpoint('model_train.h5',save_best_only=True)])
    #validation_split  0.2 means it tests 20% of the trained images while training the remaining images 
    # callbacks are like checkpoint in case of trmination it continues from where it stopped
    
if __name__ == "__main__":
    train_model()
    
    