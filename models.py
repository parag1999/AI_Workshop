# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout ,Flatten , Dense
from keras import backend as K 

#Arranges the input for additional layers 
#channels_first means take dimension size one at a time in order to make the matrix
K.set_image_data_format('channels_first')

def cnn_model(image_size, number_of_classes):
    
    model = Sequential()
    #Initializing
    #defining the type of model using google sequential model
    
    model.add(Conv2D(32, (3,3),padding="same",input_shape = (3, image_size[0],image_size[1]),activation = "relu"))
    #32 indicates the depth, (3,3)the kernel size ,same is the type of padding since the iage dimension is 48*48*3 ,relu is a type of activation function
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    #Max Pooling
    
    model.add(Dropout(0.2))
    # 0.2 indicates to disable 20% of output in order to counter overfitting
    
    model.add(Conv2D(64, (3,3), padding="same", activation = "relu"))
    #here we as increased the depth thus increasing the layers
    #We have not defined the shape here bcoz we don't know what type of input we will get from previous output or layer in this case MaxPooling or Dropout layer
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    #2,2 indicates it makes a kernel of 2x2 from which it takes the aximum value
    
    model.add(Dropout(0.05))
    
    model.add(Flatten())
    #Makes the array 2D instead of 3D so that computer can learn from it google rolling and unrolling
    
    model.add(Dense(128, activation = "relu" ))
    #it connect the fully connected layer created by flatten and helps in giving the probability of each and every outcome
    #here 128 indicates 128 outcomes
    
    model.add(Dropout(0.05))
    
    model.add(Dense(number_of_classes, activation = "softmax"))
    # softmax is used for giving probability
    # number_of_classes is taken because since the dimension of Y is (no of images, no of classes) so we want exactly that many output
    return model