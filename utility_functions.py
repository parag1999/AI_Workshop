# -*- coding: utf-8 -*-
#here we write function which are used a lot of times in order to avoid duplication of code
#this function is used to make the input suitable for testing as well as training
from skimage import color , exposure , transform
import numpy as np
 
#the image_size should be a tuple bcoz it is immutable
def pre_process_image(image, image_size):
    #hsv means hue,saturation and value a type of img representation format
    #converting rgb to hsv
    hsv_image = color.rgb2hsv(image)
    #v in hsv i.e value relates to the brightness of an image
    #taking values of v channel in hsv 
    # histogram equalize is used to improve the contrast of an image to make the features more defined
    hsv_image[:,:,2] = exposure.equalize_hist(hsv_image[:,:,2])
    #Changes are saved back to the original image
    image = color.hsv2rgb(hsv_image)
    #So that all images are of same size
    image = transform.resize(image, image_size)
    
    #Used to make the RGB input as BGR input cause many libraries follow the order BGR
    image = np.rollaxis(image, -1)
    
    return image


#In the Final_Training/Images each folder is named as 00001,00002 and so on so these folders are considered as classes containing images
#this function in a way returns the class    
def get_class_label(image_path):
    #Since split returns a list so -2 is the index at which the class name exists
    return int(image_path.split('\\')[-2])
    #so this will give class 00001 as 1
    
#decreases the learning rate as we go on to the next epoch
def learning_rate_schedule(epoch, lr=0.01):
    return lr*(0.1**int(epoch/2))