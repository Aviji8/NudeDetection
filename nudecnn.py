# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:58:34 2020

@author: Aviji8
"""


# Importing the Keras libraries and packages
from keras.models import Sequential #initialize NN
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense #TO ADD FULLY CONNECTION LAYER 
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu')) #hidden layer
classifier.add(Dense(units = 1, activation = 'sigmoid')) #op layer
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'E:\Nude_Clasification',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'E:\Nude_Clasification',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

classifier.fit_generator(training_set,
steps_per_epoch = 300,
epochs = 1,
validation_data = test_set,
validation_steps = 200)

classifier.save("nude_non_nude_classifier.h5")

from skimage.io import imread
from skimage.transform import resize
import numpy as np
     
class_labels = {v: k for k, v in training_set.class_indices.items()}
     
img = imread(r 'E:\Nude_Clasification') 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
     
if(np.max(img)>1):
        
        img = img/255.0
     
prediction = classifier.predict_classes(img)
     
print (class_labels[prediction[0][0]])

