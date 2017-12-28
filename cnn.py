# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 21:41:01 2017

@author: Eoghan
"""

# Part 1 - Building the CNN

# Import the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Initalizing the CNN
classifier = Sequential()

# Step 1 - Convolution and Pooling

# Adding a first convolutional layer
classifier.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

# compute quantities required for featurewise normalization
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=200),
                    steps_per_epoch=len(x_train)/200,
                    epochs=25,
                    validation_data = (x_test, y_test),
                    workers = 4)

# Make a single prediction
test_image = image.load_img('test_images/plane.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image /= 255
result = classifier.predict(test_image)

# Save model
classifier.save('model5.h5') 

# Load model
classifier = load_model('model4.h5')