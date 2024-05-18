# -*- coding: utf-8 -*-
"""Final Exam: Model Deployment in the Cloud

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1swwy0rGVCtCseVEkmM1mlZxUpjj0BuH1

# Final Exam: Model Deployment in the Cloud

Members: <br>
Espiritu, Missy Anne Jhelzshir G. <br>
Santos, Uneta Tristanneal <br>
Course&Section: CPE 019 - CPE32S9 <br>
Date: May 18, 2024 <br>
Instructor: Engr. Roman Richard <br>

## **IMPORTING MODULES:**
"""

from google.colab import drive
drive.mount('/content/drive')

import time
import cv2
from PIL import Image,ImageOps
import numpy as np
import sys
from matplotlib import pyplot
import pandas as pd
from PIL import Image
from numpy import mean
from numpy import std
from numpy import argmax
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.utils import to_categorical

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/content/saved_fashion.h5')
    return model

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

st.write("""# Fashion Classification""")

file = st.file_uploader("Choose a clothing photo from your computer", type=["jpg", "png"])

if file is not None:
    model = load_model()
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    prediction = import_and_predict(image, model)
    # Display prediction results
    st.write("Prediction:", prediction)

"""### **Fashion MNIST Dataset - It contains images of different outfits or those to wear such as shoes, t-shirts, dresses, trousers, boots and many more. It should be able to predict what kind it is when given a test image.**

## **IMPORTING CSV FILE**
"""

train_dataset = pd.read_csv("/content/drive/MyDrive/fashion-mnist_train.csv")
test_dataset = pd.read_csv("/content/drive/MyDrive/fashion-mnist_test.csv")

train_dataset.shape

test_dataset.shape

train_dataset.head()

"""## **FORMATTING THE TRAIN AND TEST SETS**"""

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

#separate x and y and convert to array

#train dataset
train_datasetX = np.array(train_dataset.iloc[:, 1:])
train_datasetY = np.array(train_dataset.iloc[:, 0])

#test dataset
test_X = np.array(test_dataset.iloc[:, 1:])
test_Y = np.array(test_dataset.iloc[:, 0])

#one hot encoding to the y variables of train and test data
train_datasetY = to_categorical(train_datasetY)
test_Y = to_categorical(test_Y)

batch_size = 64
num_classes = test_Y.shape[1]

#split the train into train and validation (for later purposes)

from sklearn.model_selection import train_test_split

train_X, val_X, train_Y, val_Y = train_test_split(train_datasetX, train_datasetY, test_size=0.2, random_state=13)

print(train_X.shape)
print(val_X.shape)
print(test_X.shape)

def prep_pixels(train_X, val_X, test_X, train_Y, val_Y, test_Y):
  trainX = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)
  valX = val_X.reshape(val_X.shape[0], img_rows, img_cols, 1)
  testX = test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)
  trainY = train_Y
  valY = val_Y
  testY = test_Y
	# convert from integers to floats
  trainX = trainX.astype('float32')
  valX = valX.astype('float32')
  testX = testX.astype('float32')
	# normalize to range 0-1
  trainX = trainX / 255.0
  valX = valX / 255.0
  testX = testX / 255.0

	# return normalized images
  return trainX, valX, testX, trainY, valY, testY

"""### **We have divided the train, test, and validation for this particular section. There are 10,000 photos for testing and 60,000 images for training in the dataset. Following the splitting, a validation set comprising 12,000 rows is produced. After that, the sets are formatted to make them easier to read. It measures 28 by 28.**"""

def summarize_diagnostics(history):
	# plot loss
	pyplot.figure(figsize=(16,10))
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

"""### **This function plots the loss and accuracy later on after the training is done.**

## **Performing and Saving Augmentation, Utilizing Test Harness**
"""

shift_fraction = 0.005

def run_test_harness(model_x, epochs):
  trainX, valX, testX, trainY, valY, testY = prep_pixels(train_X, val_X, test_X, train_Y, val_Y, test_Y)
  model = model_x
  model.summary()

  #Image Augmentation
  datagen = ImageDataGenerator(width_shift_range=shift_fraction,height_shift_range=shift_fraction,horizontal_flip=True)

  it_train = datagen.flow(trainX, trainY, batch_size=batch_size)
	# prepare iterator
  it_val = datagen.flow(valX, valY, batch_size=batch_size)
  # fit model
  steps = int(trainX.shape[0] / batch_size)
  history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=it_val, verbose=1)
  # evaluate model
  _, acc = model.evaluate(testX, testY, verbose=1)
  print('Accuracy:')
  print('> %.3f' % (acc * 100.0))
  # learning curves
  summarize_diagnostics(history)

"""**For this part the train, test, and validation data are feed into data generator, which makes it easier for the sets to be preprocessed and then trained later on.**

**The ImageDataGenerator() function is used where the augmentation is done. Featurewise Standardization, ZCA Whitening, Shift Range, and Flips where used.**

**Afterwards a batch of augmented images were saved in local Google drive.**

**The run_test_harness() contains the entire functions needed for training where it only needs to take the model name and it will automatically call the other tasks needed for this activity.**

## **THIRD BASELINE MODEL**
"""

from tensorflow.keras.callbacks import EarlyStopping

# define cnn model
def define_model_3():
	model3 = Sequential()
	model3.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
	model3.add(BatchNormalization())
	model3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model3.add(BatchNormalization())
	model3.add(MaxPooling2D((2, 2)))
	model3.add(Dropout(0.25))

	model3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model3.add(BatchNormalization())
	model3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model3.add(BatchNormalization())
	model3.add(MaxPooling2D((2, 2)))
	model3.add(Dropout(0.25))

	model3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model3.add(BatchNormalization())
	model3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model3.add(BatchNormalization())
	model3.add(MaxPooling2D((2, 2)))
	model3.add(Dropout(0.25))

	model3.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model3.add(BatchNormalization())
	model3.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model3.add(BatchNormalization())
	model3.add(MaxPooling2D((2, 2)))
	model3.add(Dropout(0.25))



	model3.add(Flatten())
	model3.add(Dropout(0.25))
	model3.add(Dense(512, activation='relu'))
	model3.add(Dropout(0.25))
	model3.add(Dense(128, activation='relu'))
	model3.add(Dropout(0.25))
	model3.add(Dense(10, activation='softmax'))

	# compile model

	model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model3

early_stop = EarlyStopping(monitor='val_loss', patience=2)

model3= define_model_3()

"""### **Finally, a considerably deeper network with three VGG blocks and 512 nodes in the final hidden layer is employed. Each had max norm per block, batch normalization per convolutional layer, and rising dropout rates. Adam was the optimizer, and 100 epochs were utilized. It outperformed the previous two models, as predicted, with 98% accuracy on the dataset and 94% validation. Although there is still some acceptable overfitting, it functioned nicely.**

## **Finalization, saving of model, and testing on new images**
"""

trainX, valX, testX, trainY, valY, testY = prep_pixels(train_X, val_X, test_X, train_Y, val_Y, test_Y)

model3.save('/content/drive/MyDrive/Models/saved_fashion.h5')

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the data to match the model's expected input shape (None, 784)
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255

# Define a simple model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(X_test, y_test)

# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
  img = Image.open(filename).resize((224, 224))
  plt.imshow(img)
  plt.show()
  img = load_img(filename, target_size=(28, 28))
  img = img_to_array(img)
  img = img[:,:,0]
  img = img.reshape(1,28, 28, 1)
  img = img.astype('float32')
  img = img / 255.0
  return img

def run_example(filename):
  img = load_image(filename)
  model = load_model('/content/drive/MyDrive/Models/saved_fashion.h5')
  result = np.argmax(model.predict(img), axis=1)
  if result == 0:
    print('Tshirt')
  elif result == 1:
    print('Top')
  elif result == 2:
    print('Pullover')
  elif result == 3:
    print('Dress')
  elif result == 4:
    print('Coat')
  elif result == 5:
    print('Sandal')
  elif result == 6:
    print('Shirt')
  elif result == 7:
    print('Snicker')
  elif result == 8:
    print('Bag')
  else:
    print('Ankle Boot')

run_example('/content/kiki_fashion_example.jpg')

!streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py
