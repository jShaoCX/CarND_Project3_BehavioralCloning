import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import csv

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print(len(lines))

images = []
measurements = []

from collections import Counter

myCounter = Counter()
for i in range(1,len(lines)):
    path = './data/' + lines[i][0] #center image

    measurement = float(lines[i][3])
    if measurement == 0:
        if i%2==0 or i%3 == 0 or i%5 == 0:
            pass
        else:
            image = cv2.imread(path)
            images.append(image)
            measurements.append(measurement)
            myCounter[measurement] += 1
    else:
        image = cv2.imread(path)
        images.append(image)
        im_flip = np.fliplr(image)
        images.append(im_flip)
        measurements.append(measurement)
        measurements.append(-measurement)
        myCounter[measurement] += 1
        myCounter[-measurement] += 1

print(myCounter.most_common())


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Dense, Lambda, Cropping2D, Activation

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, border_mode='same', activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, border_mode="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, border_mode="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=4)

model.save('model.h5')
