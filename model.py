import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import csv
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

ave_bright = 128 #128
std_bright = 32 #should be around 10

def data_aug(X_orig,y_orig):
    currentIm = X_orig
    currentSteer = y_orig

    #effort to make brightness insignificant by setting all to same number
    #Note: tried brightness augmentation but it didn't help significantly
    #remnants of that code can be seen commented out
    newImg = currentIm
    #newImg = np.array(newImg, dtype=np.float32)
    #rand_bright = np.random.normal(ave_bright, std_bright)/255.0
    newImg[:,:,1] = 128
    #newImg[:,:,1][newImg[:,:,1] > 191] = 191
    #newImg[:, :, 1][newImg[:, :, 1] < 64] = 64
    aug_bright_im = np.array(newImg)#,dtype=np.uint8)
    idx = random.randint(0,1)
        #rotate
    if idx == 0:
        angle = 2 * random.randint(-3, 3)
        M = cv2.getRotationMatrix2D((160, 80), angle, 1)
        newImg = np.array(cv2.warpAffine(aug_bright_im, M,(320,160)))
        aug_X=newImg
        aug_y=currentSteer
        #shift
    else:
        shiftX = 2*random.randint(-5, 5)
        M = np.float32([[1,0,shiftX],[0,1,0]])
        newImg = np.array(cv2.warpAffine(aug_bright_im, M,(320,160)))
        aug_X=newImg
        newSteer = -0.0036*shiftX + currentSteer
        aug_y=newSteer
    return aug_X, aug_y



#read in the csv file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = shuffle(lines)

#training validation split
lines_train = lines[0:int(0.8*len(lines))]
lines_valid = lines[int(0.8*len(lines))+1:len(lines)-1]

print(len(lines))
print(len(lines_train))
print(len(lines_valid))

#read in validation set
valid_images = []
valid_measurements = []
for i in range(0,len(lines_valid)):
    path = './data/' + lines_valid[i][0] #center image
    measurement = float(lines_valid[i][3])
    imageBGR = cv2.imread(path)
    image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YUV)
    valid_images.append(image)
    valid_measurements.append(measurement)
X_valid = np.array(valid_images)
y_valid = np.array(valid_measurements)

'''
all_images = []
all_measurements = []
for i in range(0,len(lines)):
    path = './data/' + lines[i][0] #center image
    measurement = float(lines[i][3])
    imageBGR = cv2.imread(path)
    image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YUV)
    all_images.append(image)
    all_measurements.append(measurement)
X_train = np.array(all_images)
y_train = np.array(all_measurements)
'''


#getting a sense of how unbalanced the data is

from collections import Counter

my_data = []

myCounter = Counter()
for i in range(0,len(lines)):
    path = './data/' + lines[i][0] #center image
    measurement = float(lines[i][3])
    if measurement == 0:
        if i%2==0 or i%3 == 0 or i%5 == 0 or i%7 == 0:
            pass
        else:
            myCounter[measurement] += 1
            my_data.append(measurement)
    else:
        myCounter[measurement] += 1
        myCounter[-measurement] += 1
        my_data.append(measurement)
        my_data.append(-measurement)

print(myCounter.most_common()[:30])

print(len(my_data))

from matplotlib.pyplot import savefig

labels =  [entry[0] for entry in myCounter.most_common()]
frequencies = [entry[1] for entry in myCounter.most_common()]

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

y = frequencies
N = len(y)
x = labels
width = 0.01
plt.bar(x, y, width, color="blue")
ax.set_title('Testing Data Distribution')
ax.set_ylabel('Frequency')
ax.set_xlabel('Steering Angle')
fig = plt.gcf()
savefig('training_data_graph.png')

def myGenerator(my_lines, batch_size):
    while True:
        count=0
        batch_features = np.zeros((batch_size, 160, 320, 3))
        batch_labels = np.zeros((batch_size, 1))
        while count in range(0, batch_size):
            randIdx = random.randint(0, len(my_lines) - 1)
            path = './data/' + my_lines[randIdx][0]  # center image
            measurement = float(my_lines[randIdx][3])
            if measurement == 0:
                if randIdx % 2 == 0 or randIdx % 3 == 0 or randIdx % 5 == 0 or randIdx % 7 == 0:
                    pass
                else:
                    imageBGR = cv2.imread(path)
                    image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YUV)
                    aug_X, aug_y = data_aug(image, measurement)
                    batch_features[count] = aug_X
                    batch_labels[count] = aug_y
                    count+=1
            else:
                imageBGR = cv2.imread(path)
                image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YUV)

                if(randIdx%2==0):
                    aug_X, aug_y = data_aug(image, measurement)
                    batch_features[count] = aug_X
                    batch_labels[count] = aug_y
                    count+=1
                else:
                    im_flip = np.fliplr(image)
                    aug_X, aug_y = data_aug(im_flip, -measurement)
                    batch_features[count] = aug_X
                    batch_labels[count] = aug_y
                    count+=1
        yield batch_features,batch_labels


#need distribution of brightness of images
#print(np.mean(X_valid[:,:,:,1])) #128
#print(np.std(X_valid[:,:,:,1])) #10

''''
from keras.models import Sequential
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Dense, Lambda, Cropping2D, Activation, Dropout, AveragePooling2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape=(160,320,3)))
model.add(AveragePooling2D())
model.add(Lambda(lambda x: x/128.0 - 1.0))
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
#model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(X_train, y_train, validation_split=0.2,shuffle=True, epochs=5)
history_object = model.fit_generator(myGenerator(lines_train, 64),samples_per_epoch=115, epochs=6, validation_data=(X_valid,y_valid), verbose=1)
model.save('model.h5')

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
'''