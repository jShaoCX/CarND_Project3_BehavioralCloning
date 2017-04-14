import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import csv
import random
from sklearn.utils import shuffle

AUG_BATCH_SIZE = 1
ave_bright = 128 #128
std_bright = 32 #should be around 10
'''
def data_aug_generator(X_orig,y_orig):
    aug_X = []
    aug_y = []
    #num_batches = int(len(y_orig)/AUG_BATCH_SIZE)
    #print(num_batches)
    #for i in range(0, num_batches):
        #for j in range(i*AUG_BATCH_SIZE, (i+1)*AUG_BATCH_SIZE):
    for i in range(0,len(X_orig)):
        currentIm = X_orig[i]
        currentSteer = y_orig[i]
        idx = random.randint(0,2)
        #rotate
        if idx == 0:
            angle = 2 * random.randint(-3, 3)
            M = cv2.getRotationMatrix2D((160, 80), angle, 1)
            newImg = np.array(cv2.warpAffine(currentIm, M,(320,160)))
            aug_X.append(newImg)
            aug_y.append(currentSteer)
        #shift
        elif idx == 1:
            shiftX = random.randint(-5, 5)
            M = np.float32([[1,0,shiftX],[0,1,0]])
            newImg = np.array(cv2.warpAffine(currentIm, M,(320,160)))
            aug_X.append(newImg)
            newSteer = 0.3*shiftX + currentSteer
            aug_y.append(newSteer) #may need to adjust steering??
        #brightness
        elif idx==2:
            newImg = currentIm
            newImg = np.array(newImg, dtype=np.float32)
            rand_bright = np.random.normal(ave_bright, std_bright)/255.0
            newImg[:,:,1] = currentIm[:,:,1]*rand_bright
            newImg[:,:,1][newImg[:,:,1] > 175] = 175
            newImg[:, :, 1][newImg[:, :, 1] < 75] = 75
            newImg = np.array(newImg,dtype=np.uint8)
            aug_X.append(newImg)
            aug_y.append(currentSteer)
        else:
            aug_X.append(currentIm)
            aug_y.append(currentSteer)
    #aug_X = np.asarray(aug_X)
    #aug_y = np.asarray(aug_y)
    while 1:
        for j in range(284): # 1875 * 32 = 60000 -> # of training samples
            if i%100==0:
                print("i = " + str(j))
            yield np.asarray(aug_X[j*32:(j+1)*32]), np.asarray(aug_y[j*32:(j+1)*32])
        #yield np.asarray(aug_X), np.asarray(aug_y)
'''

def data_aug(X_orig,y_orig):
    currentIm = X_orig
    currentSteer = y_orig
    idx = random.randint(0,2)
        #rotate
    if idx == 0:
        angle = 2 * random.randint(-3, 3)
        M = cv2.getRotationMatrix2D((160, 80), angle, 1)
        newImg = np.array(cv2.warpAffine(currentIm, M,(320,160)))
        aug_X=newImg
        aug_y=currentSteer
        #shift
    elif idx == 1:
        shiftX = random.randint(-5, 5)
        M = np.float32([[1,0,shiftX],[0,1,0]])
        newImg = np.array(cv2.warpAffine(currentIm, M,(320,160)))
        aug_X=newImg
        newSteer = 0.3*shiftX + currentSteer
        aug_y=newSteer
        #brightness
    else:
        newImg = currentIm
        newImg = np.array(newImg, dtype=np.float32)
        rand_bright = np.random.normal(ave_bright, std_bright)/255.0
        newImg[:,:,1] = currentIm[:,:,1]*rand_bright
        newImg[:,:,1][newImg[:,:,1] > 175] = 175
        newImg[:, :, 1][newImg[:, :, 1] < 75] = 75
        newImg = np.array(newImg,dtype=np.uint8)
        aug_X=newImg
        aug_y=currentSteer
    return aug_X, aug_y




lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print(len(lines))

'''
images = []
measurements = []

from collections import Counter

myCounter = Counter()
for i in range(0,len(lines)):
    path = './data/' + lines[i][0] #center image
    measurement = float(lines[i][3])
    if measurement == 0:
        if i%2==0 or i%3 == 0 or i%5 == 0 or i%7 == 0 or i%11==0:
            pass
        else:
            imageBGR = cv2.imread(path)
            image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YUV)
            images.append(image)
            measurements.append(measurement)
            myCounter[measurement] += 1
    else:
        imageBGR = cv2.imread(path)
        image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YUV)
        images.append(image)
        im_flip = np.fliplr(image)
        images.append(im_flip)
        measurements.append(measurement)
        measurements.append(-measurement)
        myCounter[measurement] += 1
        myCounter[-measurement] += 1
'''

def myGenerator(my_lines, batch_size):


    while True:
        #my_lines = shuffle(my_lines)
        count=0
        batch_features = np.zeros((batch_size, 160, 320, 3))
        batch_labels = np.zeros((batch_size, 1))
        while count in range(0, batch_size):
            randIdx = random.randint(0, len(my_lines) - 1)
            path = './data/' + my_lines[randIdx][0]  # center image
            measurement = float(my_lines[randIdx][3])
            if measurement == 0:
                if randIdx % 2 == 0 or randIdx % 3 == 0 or randIdx % 5 == 0 or randIdx % 7 == 0 or randIdx % 11 == 0:
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

#print(myCounter.most_common()[:30])


'''
X_all = np.array(images)
y_all = np.array(measurements)

X_all, y_all = shuffle(X_all, y_all)

X_train = X_all[0:int(0.8*len(X_all))]
X_valid = X_all[int(0.8*len(X_all))+1:len(X_all)-1]

y_train = y_all[0:int(0.8*len(y_all))]
y_valid = y_all[int(0.8*len(y_all))+1:len(y_all)-1]

#mygen = data_aug_generator(X_train, y_train)
#for i in range(0,909):
#    temp = next(mygen)
#    print(temp[0].shape)

#need average brightness of images
#print(np.mean(X_train[:,:,:,1])) #128
#print(np.std(X_train[:,:,:,1])) #10
'''

from keras.models import Sequential
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Dense, Lambda, Cropping2D, Activation, Dropout

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, border_mode='same', activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, border_mode="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Convolution2D(48, 5, 5, border_mode="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_all, y_all, validation_split=0.2,shuffle=True, epochs=3)
model.fit_generator(myGenerator(lines, 32),samples_per_epoch=284, epochs=4)
#model.fit_generator(myGenerator(lines),samples_per_epoch=9088, epochs=2, validation_data=(X_valid,y_valid))
model.save('model.h5')