
# coding: utf-8

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Dense, Dropout, Flatten,Conv2D,Lambda
from keras.optimizers import adam
import cv2
import os
import csv
import numpy as np
import sklearn


# ## Build the generator

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# ## Preprocess the data
samples = []
fliped_samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
for sample in samples:
#     print(sample[3])
    if float(sample[3]) > 0.05:
        flip_line = ['','','',0,0,0,0]
        name = './data/'+sample[0]
        image = cv2.imread(name)
        image_fliped = np.fliplr(image)
        flip_name = 'IMG/fliped_'+sample[0].split('/')[-1]
        cv2.imwrite('./data/'+flip_name,image_fliped)
        angle_fliped = float(sample[3])*-1.
        flip_line[0] = flip_name
        flip_line[3] = angle_fliped
        fliped_samples.append(flip_line)
all_samples = np.concatenate((samples, fliped_samples), axis=0)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(all_samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

# ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Conv2D(24, (5, 5),strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5),strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5),strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


# ## Train the model
model.fit_generator(train_generator, steps_per_epoch= /
            len(train_samples)/128, validation_data=validation_generator, /
            validation_steps=len(validation_samples)/128, nb_epoch=3)


# ## Save the model
model.save('./model.h5')

