# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 02:11:35 2017

@author: Raghav
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# Pre-processing
def get_data(data_frame):
    imgs = []
    
    for i, row in data_frame.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = (band_1 + band_2)/2
        
        imgs.append(np.dstack((band_1, band_2, band_3)))
        
    return np.array(imgs)

def get_norm_params(data):
    
    return [
           (data[:,:,:,0].min(), data[:,:,:,0].max(), data[:,:,:,0].mean()), 
           (data[:,:,:,1].min(), data[:,:,:,1].max(), data[:,:,:,1].mean()),
           (((data[:,:,:,0] + data[:,:,:,1])/2).min(), 
           ((data[:,:,:,0] + data[:,:,:,1])/2).max(), 
           ((data[:,:,:,0] + data[:,:,:,1])/2).mean())
           ]
    
def normalize(data, params):
    
    # p = (min, max, mean)
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]
    
    data[:, :, :, 0] = (data[:, :, :, 0] - p1[2]) / (p1[1] - p1[0])
    data[:, :, :, 1] = (data[:, :, :, 1] - p2[2]) / (p2[1] - p2[0])
    data[:, :, :, 2] = (data[:, :, :, 2] - p3[2]) / (p3[1] - p3[0])

    return data

# Plotting

def show_all_bands(data, current):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(data[current, :, :, 0])
    ax1.set_title("Band 1")

    ax2.imshow(data[current, :, :, 1])
    ax2.set_title("Band 2")
    
    ax3.imshow(data[current, :, :, 2])
    ax3.set_title("Band 3")

    plt.show()
    
#Model
def get_model():
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    
    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    #Model compilation: optimizing using adams
    model.compile(loss='binary_crossentropy',  optimizer = Adam(lr=1e-4), metrics=["accuracy"])
    return model

#Load data
df_train = pd.read_json('train.json')  # load pd dataframe
Y = np.array(df_train['is_iceberg'])  # extract labels
X = get_data(df_train)  # convert to [1603, 75, 75, 3]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


params = get_norm_params(X_train)  # get mean, max and min from TRAINING data
                                           # then normalize TEST data using these parameters
                                           # to prevent data-leak
#params
X_train = normalize(X_train, params=params)
X_test = normalize(X_test, params=params)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# Plot the pictures for particular instance
show_all_bands(data=X_train, current=150)

#Parameters:
batch_size = 32
epochs = 20
datagen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0, 
                         height_shift_range = 0,
                         channel_shift_range=0,
                         zoom_range = 0.1,
                         rotation_range = 10)
model = get_model()
model.summary() 

    
#Reduce the learning rate by 10% after every epoch
from keras.callbacks import LearningRateScheduler
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

# #Fitting the model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=len(X_train)/30, epochs=epochs, callbacks=[annealer],
                            validation_data=(X_test,Y_test))

#Plot learning process
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# Test data
df_test = pd.read_json('test.json')
ID = df_test['id']
X = get_data(df_test)
X = normalize(X, params)
print(X[:,:,:,1].mean())
pred = model.predict_proba(X)
print(pred)

submission = pd.DataFrame()
submission['id'] = ID
submission['is_iceberg'] = pred
submission.to_csv('Project 2_sub3.csv', index=False)