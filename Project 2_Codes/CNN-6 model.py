
# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# Extract and Reshape the 3 Bands
def get_data(df):
    bands = []
    
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = (band_1 + band_2)/2 # Adding Channel 3 for converting to RGB equvivalent
        
        bands.append(np.dstack((band_1, band_2, band_3)))
        
    return np.array(bands)

# Extract the normalization parameters
def get_norm_params(data):
    
    return [
           (data[:,:,:,0].min(), data[:,:,:,0].max(), data[:,:,:,0].mean()), 
           (data[:,:,:,1].min(), data[:,:,:,1].max(), data[:,:,:,1].mean()),
           (((data[:,:,:,0] + data[:,:,:,1])/2).min(), 
           ((data[:,:,:,0] + data[:,:,:,1])/2).max(), 
           ((data[:,:,:,0] + data[:,:,:,1])/2).mean())
           ]


# Normalize the Trainig dataset
def normalize(data, params):
    
    p1 = params[0] # p = (min, max, mean)
    p2 = params[1]
    p3 = params[2]
    
    data[:, :, :, 0] = (data[:, :, :, 0] - p1[2]) / (p1[1] - p1[0])
    data[:, :, :, 1] = (data[:, :, :, 1] - p2[2]) / (p2[1] - p2[0])
    data[:, :, :, 2] = (data[:, :, :, 2] - p3[2]) / (p3[1] - p3[0])

    return data



#Plotting:
def show_all_bands(data, current):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(data[current, :, :, 0])
    ax1.set_title("Band 1")

    ax2.imshow(data[current, :, :, 1])
    ax2.set_title("Band 2")
    
    ax3.imshow(data[current, :, :, 2])
    ax3.set_title("Band 3")

    plt.show()



# Configure Neural Network Model
def get_model():
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# Importing the data
df_train = pd.read_json('train.json')  # load pd dataframe
Y = np.array(df_train['is_iceberg'])  # extract labels
X = get_data(df_train)  # convert to [1603, 75, 75, 3]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

params = get_norm_params(X_train)  # get mean, max and min from TRAINING data
                                           # then normalize TEST data using these parameters
                                           # to prevent data-leak

X_train = normalize(X_train, params=params)
X_test = normalize(X_test, params=params)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

show_all_bands(X, 150)

# Defining the attributes for the CNN model
batch_size = 32
epochs = 30

# Data Augmentation
datagen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0, 
                         height_shift_range = 0, 
                         channel_shift_range=0, 
                         zoom_range = 0.1,
                         rotation_range = 10)

# Training the CNN model
model = get_model()

# Print summary representation of the CNN model
model.summary() 

#Reduce learning rate when a metric has stopped improving
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, 
                                   verbose=1, epsilon=1e-4, mode='min')

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=len(X_train)/batch_size, epochs=epochs, callbacks=[reduce_lr_loss],
                            validation_data=(X_test,Y_test))


#Plot learning process
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('accuracy')
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


# Predicting the Test Data
del X_test, Y_test, X_train, Y_train
df_test = pd.read_json('test.json')
ID = df_test['id']
X = get_data(df_test)
X = normalize(X, params)
print(X[:,:,:,1].mean())
pred = model.predict_proba(X)
print(pred)

# Generating the Submission File
submission = pd.DataFrame()
submission['id'] = ID
submission['is_iceberg'] = pred
submission.to_csv('Project 2_sub.csv', index=False)