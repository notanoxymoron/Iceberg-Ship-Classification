
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


# Extract and Reshape the Bands
def get_data(df):
  
  bands = []
  for i, row in df.iterrows():
      band_1 = np.array(row['band_1']).reshape(75, 75)
      band_2 = np.array(row['band_2']).reshape(75, 75)
      
      bands.append(np.dstack((band_1, band_2)))
  
  return np.array(bands)


#Plotting:
def show_all_bands(data, current):
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(data[current, :, :, 0])
    ax1.set_title("Band 1")

    ax2.imshow(data[current, :, :, 1])
    ax2.set_title("Band 2")
    
    plt.show()
    
    
def train():
  model = Sequential()
  
  # CNN 1
  model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(75,75,2)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  
  # CNN 2
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.25))
  
  # CNN 3
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.25))
  
  model.add(Flatten())
  
  #Dense 1
  model.add(Dense(48, activation='relu'))
  model.add(Dropout(0.5))
  
  #Dense 2
  model.add(Dense(24, activation='relu'))
  model.add(Dropout(0.5))
  
  #Output
  model.add(Dense(1, activation='sigmoid'))

  optimizer = Adam(lr=0.001, decay=0.0)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
  return model


# Importing the data
df_train = pd.read_json('train.json')
Y = np.array(df_train['is_iceberg'])  # extract labels
X = get_data(df_train)  # convert to [1603, 75, 75, 2]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(X_train.shape[0], 'train samples')
print(X_train.shape)
print(Y_train.shape)

print(X_test.shape[0], '\ntest samples')
print(X_test.shape)
print(Y_test.shape)

show_all_bands(X_train, 150)

# Defining the attributes for the CNN model
batch_size = 32
epochs = 20
  
# Training the CNN model
model = train()

# Print summary representation of the CNN model
model.summary() 

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, Y_test))


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
print(X[:,:,:,1].mean())
y_pred = model.predict_proba(X)
print(y_pred)


# Generating the Submission File
print ("Generating Submission File")
submission = pd.DataFrame()
submission['id'] = ID
submission['is_iceberg'] = y_pred
submission.to_csv('Project 2_sub.csv', index=False)
