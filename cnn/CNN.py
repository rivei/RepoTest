import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K

# Loading the training and testing data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test_orig = X_test
print(X_train.shape, X_train.dtype)
print(y_train[0:10])

# IMPORTANT TO HANDLE image_data_format
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channel_first':
    shape_ord = (1, img_rows, img_cols)
else:  # channel last
    shape_ord = (img_rows, img_cols, 1)

# Preprocess and reshape data
X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
X_test = X_test.reshape((X_test.shape[0],) + shape_ord)
print(X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

np.random.seed(1338)

# Test data
X_test = X_test.copy()
Y = y_test.copy()

# Converto il vettore test dei labels in binary (six=1, not six=0)
Y_test = Y == 6
Y_test = Y_test.astype(int)
print(Y_test)

# Seleziono gli elementi uguali a 6
X_six = X_train[y_train == 6].copy()
Y_six = y_train[y_train == 6].copy()
print(X_six.shape[0])

# Seleziono gli elementi diversi da 6
X_not_six = X_train[y_train != 6].copy()
Y_not_six = y_train[y_train != 6].copy()
print(X_not_six.shape[0])

# Seleziono 6000 elementi casuali tra quelli diversi da 6
random_rows = np.random.randint(0, X_six.shape[0], 6000)
X_not_six = X_not_six[random_rows]
Y_not_six = Y_not_six[random_rows]

# Appending the data with output as 6 and data with output as <> 6
X_train = np.append(X_six, X_not_six)

# Reshaping the appended data to appropriate form
X_train = X_train.reshape((X_six.shape[0] + X_not_six.shape[0],) + shape_ord)

# Appendo le labels e le converto in classi binarie (six=1, not six=0)
Y_labels = np.append(Y_six, Y_not_six)
Y_train = Y_labels == 6
Y_train = Y_train.astype(int)

print(X_train.shape, Y_labels.shape, X_test.shape, Y_test.shape)

# Converto le classi nella loro forma categorica binaria
nb_classes = 2
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# -- Inizializzo i valori per il CNN

nb_epoch = 10
batch_size = 64

# numero di filtri convoluzionali
nb_filters = 32
# dimensione dell'area di max-pooling
nb_pool = 2
# dimensione del conv kernel
nb_kernel = 3

# Vanilla SGD(learnign rate, lr decay over eanch update, parameter updates momentum, nesterov momentum)
sgd = SGD(lr=0.1, decay=1e-09, momentum=0.9, nesterov=True)

#  ---- MODEL DEFINITION ---
model = Sequential()

model.add(Conv2D(nb_filters, (nb_kernel, nb_kernel), padding='valid', input_shape=shape_ord))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# ---- COMPILE ----
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# ---- FIT ----
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
