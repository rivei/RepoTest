from functions import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import initializers
from keras.regularizers import l1, l2
from keras.optimizers import RMSprop, Adagrad, Adam, SGD
#from keras import backend as K
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import numpy as np


np.random.seed(2017)


def data():
    ##############################################################
    K, classes, input_shape, X_train, X_test, Y_train, Y_test = imgload(
        120, 120, '/home/lorollo/Desktop/Emotion_recognition')
    ##############################################################
    Y_train = np_utils.to_categorical(Y_train, len(classes))
    Y_test = np_utils.to_categorical(Y_test, len(classes))
    ########################################################

    return X_train, Y_train, X_test, Y_test


def create_model():
    # CNN MODEL (stacking layers)
    # K, classes, input_shape, f_train, f_test, l_train, l_test = imgload(
    #     120, 120, '/home/lorollo/Desktop/Emotion_recognition')

    model = Sequential()
    #
    # 1st CONV
    model.add(Conv2D({{choice(np.arange(1, 7, dtype=int))}}, kernel_size={{choice([(5, 5), (7, 7), (9, 9)])}}, padding='same',
                     data_format='channels_last',
                     input_shape=(120, 120, 1),
                     kernel_initializer='TruncatedNormal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    #
    # 2nd CONV
    model.add(Conv2D({{choice(np.arange(1, 37, dtype=int))}}, kernel_size={{choice([(3, 3), (5, 5), (7, 7)])}}, padding='same',
                     data_format='channels_last',
                     kernel_initializer='TruncatedNormal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # 3rd CONV - If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Conv2D({{choice(np.arange(1, 129, dtype=int))}}, kernel_size={{choice([(3, 3), (5, 5)])}}, padding='same',
                         data_format='channels_last',
                         kernel_initializer='TruncatedNormal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully conncected + Droopout
    model.add(Flatten())
    model.add(Dense({{choice([32, 64, 128])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0.1, 1)}}))
    #
    # Fully connected SOFTMAX
    model.add(Dense(3))
    model.add(Activation('softmax'))

    # optimizer = RMSprop()
    # if (params['optimizer'][0] == 'rms'):
    #     optimizer = RMSprop()
    # elif (params['optimizer'][0] == 'adagrad'):
    #     optimizer = Adagrad()
    # elif(params['optimizer'][0] == 'adam'):
    #     optimizer = Adam()

    model.compile(optimizer={{choice(['rmsprop', 'adagrad', 'adam', 'sgd'])}},
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model_history = model.fit(X_train, Y_train,
                              batch_size={{choice([64, 128])}},
                              epochs=30,
                              validation_data=(X_test, Y_test),
                              shuffle=True)

    score, acc = model.evaluate(X_test, Y_test, verbose=1)
    print('Test accuracy: ', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    f = open('best_hyperparam.txt', 'a')
    f.write(str(best_run))
