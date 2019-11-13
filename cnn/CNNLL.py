#Model number
namebm=14

#Libraries
import numpy as np
from guardar import saveparreduced
from functions_trainset_seperation import imgload, plot_model_history
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, ZeroPadding1D
from keras.optimizers import SGD
import os
#Load images
K, classes, input_shape, X_train, X_test, Y_train, Y_test= imgload(120, 120, 'C:/Users/laura/Downloads/Datasets')           


#Convolutional Neural Networks Model

namebm=1
Y_train=keras.utils.to_categorical(Y_train,num_classes)
Y_test=keras.utils.to_categorical(Y_test,num_classes)

save_dir = os.path.join(os.getcwd(), 'saved_models')#Directory where the model will be saved
model_name ='model'+ str(namebm)+'.h5'#Name of the model

model = Sequential()
#
# 1st CONV
model.add(Conv2D(6, kernel_size=(9, 9), padding='same',
                 data_format=K.image_data_format(),
                 input_shape=input_shape,
                 kernel_initializer='TruncatedNormal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#
# 2nd CONV
model.add(Conv2D(24, kernel_size=(3, 3), padding='same',
                 data_format=K.image_data_format(),
                 kernel_initializer='TruncatedNormal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#
# 3rd CONV
model.add(Conv2D(92, kernel_size=(5, 5), padding='same',
                 data_format=K.image_data_format(),
                 kernel_initializer='TruncatedNormal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#
# Fully conncected + Droopout
model.add(Flatten())
model.add(Dense(128, kernel_initializer='TruncatedNormal'))
model.add(Activation('relu'))
model.add(Dropout(0.6254654721147688))
#
# Fully connected SOFTMAX
model.add(Dense(3, kernel_initializer='TruncatedNormal'))
model.add(Activation('softmax'))


#rmsprop = RMSprop(lr=0.001)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_history = model.fit(X_train, Y_train,
                              batch_size=64,
                              epochs=30,
                              validation_data=(X_test, Y_test),
                              shuffle=True,
                              verbose=2)
a=model_history.history['acc']
av=model_history.history['val_acc']
lasta=a[len(a)-1]
lastav=av[len(av)-1]
# Save model and weights
if not os.path.isdir(save_dir):#verify if it is a directory
    os.makedirs(save_dir)#if not creates a new directory
model_path=os.path.join(save_dir,model_name)#define the model path
model.save(model_path)#save the model
print('Saved trained model at %s' %model_path)
#Score trained model
score=model.evaluate(X_test,Y_test,verbose=1)#see the score in the validation set
print('Test loss:',score[0])
print('Test accuracy:',score[1])
saveparreduced(model_name,score,lasta,lastav)
plot_model_history(hist)
plt.show()
