import os
import numpy as np
from scipy import misc
from keras import backend as K
import matplotlib.pyplot as plt


def imgload(height, width, percorso):
    np.random.seed(1338)
    #############################################
    # imposto il formato delle immagini da importare (tensore 3x3: height x weight x channel)
    i_width = width
    i_height = height
    if K.image_data_format() == 'channel_first':
        input_shape = (1, i_height, i_width)
    else:  # channel last
        input_shape = (i_height, i_width, 1)

    #############################################

    path = percorso

    # estraggo i nomi delle classi dai nomi delle cartelle presenti nel percorso indicato
    classes = [name for name in os.listdir(path) if os.path.isdir(path + os.sep + name)]
    # print(classes)

    # genero tensore vuoto da riempire con tutte le immagini formattate delle 3 classi
    # e tensore delle labels corrispondenti (rispettivi indici)
    X_train = np.empty((0,) + input_shape)
    Y_train = np.empty((0,))

    X_test = np.empty((0,) + input_shape)
    Y_test = np.empty((0,))

    ##################################  INIZIO CICLO  #################################################
    # ciclo sulle 3 cartelle
    for emotion in classes:

        imlist = []
        lblist = []

        # ciclo su tutti i file conenuti in ogni cartella
        for fname in sorted(os.listdir(path + os.sep + emotion)):
            img = misc.imread(path + os.sep + emotion + os.sep + fname).astype('uint8')
            # print(img.shape)

            # alcune immagini erano 114x114: ridimensiono tutte le img in un 120x120
            if img.shape != (i_height, i_width):
                img = misc.imresize(img, (i_height, i_width))
                if img.shape != (i_height, i_width):
                    img = img[:, :, 0]

            label = classes.index(emotion)
            # print(label)

            imlist.append(img)
            lblist.append(label)
            # print(img.shape)

        # trasformo la lista di array in formato numpy (tensore)
        data_set = np.stack(imlist).astype('uint8')
        label_set = np.stack(lblist).astype('uint8')

        # correggo formato tensore (ho bisogno di num_immagini x height x weight x channel)
        data_set = data_set.reshape((data_set.shape[0],) + input_shape)
        # print('-> ' + emotion + ' faces tensor shape: ', data_set.shape)

        # seleziono 1000 elementi per ogni categoria da mettere nel TRAINING set
        # (i restanti 300 elementi per classe andranno nel TEST set)
        indices = np.full(data_set.shape[0], False, bool)
        randices = np.random.choice(np.arange(indices.shape[0]), 1000, replace=False)
        indices[randices] = True

        # genere i vettori di immagini e label er il TRAINING e TEST set
        X_train = np.append(X_train, data_set[randices], axis=0)
        Y_train = np.append(Y_train, label_set[randices])

        X_test = np.append(X_test, data_set[~indices], axis=0)
        Y_test = np.append(Y_test, label_set[~indices])

    ##################################  FINE CICLO  #################################################
    # normalizzo i valori dei pixel in modo che ogni dimensione abbia circa lo stesso ordine di grandezza
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print('>>> Training set has shape: ', X_train.shape, ' --- Training labels: ', Y_train.shape[0])
    print('>>> Test set has shape: ', X_test.shape, ' --- Test labels: ', Y_test.shape[0])

    ###########################################################
    # stampo un esempio di immagine per ogni classe
    # plt.close('all')
    # fig = plt.figure(figsize=(8, 3))
    # for i in range(len(classes)):
    #     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    #     features_idx = X_train[Y_train[:] == i, :]
    #     # print(features_idx)
    #     ax.set_title("Class: " + classes[i])
    #     plt.imshow(features_idx[1].reshape(120, 120), cmap="gray")
    # plt.show()
    ###########################################################

    return K, classes, input_shape, X_train, X_test, Y_train, Y_test


def plot_model_history(model_history):
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'validation'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'validation'], loc='best')
    plt.show()
