from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import tensorflow as tf
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from keras.regularizers import l2
import numpy as np
from art.attacks import DeepFool
from art.classifiers import KerasClassifier
from art.utils import load_dataset

def evaluate(clf, x_train, y_train, x_test, y_test):
    # train acc
    train_preds = np.argmax(clf.predict(x_train), axis=1)
    train_acc = np.sum(train_preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
    # test acc
    test_preds = np.argmax(clf.predict(x_test), axis=1)
    test_acc = np.sum(test_preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTraining and Test accuracies: %.2f %.2f%%" % (train_acc*100, test_acc*100))
    return train_acc, test_acc

def plot_curves(model, title='model accuracy and loss'):
    import matplotlib.pyplot as plt
    plt.plot(model.history.history['acc'])
    plt.plot(model.history.history['loss'])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    return plt

def save_clf(c, folder, clf_filename, model_filename):
    cwd = os.getcwd()
    os.chdir(folder)
    import pickle
    with open(clf_filename, "wb") as f:
        pickle.dump(c.__getstate__, f)
    c._model.save(model_filename)
    os.chdir(cwd)

DUMP_FOLDER = '/home/surthi/models/'
def pickle_dump(data, filename, folder=DUMP_FOLDER):
    cwd = os.getcwd()
    os.chdir(folder)
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    os.chdir(cwd)

def pickle_load(filename):
    import pickle
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def load_clf(folder, clf_filename, model_filename):
    cwd = os.getcwd()
    os.chdir(folder)
    import pickle
    with open(clf_filename, "rb") as f:
        clf_state = pickle.load(f)
    model = tf.keras.models.load_model(model_filename)
    os.chdir(cwd)
    clf = KerasClassifier(model=model)
    clf.__setstate__(clf_state())
    return clf, model
