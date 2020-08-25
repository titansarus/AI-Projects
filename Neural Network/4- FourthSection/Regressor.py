import logging

import tensorflow
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.optimizers import *
from keras.losses import *
import numpy as np
import os
import skimage
from PIL import Image
import requests
from io import BytesIO

from keras.layers import LeakyReLU, Dense


class Regressor_Simple:
    def __init__(self, NUMBER_OF_FOLDS=5, BATCH_SIZE=32, EPOCHS=50,
                 LOSS_FUNCTION=mean_squared_error
                 , address = " " ):
        self.NUMBER_OF_FOLDS = NUMBER_OF_FOLDS
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LOSS_FUNCTION = LOSS_FUNCTION
        self.width = 0
        self.height = 0
        self.X = None
        self.y = None
        self.address = address
        self.model = None

    def vertical_scanner(self, img: np.array, col, height):
        for i in range(height):
            if (img[i, col] == 0):
                return i
        return height

    def find_first_valid(self, arr: np.array, invalid_number):
        size = len(arr)
        for i in range(size):
            if (arr[i] != invalid_number):
                return i

    def find_last_valid(self, arr: np.array, invalid_number):
        size = len(arr) - 1
        for i in range(size, -1, -1):
            if (arr[i] != invalid_number):
                return i

    def find_first_last_valid(self, arr, invalid_number):
        return self.find_first_valid(arr, invalid_number), self.find_last_valid(arr, invalid_number)

    def initial_dataset_maker(self):
        img = Image.open(self.address, 'r')
        img = np.array(img.convert('L'))
        height, width = img.shape

        self.width = width
        self.height = height

        r = np.random.randint(0, width, width // 4)
        r = np.sort(r)

        y_s = np.array(list(map(lambda x: self.vertical_scanner(img, x, height), r)))

        first, last = self.find_first_last_valid(y_s, height)
        y_s = y_s[first:last]
        r = r[first:last]

        y_s = (height - y_s)

        new_len = len(y_s)
        p = np.random.permutation((new_len))
        self.X = r[p] / width
        self.y = y_s[p] / height


    def all_points(self):
        img = Image.open(self.address, 'r')
        img = np.array(img.convert('L'))
        height, width = img.shape

        r = list(range(0, width))
        r = np.sort(r)

        y_s = np.array(list(map(lambda x: self.vertical_scanner(img, x, height), r)))

        first, last = self.find_first_last_valid(y_s, height)
        y_s = y_s[first:last]
        r = r[first:last]

        r = r
        y_s = (height - y_s)

        return r / width, y_s / height

    def train(self):
        kfold = KFold(n_splits=self.NUMBER_OF_FOLDS, shuffle=True)

        loss_per_fold = []
        all_models = []
        fold_no = 1
        for train, test in kfold.split(self.X, self.y):
            model = keras.Sequential()
            model.add(keras.layers.Dense(1, input_dim=1, activation='linear', kernel_initializer='he_uniform'))
            model.add(keras.layers.Dense(40, activation='relu', kernel_initializer='he_uniform'))
            model.add(keras.layers.Dense(40, activation='relu', kernel_initializer='he_uniform'))
            model.add(keras.layers.Dense(40, activation='relu', kernel_initializer='he_uniform'))
            model.add(keras.layers.Dense(1 , activation = "linear"))

            opt = keras.optimizers.Nadam(learning_rate=0.01)

            model.compile(optimizer=opt, loss='mse')

            history = model.fit(self.X[train], self.y[train],
                                batch_size=self.BATCH_SIZE,
                                epochs=self.EPOCHS,
                                verbose=True)

            scores = model.evaluate(self.X[test], self.y[test], verbose=0)
            loss_per_fold.append(scores)
            all_models.append(model)

            # Increase fold number
            fold_no = fold_no + 1

        logging.info('------------------------------------------------------------------------')
        logging.info('Score per fold')
        for i in range(0, len(loss_per_fold)):
            logging.info('------------------------------------------------------------------------')
            logging.info(f'> Fold {i + 1} - Loss: {loss_per_fold[i]}')
        logging.info('------------------------------------------------------------------------')
        logging.info('Average scores for all folds:')
        logging.info(f'> Loss: {np.mean(loss_per_fold)}')
        logging.info('------------------------------------------------------------------------')

        best_model_index = np.argmin(loss_per_fold)
        best_model = all_models[best_model_index]
        self.model = best_model
        self.best_loss = loss_per_fold[best_model_index]
        self.avg_loss = np.mean(loss_per_fold)
        return best_model

    def plot(self):
        fig = plt.figure()
        myRange = np.linspace(0, 1, self.width // 2)
        myY = self.model.predict(myRange)
        plt.plot(myRange, myY, '-r')
        actualx, actualY = self.all_points()
        plt.plot(actualx, actualY, '--b', alpha=0.5)
        return fig


regressor = Regressor_Simple(NUMBER_OF_FOLDS=10, EPOCHS=50, BATCH_SIZE=8)

