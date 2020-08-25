import logging

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.optimizers import *
from keras.losses import *
import numpy as np
import os


class Image_Classifier:
    def __init__(self, NUMBER_OF_FOLDS=5, BATCH_SIZE=32, EPOCHS=6,
                 LOSS_FUNCTION=sparse_categorical_crossentropy,
                 ):
        self.NUMBER_OF_FOLDS = NUMBER_OF_FOLDS
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LOSS_FUNCTION = LOSS_FUNCTION
        self.images = None
        self.labels = None
        self.model = None

    def load_mnist(self):
        MNIST_DS = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = MNIST_DS.load_data()

        train_images = train_images / 255
        test_images = test_images / 255

        images = np.concatenate((train_images, test_images), axis=0)
        labels = np.concatenate((train_labels, test_labels), axis=0)

        self.images = images
        self.labels = labels

    def train(self):
        kfold = KFold(n_splits=self.NUMBER_OF_FOLDS, shuffle=True)

        accuracy_per_fold = []
        loss_per_fold = []
        fold_no = 1
        all_models = []
        for train, test in kfold.split(self.images, self.labels):
            model = keras.Sequential()
            model.add(keras.layers.Flatten(input_shape=(28, 28)))
            model.add(keras.layers.Dense(28*28+2, activation='sigmoid'))
            model.add(keras.layers.Dense(10, activation='softmax'))

            model.compile(loss=self.LOSS_FUNCTION, metrics=['accuracy'], optimizer='adam')

            history = model.fit(self.images[train], self.labels[train],
                                batch_size=self.BATCH_SIZE,
                                epochs=self.EPOCHS,
                                verbose=True)

            scores = model.evaluate(self.images[test], self.labels[test], verbose=0)
            print(
                f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
            accuracy_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            all_models.append(model)
            # Increase fold number
            fold_no = fold_no + 1

        logging.info('------------------------------------------------------------------------')
        logging.info('Score per fold')
        for i in range(0, len(accuracy_per_fold)):
            logging.info('------------------------------------------------------------------------')
            logging.info(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {accuracy_per_fold[i]}%')
        logging.info('------------------------------------------------------------------------')
        logging.info('Average scores for all folds:')
        logging.info(f'> Accuracy: {np.mean(accuracy_per_fold)} (+- {np.std(accuracy_per_fold)})')
        logging.info(f'> Loss: {np.mean(loss_per_fold)}')
        logging.info('------------------------------------------------------------------------')
        best_model_index = np.argmin(loss_per_fold)
        best_model = all_models[best_model_index]
        self.model = best_model
        self.best_loss = loss_per_fold[best_model_index]
        self.avg_loss = np.mean(loss_per_fold)
        return best_model

    def image_predict(self, index):
        fig = plt.figure()
        plt.imshow(self.images[index])
        image = np.array(self.images[index]).reshape(-1, 28, 28)
        answer = self.model.predict(image)
        return fig, np.argmax(answer)


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_KERAS'] = '1'
LOSS_FUNCTION = sparse_categorical_crossentropy

clf = Image_Classifier(EPOCHS=10, NUMBER_OF_FOLDS=3)
