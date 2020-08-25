import logging

from tensorflow import keras
from sklearn.model_selection import KFold
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.optimizers import *
from keras.losses import *
import numpy as np
import os

mpl.use('Qt5Agg')


class Regressor_Simple:
    def __init__(self, function = None, NUMBER_OF_FOLDS=5, BATCH_SIZE=32, EPOCHS=6,
                 LOSS_FUNCTION=mean_squared_error, NUMBER_OF_SAMPLES=20000,
                 train_low=[0.1], train_high=[2], plot_low=[0], plot_high=[4],
                 NUMBER_OF_SAMPLE_FOR_PLOT=20000 * 10, NUMBER_OF_FEAUTRES=2):
        self.NUMBER_OF_FOLDS = NUMBER_OF_FOLDS
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LOSS_FUNCTION = LOSS_FUNCTION
        self.NUMBER_OF_SAMPLES = NUMBER_OF_SAMPLES
        self.train_low = train_low
        self.train_high = train_high
        self.plot_low = plot_low
        self.plot_high = plot_high
        self.function = function
        self.NUMBER_OF_SAMPLE_FOR_PLOT = NUMBER_OF_SAMPLE_FOR_PLOT
        self.X = None
        self.y = None
        self.model = None
        self.NUMBER_OF_FEAUTRES = NUMBER_OF_FEAUTRES

    def initial_dataset_maker(self):
        X = np.random.uniform(self.train_low, self.train_high, (self.NUMBER_OF_SAMPLES, self.NUMBER_OF_FEAUTRES))
        y = self.function(*X)
        p = np.random.permutation((self.NUMBER_OF_SAMPLES))
        self.X = X[p]
        self.y = y[p]

    def train(self):
        kfold = KFold(n_splits=self.NUMBER_OF_FOLDS, shuffle=True)

        loss_per_fold = []
        all_models = []
        fold_no = 1
        for train, test in kfold.split(self.X, self.y):
            model = keras.Sequential()
            model.add(keras.layers.Dense(128, input_dim=self.NUMBER_OF_FEAUTRES, kernel_initializer='normal',
                                         activation='relu'))
            model.add(keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))
            model.add(keras.layers.Dense(512, kernel_initializer='normal', activation='exponential'))
            model.add(keras.layers.Dense(512, kernel_initializer='normal', activation='relu'))
            model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='linear'))

            model.compile(loss=self.LOSS_FUNCTION, optimizer='adam')

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
        myRange = np.random.uniform(self.plot_low, self.plot_high,
                                    (self.NUMBER_OF_SAMPLE_FOR_PLOT, self.NUMBER_OF_FEAUTRES))
        actualZ = self.function(*myRange)
        myZ = self.model.predict(myRange)
        actualZ = actualZ.reshape(self.NUMBER_OF_SAMPLE_FOR_PLOT)
        myZ= myZ.reshape(self.NUMBER_OF_SAMPLE_FOR_PLOT)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(myRange[:, 0], myRange[:, 1], myZ, alpha=0.6)
        ax.plot_trisurf(myRange[:, 0], myRange[:, 1], actualZ, alpha=0.5)
        return fig


def test_func(*args):
    arr = np.array(args)
    return np.sin(arr[:, 0]) + np.sin(arr[:, 1])


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_KERAS'] = '1'


regressor = Regressor_Simple(test_func, NUMBER_OF_SAMPLES=2000, NUMBER_OF_SAMPLE_FOR_PLOT=4000,
                             train_low=[-np.pi, -np.pi], train_high=[np.pi, np.pi],
                             plot_low=[-2 * np.pi, -2 * np.pi],
                             plot_high=[2 * np.pi, 2 * np.pi])
