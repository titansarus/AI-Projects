import logging
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.losses import *
import numpy as np
import skimage
from PIL import Image, ImageFilter
from matplotlib import cm as cm

LOW = 0
MEDIUM = 1
HIGH = 2
ULTRA = 3
ULTRA_HIGH = 4


class Image_Reconstructor:
    def __init__(self, NUMBER_OF_FOLDS=5, BATCH_SIZE=32, EPOCHS=10,
                 LOSS_FUNCTION=mean_absolute_error, NUMBER_OF_IMAGES_TO_USE=10000,
                 noise_type='gaussian',
                 noise_degree=LOW
                 ):
        self.NUMBER_OF_FOLDS = NUMBER_OF_FOLDS
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LOSS_FUNCTION = LOSS_FUNCTION
        self.images = None
        self.noised_images = None
        self.NUMBER_OF_IMAGES_TO_USE = NUMBER_OF_IMAGES_TO_USE
        self.model = None
        self.noise_type = noise_type
        self.noise_degree = noise_degree

    def add_noise(self, img, mode):
        gimg = img
        if mode is not None:
            if (self.noise_degree == LOW):
                gimg = skimage.util.random_noise(img, mode=mode, clip=True)
            if (self.noise_degree == MEDIUM):
                gimg = skimage.util.random_noise(img, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
            if (self.noise_degree == HIGH):
                gimg = skimage.util.random_noise(img, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=False)
            if (self.noise_degree == ULTRA):
                gimg = skimage.util.random_noise(img, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=False)
                gimg = skimage.util.random_noise(gimg, mode='s&p', clip=False)
            if (self.noise_degree == ULTRA_HIGH):
                gimg = skimage.util.random_noise(img, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=True)
                gimg = skimage.util.random_noise(gimg, mode=mode, clip=False)
                gimg = skimage.util.random_noise(gimg, mode='s&p', clip=False)
                gimg = Image.fromarray(np.uint8(cm.gist_gray(gimg) * 255))
                gimg = gimg.filter(ImageFilter.BLUR)
                gimg = np.asarray(gimg) / 255
                gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
                gimg = gray(gimg)

        return gimg

    def load_noised_digits(self):
        MNIST_DS = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = MNIST_DS.load_data()
        train_images = train_images / 255
        test_images = test_images / 255
        images = np.concatenate((train_images, test_images), axis=0)

        p = np.random.permutation((len(images)))
        images = images[p]
        images = images[0:self.NUMBER_OF_IMAGES_TO_USE]
        noised_images = np.array(list(map(lambda x: self.add_noise(x, self.noise_type), images))).reshape(
            -1, 28, 28)
        self.images = images
        self.noised_images = noised_images

    def train(self):
        kfold = KFold(n_splits=self.NUMBER_OF_FOLDS, shuffle=True)

        loss_per_fold = []
        fold_no = 1
        all_models = []
        for train, test in kfold.split(self.noised_images, self.images):
            model = keras.Sequential()
            model.add(keras.layers.Dense(784, activation='relu'))
            model.add(keras.layers.Dense(784 * 2, activation='sigmoid'))
            model.add(keras.layers.Dense(784, activation='sigmoid'))

            model.compile(loss=self.LOSS_FUNCTION, optimizer='adam')

            history = model.fit(self.noised_images[train].reshape(-1, 784),
                                self.images[train].reshape(-1, 784),
                                batch_size=self.BATCH_SIZE,
                                epochs=self.EPOCHS,
                                verbose=True)
            scores = model.evaluate(self.noised_images[test].reshape(-1, 784), self.images[test].reshape(-1, 784),
                                    verbose=0)
            print(
                f'Score for fold {fold_no}: {model.metrics_names} of {scores};')
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

    def enhance_image(self, index):
        image_to_enhance = self.noised_images[index]
        enhanced_image = self.model.predict(np.array(image_to_enhance).reshape(-1, 28 * 28))
        original_image = self.images[index]
        return image_to_enhance.reshape(28, 28), enhanced_image.reshape(28, 28), original_image.reshape(28, 28)

    def add_to_plot(self, index, axs, r, is_one_line):
        noised, enhanced, original = self.enhance_image(index)
        if (is_one_line):
            axs[0].imshow(noised)
            axs[1].imshow(enhanced)
            axs[2].imshow(original)

        else:
            axs[r, 0].imshow(noised)
            axs[r, 1].imshow(enhanced)
            axs[r, 2].imshow(original)
        return axs

    def plot(self, indices):
        r = len(indices)
        fig = plt.figure(dpi=100)
        axs = fig.subplots(r, 3)
        is_one_line = True if r == 1 else False
        for i in range(r):
            axs = self.add_to_plot(indices[i], axs, i, is_one_line=is_one_line)
        if is_one_line:
            axs[0].set_title("Noised")
            axs[1].set_title("Enhanced")
            axs[2].set_title("Original")
        else:
            axs[0, 0].set_title("Noised")
            axs[0, 1].set_title("Enhanced")
            axs[0, 2].set_title("Original")
        return fig


clf = Image_Reconstructor(EPOCHS=20, NUMBER_OF_FOLDS=5, noise_type='localvar',
                          noise_degree=ULTRA)

