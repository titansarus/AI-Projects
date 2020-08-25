from tensorflow import keras
from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
import os



def kfold_svm(X, y, number_of_folds=5, kernel='rbf', C=1, gamma='auto', degree=5, coef0=0.0):
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
    cross_validate_result = cross_validate(clf, X, y, cv=number_of_folds, scoring='accuracy',
                                           return_estimator=True, return_train_score=True)
    clfs = cross_validate_result['estimator']
    train_scores = cross_validate_result['train_score']
    test_scores = cross_validate_result['test_score']

    print("train_scores: " )
    print(train_scores)
    print()
    print("test_scores: " )
    print(test_scores)
    best_one_index = np.argmax(test_scores)

    return clfs[best_one_index], train_scores[best_one_index]

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
MNIST_DS = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = MNIST_DS.load_data()

train_images = train_images / 255
test_images = test_images / 255


images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

number_of_samples = len(images)

images_flatten = images.reshape((number_of_samples,-1))


clfs = []
scores = []
clf , score= kfold_svm(images_flatten[:],labels[:],number_of_folds=2 ,
                           kernel='linear' , C = 1)

clfs.append(clf)
scores.append(score)

print("best score" + str(scores))
