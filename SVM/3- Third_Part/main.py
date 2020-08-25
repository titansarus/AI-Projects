from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt

def kfold_svm(X, y, number_of_folds=5, kernel='rbf', C=1, gamma='auto', degree=5, coef0=0.0):
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
    cross_validate_result = cross_validate(clf, X, y, cv=number_of_folds, scoring='accuracy',
                                           return_estimator=True, return_train_score=True)
    clfs = cross_validate_result['estimator']
    test_scores = cross_validate_result['test_score']
    train_scores = cross_validate_result['train_score']

    print("train_scores: " )
    print(train_scores)
    print()
    print("test_scores: " )
    print(test_scores)
    best_one_index = np.argmax(test_scores)
    clfs[best_one_index].predict([X[0]])

    return clfs[best_one_index], test_scores[best_one_index]

dataset = np.load('persian_lpr.npz')
images = dataset['images']
labels = dataset['targets']

images = images/255
number_of_samples = len(images)
images_flatten = images.reshape((number_of_samples,-1))
number_of_folds = 3



print("RBF:")
clf_rbf , score_rbf= kfold_svm(images_flatten,labels,number_of_folds=number_of_folds ,
                       kernel='rbf' , C = 1,gamma=0.1)
print("Best RBF accuracy= ",score_rbf)
print("--------------------")


print("Linear:")
clf_linear , score_linear= kfold_svm(images_flatten,labels,number_of_folds=number_of_folds ,
                       kernel='linear' , C = 1)
print("Best Linear accuracy= ",score_linear)
print("--------------------")

print("Poly Degree 6:")
clf_poly , score_poly= kfold_svm(images_flatten,labels,number_of_folds=number_of_folds ,
                       kernel='poly' ,degree = 6 , coef0=1.2, C = 1)
print("Best Poly degree 6 accuracy= ",score_poly)
print("--------------------")


do_test_images = input("Do you want to test an image?\n 1:Yes \t 0:No\n")
if (do_test_images=='1'):
    image_index = int(input("input image index: "))
    plt.imshow(images[image_index])
    plt.show()
    print("Images is: " ,labels[image_index])
    print('rbf predicts:' ,clf_rbf.predict([images_flatten[image_index]]))
    print('linear predicts:' ,clf_linear.predict([images_flatten[image_index]]))
    print('poly degree 6 predicts:' ,clf_poly.predict([images_flatten[image_index]]))
