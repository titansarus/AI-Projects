from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt



def plot_svc(model):
    """Plot the decision function for a 2D SVC"""
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


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

    return clfs[best_one_index], test_scores[best_one_index]


def test_case_evaluator(test_case):
    if test_case == 0:
        x, y = datasets.make_circles(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=42, factor=0.2)
        clf, best_score = kfold_svm(x, y, kernel='rbf', C=1, gamma=1)
    elif test_case == 1:
        x, y = datasets.make_circles(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=42, factor=0.2)
        clf, best_score = kfold_svm(x, y, kernel='poly', C=1, degree=5, coef0=0.5)
    elif test_case == 2:
        x, y = datasets.make_circles(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=42, factor=0.2)
        clf, best_score = kfold_svm(x, y, kernel='linear', C=1)  # which doesn't work fine

    elif test_case == 10:
        x, y = datasets.make_circles(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=30, factor=0.8)
        clf, best_score = kfold_svm(x, y, kernel='rbf', C=1, gamma=1)
    elif test_case == 11:
        x, y = datasets.make_circles(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=30, factor=0.8)
        clf, best_score = kfold_svm(x, y, kernel='rbf', C=1, gamma=1000)  # overfit
    elif test_case == 12:
        x, y = datasets.make_circles(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=30, factor=0.8)
        clf, best_score = kfold_svm(x, y, kernel='linear', C=1)  # very bad
    elif test_case == 13:
        x, y = datasets.make_circles(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=30, factor=0.8)
        clf, best_score = kfold_svm(x, y, kernel='poly', C=1, degree=5, coef0=0.5)

    elif test_case == 20:
        x, y = datasets.make_moons(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=32)
        clf, best_score = kfold_svm(x, y, kernel='rbf', C=1, gamma=1)
    elif test_case == 21:
        x, y = datasets.make_moons(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=32)
        clf, best_score = kfold_svm(x, y, kernel='rbf', C=1, gamma=1000)  # very overfit
    elif test_case == 22:
        x, y = datasets.make_moons(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=32)
        clf, best_score = kfold_svm(x, y, kernel='poly', C=1, degree=3, coef0=0.5)
    elif test_case == 23:
        x, y = datasets.make_moons(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=32)
        clf, best_score = kfold_svm(x, y, kernel='linear', C=1)
    elif test_case == 24:
        x, y = datasets.make_moons(n_samples=NUMBER_OF_SAMPLES, noise=0.1, random_state=32)
        clf, best_score = kfold_svm(x, y, kernel='rbf', C=0.0001, gamma=1)  # very unsensitive

    elif test_case == 30:
        x, y = datasets.make_blobs(n_samples=NUMBER_OF_SAMPLES, centers=2, cluster_std=1, random_state=150)
        clf, best_score = kfold_svm(x, y, kernel='rbf', C=1, gamma=1)
    elif test_case == 31:
        x, y = datasets.make_blobs(n_samples=NUMBER_OF_SAMPLES, centers=2, cluster_std=1, random_state=150)
        clf, best_score = kfold_svm(x, y, kernel='poly', C=1, degree=3, coef0=0.5)
    elif test_case == 32:
        x, y = datasets.make_blobs(n_samples=NUMBER_OF_SAMPLES, centers=2, cluster_std=1, random_state=150)
        clf, best_score = kfold_svm(x, y, kernel='linear', C=1)

    elif test_case == 40:
        x, y = datasets.make_blobs(n_samples=NUMBER_OF_SAMPLES, centers=2, cluster_std=2, random_state=175)
        clf, best_score = kfold_svm(x, y, kernel='rbf', C=1, gamma=1)  # a liitle overfit
    elif test_case == 41:
        x, y = datasets.make_blobs(n_samples=NUMBER_OF_SAMPLES, centers=2, cluster_std=2, random_state=175)
        clf, best_score = kfold_svm(x, y, kernel='poly', C=1, degree=3, coef0=0.5)
    elif test_case == 42:
        x, y = datasets.make_blobs(n_samples=NUMBER_OF_SAMPLES, centers=2, cluster_std=2, random_state=175)
        clf, best_score = kfold_svm(x, y, kernel='linear', C=1)
    return x, y, clf, best_score


NUMBER_OF_SAMPLES = 2000

print("Choose a test case: (input number) \n")
print("0. two circles with rbf kernel")
print("1. two circles with poly degree 5 kernel")
print("2. two circles with linear kernel which doesn't work good")
print("10. two more difficult circles with rbf kernel")
print("11. two more difficult circles with rbf kernel that overfits")
print("12. two more difficult circles with linear kernel whcih doesn't work good")
print("13. two more difficult circles with poly degree 5 kernel whcih doesn't work good")
print("20. moon (crescent) with rbf kernel")
print("21. moon (crescent) with rbf kernel with very overfit")
print("22. moon (crescent) with poly degree 3 kernel")
print("23. moon (crescent) with linear kernel which is not very good")
print("24. moon (crescent) with rbf kernel with low sensitivity")
print("30. two blobs with low std and rbf kernel")
print("31. two blobs with low std and poly degree 3 kernel")
print("32. two blobs with low std and linear kernel")
print("40. two blobs with more std and rbf kernel")
print("41. two blobs with more std and poly degree 3 kernel")
print("42. two blobs with more std and linear kernel")



test_case = int(input("input the test case you want: "))

x, y, clf, best_score = test_case_evaluator(test_case)

plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc(clf)
plt.show()
plt.clf()
print("best score: "+ str(best_score))


