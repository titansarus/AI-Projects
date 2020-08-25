import numpy as np

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance
@singleton
class FitnessCounter(object):
    def __init__(self):
        self.counter = 0;
    def increase(self):
        self.counter+=1



class Fitness(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):

        return self.function(*args)


def mae(y, y_pred):

    return np.average(np.abs(y_pred - y))


def mse(y, y_pred, w):

    return np.average(((y_pred - y) ** 2))


def rmse(y, y_pred, w):

    return np.sqrt(np.average(((y_pred - y) ** 2)))


mean_absolute_error = Fitness(function=mae
                              )
mean_square_error = Fitness(function=mse
                            )
root_mean_square_error = Fitness(function=rmse
                                 )

fitness_map = {
    'mean absolute error': mean_absolute_error,
    'mse': mean_square_error,
    'rmse': root_mean_square_error,
}
