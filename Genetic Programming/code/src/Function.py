import numpy as np


class Function(object):

    def __init__(self, function, name, arg_count):
        self.function = function
        self.name = name
        self.arg_count = arg_count

    def __call__(self, *args):
        return self.function(*args)


def protected_div(x1, x2):
    # Protect From Divison by Zero and very small numbers
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def protected_sqrt(x1):
    # Protect From negative numbers
    return np.sqrt(np.abs(x1))


def protected_log(x1):
    # Protect from very small numbers and zero numbers
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def protected_inverse(x1):
    # Protect From Divison by Zero and very small numbers
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 1.)


def _sigmoid(x1):
    # I read that it is good to include sigmoid function for this type of programs So I included it.
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


def protected_exp(x1):
    with np.errstate(invalid='ignore'):
        return np.where(x1 < 10., np.exp(x1), np.exp(10))


add2 = Function(function=np.add, name='add', arg_count=2)
sub2 = Function(function=np.subtract, name='sub', arg_count=2)
mul2 = Function(function=np.multiply, name='mul', arg_count=2)
div2 = Function(function=protected_div, name='div', arg_count=2)
sqrt1 = Function(function=protected_sqrt, name='sqrt', arg_count=1)
log1 = Function(function=protected_log, name='log', arg_count=1)
neg1 = Function(function=np.negative, name='neg', arg_count=1)
inv1 = Function(function=protected_inverse, name='inv', arg_count=1)
abs1 = Function(function=np.abs, name='abs', arg_count=1)
sin1 = Function(function=np.sin, name='sin', arg_count=1)
cos1 = Function(function=np.cos, name='cos', arg_count=1)
tan1 = Function(function=np.tan, name='tan', arg_count=1)
sig1 = Function(function=_sigmoid, name='sig', arg_count=1)
exp1 = Function(function=protected_exp, name='exp', arg_count=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'exp': exp1,
                 }
