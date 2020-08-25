import time

import numpy as np

from src.Function import *
from src.Genetic import FunctionRegressor
from graphviz import *
from src.Fitness import *

# Default:
population_size = 5000
generations = 20
stop_limit = 0.01
p_crossover = 0.7
p_mutation1 = 0.1
p_mutation2 = 0.05
p_mutation3 = 0.1
long_answer_penalty_multiplier = 0.01
function_set = ('add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt')

low, high = -10, 10
number_of_points = 100

long_answer_penalty_multiplier = 0.01,
function_set = ('add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt')

print("Genetic Programming Function Regressor")
print("If you want to use default values, just input -1, else input 1 ; please don't input other numbers.")

check = input().strip()
if check == "-1":
    pass
else:
    print(
        "Please Input desired values ; note that p_crossover and p_mutation1 and 2 and 3 sum must be less than 1 default 5000")
    print("Population Size:")
    population_size = int(input())
    print("Generation Limit: default 20")
    generations = int(input())
    print("fitness (error) stop limit. (float): defualt 0.01")
    stop_limit = float(input())
    print(
        "Input crossover probability; note that sum of cross over and p_mutation sum must be less than or equal to 1; crossover default 0.7")
    p_crossover = float(input())
    print("Input p_mutation 1 default 0.1")
    p_mutation1 = float(input())
    print("Input p_mutation 2 default 0.05")
    p_mutation2 = float(input())
    print("Input p_mutation 3 default 0.1")
    p_mutation3 = float(input())
    print("Long Answer Penalty Multiplier. (Small number like 0.01")
    long_answer_penalty_multiplier = float(input())

    print("INPUT FUNCTION SET AS A COMMA SEPERATED LIST; example add,sub,mul")
    print("AVAILABLE FUNCTIONS:")
    print(
        "add , sub , mul , div , sin , cos , sqrt , tan , log , neg (negation) , abs , exp , inv , sig (sigmoid function)")
    funcs = input()
    function_set = tuple(funcs.split(','))
    print(function_set)

    print("number of Points (default 100)")
    number_of_points = int(input())

    print("Points will be generated using np.random.uniform")
    print("input low bound ; defualt: -10")
    low = float(input())
    print("input high bound ; defualt: 10")
    high = float(input())

X_train = np.random.uniform(low, high, number_of_points).reshape(-1, 1)
X_test = np.random.uniform(low, high, number_of_points).reshape(-1, 1)
y_train = 0
y_test = 0

print("Do You want to input your function? 1: yes. 0 :no")
your_func = input()
if your_func == '0':
    print("Which Default Function do you want:")
    print(" 1. x * x + 2 * x + 1")
    print(" 2. sin(x) + 2")
    print(" 3. log(x)")
    print(" 4. log(x) * x")
    print(" 5. 2 * x * x * x + 5 x + 3")
    print(" 6. sqrt(x)")
    print("7. 1/(x+5)")
    print("invalid input is considered f(x) = x")

    func_choice = input()
    if func_choice == '1':
        y_train = X_train * X_train + 2 * X_train + 1
        y_test = X_test * X_test + 2 * X_test + 1
        function_set = ('add', 'sub', 'mul', 'div')
    elif func_choice == '2':
        y_train = np.sin(X_train) + 2
        y_test = np.sin(X_test) + 2
        function_set = ('add', 'sub', 'mul', 'sin', 'cos',)
    elif func_choice == '3':
        y_train = protected_log(X_train)
        y_test = protected_log(X_test)
        function_set = ('add', 'sub', 'mul', 'log')
    elif func_choice == '4':
        y_train = protected_log(X_train) * X_train
        y_test = protected_log(X_test) * X_test
        function_set = ('add', 'sub', 'mul', 'log')
    elif func_choice == '5':
        y_train = X_train * X_train * X_train * 2 + 5 * X_train + 3
        y_test = X_test * X_test * X_test * 2 + 5 * X_test + 3
        function_set = ('add', 'sub', 'mul', 'div')
    elif func_choice == '6':
        y_train = protected_sqrt(X_train)
        y_test = protected_sqrt(X_test)
        function_set = ('add', 'sub', 'mul', 'sqrt')

    elif func_choice == '7':
        y_train = protected_inverse(X_train + 5)
        y_test = protected_inverse(X_test + 5)
        function_set = ('add', 'sub', 'mul', 'inv')
    else:
        y_train = X_train
        y_test = X_train
        function_set = ('add', 'sub', 'mul')



else:
    print("Input your own function by just using one variable x")
    print("For sin and cos; use np.sin(x) and np.cos(x) ")
    print("For exp; use protected_exp(x)")
    print("For sigmoid; use _sigmoid(x)")
    print("For div; use protected_div(...,...) (because of divison by zero)")
    print("For sqrt; use protected_sqrt(...,...) ")
    print("For log (it is just log in base e); use protected_log(...) ")
    print("For inverse; use protected_inverse(...) ")
    function_input = input()
    function_input_train = function_input.replace("x", "X_train")
    function_input_test = function_input.replace("x", "X_test")

    y_train = eval(function_input_train)
    y_test = eval(function_input_test)

print("input metric")
print("1. mean absolute error (defualt if invalid input)")
print("2. mean square error")
print("3. root mean square error")
metric = 'mean absolute error'
metric_choose = input().strip()
if metric_choose == '1':
    metric = 'mean absolute error'
elif metric_choose == '2':
    metric = 'mean square error'
elif metric_choose == '3':
    metric = 'root mean square error'
else:
    metric = 'mean absolute error'

start = time.time()

estimator = FunctionRegressor(population_size=population_size,
                              generations=generations, stop_limit=stop_limit,
                              p_crossover=p_crossover, p_mutation1=p_mutation1,
                              p_mutation2=p_mutation2, p_mutation3=p_mutation3,
                              metric=metric,
                              long_answer_penalty_multiplier=long_answer_penalty_multiplier,
                              function_set=function_set
                              )
estimator.fit(X_train, y_train)
score = estimator.score(X_test, y_test)
print("Time: ", time.time() - start)
print("Function: ", estimator.chromosome)
print("R^2 Score using sickitlearn", score)
fitnessCounter = FitnessCounter()
print("Number of Fitness Function Calls:", fitnessCounter.counter)

print("DO YOU WANT GRAPHICAL INPUT: 1: yes , 0 :No")
graphical_input = input()
if graphical_input == "1":

    src = Source(estimator.chromosome.export_graphviz())
    src.render('test-output/answer.abc', view=True)


else:
    pass
