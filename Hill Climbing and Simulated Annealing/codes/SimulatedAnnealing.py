import time

import numpy as np
import random as rand
import math as m

coeff_arrays = 0;
constant_arrays = 0;
best_state = 0;
best_state_heuristic = m.inf;
all_of_best_states = []
all_of_best_states_heurisitcs = []
n = 0

CONSTANT_FACTOR = 0.05

T_start = 100 * CONSTANT_FACTOR

T_end = 0.0000001 * CONSTANT_FACTOR

alpha = 0.9996  # changeing factor

MAX_NEIGHBOURS = 40


def initializer():
    global constant_arrays
    global coeff_arrays
    global n
    lines = open("new_example.txt").read().split('\n')

    number_of_equations = len(lines) - 1
    n = len(lines[0].split(','))

    arrays = [np.empty([n], dtype=float)] * number_of_equations;

    lines.remove(lines[number_of_equations])

    i = 0
    for line in lines:
        arrays[i] = np.array(line.split(','))
        i += 1
    i = 0

    coeff_arrays = [np.empty([n], dtype=float)] * number_of_equations;
    i = 0;

    for array in arrays:
        coeff_arrays[i] = array[0:n - 1]
        coeff_arrays[i] = coeff_arrays[i].astype(float)
        i += 1;

    constant_arrays = [np.empty([n], dtype=float)] * number_of_equations;

    i = 0;
    for array in arrays:
        constant_arrays[i] = array[n - 1]
        constant_arrays[i] = constant_arrays[i].astype(float)
        i += 1

    n = n - 1  # number of unknonws


def random_state_generator(low: float, high: float, size: int):
    return np.random.uniform(low=low, high=high, size=size)


def cost_function(state, coeff_arrays, constant_arrays):
    diff = 0;
    i = 0;
    for i in range(len(coeff_arrays)):
        mult = state * coeff_arrays[i]
        s = np.sum(mult);
        diff += m.fabs(s - constant_arrays[i])
        i += 1

    return diff / len(coeff_arrays)


def all_neighbour_generator(array, step):
    l = array.shape[0]
    answer = [0] * (l * 2)
    for i in range(l * 2):
        if i % 2 == 0:
            temp = np.copy(array)
            temp[i // 2] = temp[i // 2] + step
            answer[i] = temp

        else:
            temp = np.copy(array)
            temp[i // 2] = temp[i // 2] - step
            answer[i] = temp

    if (l * 2 > MAX_NEIGHBOURS):
        answer = rand.sample(answer, MAX_NEIGHBOURS)
    rand.shuffle(answer)
    return answer



def isExactPower(a, b):
    num = m.log(a, b)
    if (m.fabs(num - round(num)) < 0.0000001):
        return True
    return False


def simulated_annealing(low, high):
    global T_start, T_end, constant_arrays, coeff_arrays, best_state, alpha, n
    best_state = random_state_generator(low, high, n)
    t = T_start
    number_of_levels = 0
    step = (high - low) * 0.2
    while t > T_end:
        current_h = cost_function(best_state, coeff_arrays, constant_arrays)
        all_neighbours = all_neighbour_generator(best_state, step)
        while True:

            if len(all_neighbours) != 0:
                neighbour = all_neighbours.pop()
                new_h = cost_function(neighbour, coeff_arrays, constant_arrays)
                if new_h < current_h:
                    best_state = neighbour
                    break;
                else:
                    r = rand.random()
                    # note that current_h and new_h are costs. so deltaE is actually current - new
                    prob = m.exp(-1 * (new_h - current_h) / t)

                    if r < prob:
                        best_state = neighbour
                        break
            else:
                step = step / 1.3
                break;


        number_of_levels += 1;
        if number_of_levels < 10:
            pass
        else:
            if number_of_levels % 1 == 0:
                t = t * alpha


def report_maker(answer, coeff_arr, const_arr, time):
    print("ANSWER:")
    print(answer)
    answer_rhs = [0] * len(coeff_arrays);
    for i in range(len(coeff_arr)):
        temp = answer * coeff_arr[i]
        s = np.sum(temp)
        answer_rhs[i] = s
    print("REAL RIGHT HAND SIDE:")
    print(const_arr)
    print("ANSWER RIGHT HAND SIDE:")
    print(answer_rhs)

    print("Mean Square Error:")
    print(cost_function(answer, coeff_arr, const_arr))

    print("Root Mean Square Error:")
    print(m.sqrt(cost_function(answer, coeff_arr, const_arr)))

    print("TIME:")
    print(time)

low = float(input("Input Initial Lower Bound"))
high = float(input("Input Initial Higher Bound"))
step = float(
    input("Input Step; But my algorithm changes the step inside function, so the input step will not be used much"))


initializer()

start = time.time_ns()

simulated_annealing(low, high)

end = time.time_ns()

report_maker(best_state, coeff_arrays, constant_arrays, (end - start) / (10 ** 9))

# best_state = random_state_generator(-1000, 1000, n)
# print(best_state)
# print(neighbour_generator(best_state, 20));
