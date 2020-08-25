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


def cost_function(state, coeff_arrays, constant_arrays):
    diff = 0;
    i = 0;
    for i in range(len(coeff_arrays)):
        mult = state * coeff_arrays[i]
        s = np.sum(mult);
        diff += (s - constant_arrays[i]) ** 2
        i += 1

    return diff / len(coeff_arrays)


def normal_hill_climber(step):
    global best_state
    global n
    global coeff_arrays
    global constant_arrays
    global best_state_heuristic
    changed = True
    neighbours = all_neighbour_generator(best_state, step)
    neighbours_h = [0] * len(neighbours)

    i = 0
    for i in range(len(neighbours)):
        h = cost_function(neighbours[i], coeff_arrays, constant_arrays)
        neighbours_h[i] = h
        i += 1

    min_index = np.argmin(neighbours_h)
    if (neighbours_h[min_index] < cost_function(best_state, coeff_arrays, constant_arrays)):
        best_state = neighbours[min_index]
        best_state_heuristic = neighbours_h[min_index]

    else:
        changed = False
    return changed
    pass


def variable_step_hill_climber(climber_function, low, high):
    c = True
    step = (high - low) * 0.1
    number_of_levels = 0;
    h = cost_function(best_state, coeff_arrays, constant_arrays)
    number_of_step_decrease_levels = 0;
    while (h > 0.000000001 and number_of_step_decrease_levels < 30 and number_of_levels < 1000000000):
        c = climber_function(step)
        h = best_state_heuristic
        if not c:
            step = step / 2;
            number_of_step_decrease_levels += 1
        number_of_levels += 1


def random_restart_variable_step_hill_climber(climber_function, low, high):
    global best_state, best_state_heuristic, all_of_best_states_heurisitcs, all_of_best_states, n
    c = True
    step = (high - low) * 0.1
    number_of_levels = 0;
    number_of_step_decrease_levels = 0
    h = cost_function(best_state, coeff_arrays, constant_arrays)
    while (number_of_levels < 10000):
        c = climber_function(step)
        h = best_state_heuristic
        if not c:
            step = step / 2;
            number_of_step_decrease_levels += 1
        number_of_levels += 1
        if not (h > 0.000000001 and number_of_step_decrease_levels < 30):
            all_of_best_states.append(best_state)
            all_of_best_states_heurisitcs.append(h)
            best_state_heuristic = m.inf
            number_of_step_decrease_levels = 0
            step = (high - low) * 0.1
            best_state = random_state_generator(low, high, n)
    min_index = np.argmin(all_of_best_states_heurisitcs)
    best_state = all_of_best_states[min_index]


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


# ------------------------------------- #
initializer()

low = float(input("Input Initial Lower Bound"))
high = float(input("Input Initial Higher Bound"))
step = float(
    input("Input Step; But my algorithm changes the step inside function, so the input step will not be used much"))

start = time.time_ns()

best_state = random_state_generator(low, high, n)

# uncomment this and comment the next function to run this program with another method
#variable_step_hill_climber(normal_hill_climber, -1000, 1000)

random_restart_variable_step_hill_climber(normal_hill_climber, -1000, 1000)

end = time.time_ns()

report_maker(best_state, coeff_arrays, constant_arrays, (end - start) / (10 ** 9))

# print((end - start) / (10 ** 9))
#
# print(best_state)
# print(cost_function(best_state, coeff_arrays, constant_arrays))
