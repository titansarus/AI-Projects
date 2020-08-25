import threading
import numpy as np
import random as rand
import time
import math as m

coeff_arrays = 0;
constant_arrays = 0;
n = 0
NUMBER_OF_TESTS = 4
MAX_NEIGHBOURS = 40

threads = [object] * NUMBER_OF_TESTS

best_states = [object] * NUMBER_OF_TESTS
heuristics = [object] * NUMBER_OF_TESTS


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
        rand.sample(answer, MAX_NEIGHBOURS)
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


class myThread(threading.Thread):
    def __init__(self, threadID, name, low, high):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.low = low
        self.high = high

    def run(self):
        best_states[self.threadID] = random_state_generator(self.low, self.high, n)
        variable_step_hill_climber(normal_hill_climber, self.low, self.high, self.threadID)
        pass


def variable_step_hill_climber(climber_function, low, high, id):
    global heuristics
    id = int(id)
    c = True
    step = (high - low) * 0.1
    number_of_levels = 0;
    number_of_step_decrease_levels = 0;
    h = cost_function(best_states[id], coeff_arrays, constant_arrays)
    while (h > 0.000000001 and number_of_step_decrease_levels < 30 and number_of_levels < 1000000000):
        c = climber_function(step, id)
        h = heuristics[id]
        if not c:
            step = step / 2;
            number_of_step_decrease_levels += 1
        number_of_levels += 1


def normal_hill_climber(step, id):
    global best_states
    global heuristics
    global n
    global coeff_arrays
    global constant_arrays

    changed = True
    neighbours = all_neighbour_generator(best_states[id], step)
    neighbours_h = [0] * len(neighbours)

    i = 0
    for i in range(len(neighbours)):
        h = cost_function(neighbours[i], coeff_arrays, constant_arrays)

        neighbours_h[i] = h
        i += 1

    min_index = np.argmin(neighbours_h)
    curr_h = cost_function(best_states[id], coeff_arrays, constant_arrays)
    if (neighbours_h[min_index] < curr_h):
        best_states[id] = neighbours[min_index]
        heuristics[id] = neighbours_h[min_index]

    else:
        heuristics[id] = curr_h
        changed = False
    return changed


def threadRunner(number_of_threads, low, high):
    global threads

    for i in range(number_of_threads):
        t = myThread(i, "NAME" + str(i), low, high)
        threads[i] = t
        t.start()


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

threadRunner(NUMBER_OF_TESTS, low, high)

for t in threads:
    t.join()

min_index = np.argmin(heuristics)

answer = best_states[min_index]

end = time.time_ns()

report_maker(answer, coeff_arrays, constant_arrays, (end - start) / (10 ** 9))
