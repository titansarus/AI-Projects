from copy import copy

import numpy as np
from sklearn.utils.random import sample_without_replacement

from src.Fitness import FitnessCounter
from src.Function import Function



class Chromosome(object):

    def __init__(self,
                 function_set,
                 arg_counts,
                 init_depth,
                 variable_count,
                 const_range,
                 metric,
                 p_point_replace,
                 long_answer_penalty_multiplier,

                 chromosome=None):

        self.function_set = function_set
        self.arg_counts = arg_counts
        self.init_depth = (init_depth[0], init_depth[1] + 1)

        self.variable_count = variable_count
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.long_answer_penalty_multiplier = long_answer_penalty_multiplier

        self.chromosome = chromosome

        if self.chromosome is not None:
            if not self.validate_chromosome():
                pass
        else:
            self.chromosome = self.build_chromosome()
        self.normal_fitness = None
        self.fitness_with_length_penalty = None
        self.parents = None
        self.n_samples = None
        self.max_samples = None

    def build_chromosome(self, ):

        # full means it must make exactly a DS with max Depth. not-full means it must not necessary be max-length
        method = ('full' if np.random.randint(2) else 'not-full')

        max_depth = np.random.randint(*self.init_depth)

        function = np.random.randint(len(self.function_set))
        function = self.function_set[function]
        chromosome = [function]
        terminal_stack = [function.arg_count]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.variable_count + len(self.function_set)
            choice = np.random.randint(choice)
            # add function or terminal (terminal means constant or variable that end the tree branch on that point)
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = np.random.randint(len(self.function_set))
                function = self.function_set[function]
                chromosome.append(function)
                terminal_stack.append(function.arg_count)
            else:
                if self.const_range is not None:
                    terminal = np.random.randint(self.variable_count + 1)
                else:
                    terminal = np.random.randint(self.variable_count)
                if terminal == self.variable_count:
                    terminal = np.random.uniform(*self.const_range)

                chromosome.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return chromosome
                    terminal_stack[-1] -= 1

        return None

    def validate_chromosome(self):
        # check if chromosome is valid
        terminals = [0]
        for node in self.chromosome:
            if isinstance(node, Function):
                terminals.append(node.arg_count)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):

        terminals = [0]
        output = ''
        for i, node in enumerate(self.chromosome):
            if isinstance(node, Function):
                terminals.append(node.arg_count)
                output += node.name + '('
            else:
                if isinstance(node, int):

                    output += 'X%s' % node

                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.chromosome) - 1:
                    output += ', '
        return output

    def export_graphviz(self):
        '''
        A function that makes a graphviz script that can be
        changed into an graphical represenatation.
        I Used Internet And StackOverflow and some Github repos to copy paste
        Graphviz syntax

        '''
        terminals = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.chromosome):
            if isinstance(node, Function):

                fill = '#00bdaa'
                terminals.append([node.arg_count, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:

                fill = '#fe346e'
                if isinstance(node, int):

                    feature_name = 'X%s' % node

                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        return None

    def get_depth(self):

        terminals = [0]
        depth = 1
        for node in self.chromosome:
            if isinstance(node, Function):
                terminals.append(node.arg_count)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def get_length(self):
        return len(self.chromosome)

    def execute(self, X):

        # Check for single-node programs
        node = self.chromosome[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.chromosome:

            if isinstance(node, Function):
                apply_stack.append([node])
            else:
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arg_count + 1:

                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        ):
        if n_samples is not None and self.n_samples is None:
            self.n_samples = n_samples
        if max_samples is not None and self.max_samples is None:
            self.max_samples = max_samples

        not_indices = sample_without_replacement(
            self.n_samples,
            self.n_samples - self.max_samples,
            random_state=np.random.get_state())
        sample_counts = np.bincount(not_indices, minlength=self.n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def get_indices(self):

        return self.get_all_indices()[0]

    def raw_fitness(self, X, y):


        y_pred = self.execute(X)

        raw_fitness = self.metric(y, y_pred)

        fitnessCounter = FitnessCounter()
        fitnessCounter.increase();

        return raw_fitness

    def fitness(self, long_answer_penalty_multiplier=None):

        if long_answer_penalty_multiplier is None:
            long_answer_penalty_multiplier = self.long_answer_penalty_multiplier
        penalty = long_answer_penalty_multiplier * len(self.chromosome) * -1
        return self.normal_fitness - penalty

    def get_subtree(self, chromosome=None):

        if chromosome is None:
            chromosome = self.chromosome

        probs = np.array([0.9 if isinstance(node, Function) else 0.1
                          for node in chromosome])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, np.random.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = chromosome[end]
            if isinstance(node, Function):
                stack += node.arg_count
            end += 1

        return start, end

    def reproduce(self):

        return copy(self.chromosome)

    def crossover(self, donor):
        start, end = self.get_subtree()
        removed = range(start, end)

        donor_start, donor_end = self.get_subtree(donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))

        return (self.chromosome[:start] +
                donor[donor_start:donor_end] +
                self.chromosome[end:]), removed, donor_removed

    def mutation1(self, ):

        dummy_chromosome = self.build_chromosome()

        return self.crossover(dummy_chromosome)

    def mutation2(self):

        start, end = self.get_subtree()
        subtree = self.chromosome[start:end]

        sub_start, sub_end = self.get_subtree(subtree)
        hoist = subtree[sub_start:sub_end]

        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.chromosome[:start] + hoist + self.chromosome[end:], removed

    def mutation3(self):
        new_chromosome = copy(self.chromosome)

        mutate = np.where(np.random.uniform(size=len(new_chromosome)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(new_chromosome[node], Function):
                arg_count = new_chromosome[node].arg_count

                replacement = len(self.arg_counts[arg_count])
                replacement = np.random.randint(replacement)
                replacement = self.arg_counts[arg_count][replacement]
                new_chromosome[node] = replacement
            else:
                # terminal means variable or constant that ends branch of tree
                if self.const_range is not None:
                    terminal = np.random.randint(self.variable_count + 1)
                else:
                    terminal = np.random.randint(self.variable_count)
                if terminal == self.variable_count:
                    terminal = np.random.uniform(*self.const_range)
                new_chromosome[node] = terminal

        return new_chromosome, list(mutate)

    depth_ = property(get_depth)
    length_ = property(get_length)
    indices_ = property(get_indices)
