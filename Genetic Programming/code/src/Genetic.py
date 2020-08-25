from time import time

import numpy as np

MAX_INT = np.iinfo(np.int32).max

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

from src.DataStructure import Chromosome
from src.Fitness import fitness_map
from src.Function import _function_map, Function


def population_evolver(n_chromosomes, parents, X, y, params):
    not_used, variable_count = X.shape
    competition_size = params['competition_size']
    function_set = params['function_set']
    arg_counts = params['arg_counts']
    init_depth = params['init_depth']
    const_range = params['const_range']
    metric = params['_metric']
    long_answer_penalty_multiplier = params['long_answer_penalty_multiplier']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']

    def competition():

        competitors = np.random.randint(0, len(parents), competition_size);
        fitness = [parents[p].normal_fitness for p in competitors];
        parent_index = competitors[np.argmin(fitness)]
        return parents[parent_index], parent_index

    chromosomes = []

    for i in range(n_chromosomes):

        if parents is None:
            chromosome = None
            genome = None
        else:
            method = np.random.uniform()
            parent, parent_index = competition()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = competition()
                chromosome, removed, remains = parent.crossover(donor.chromosome,
                                                                )
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # mutation1
                chromosome, removed, _ = parent.mutation1()
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # mutation2
                chromosome, removed = parent.mutation2()
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # mutation3
                chromosome, mutated = parent.mutation3()
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                chromosome = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        chromosome = Chromosome(function_set=function_set,
                                arg_counts=arg_counts,
                                init_depth=init_depth,
                                variable_count=variable_count,
                                metric=metric,
                                const_range=const_range,
                                p_point_replace=p_point_replace,
                                long_answer_penalty_multiplier=long_answer_penalty_multiplier,
                                chromosome=chromosome)

        chromosome.parents = genome

        chromosome.normal_fitness = chromosome.raw_fitness(X, y)

        chromosomes.append(chromosome)

    return chromosomes


class FunctionRegressor(BaseEstimator, RegressorMixin):

    def __init__(self,
                 population_size=1000,
                 generations=20,
                 competition_size=20,
                 stop_limit=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
                 metric='mean absolute error',
                 long_answer_penalty_multiplier=0.001,
                 p_crossover=0.9,
                 p_mutation1=0.01,
                 p_mutation2=0.01,
                 p_mutation3=0.01,
                 p_point_replace=0.05,
                 ):

        self.population_size = population_size

        self.generations = generations
        self.competition_size = competition_size
        self.stop_limit = stop_limit
        self.const_range = const_range
        self.init_depth = init_depth
        self.function_set = function_set
        self.metric = metric
        self.long_answer_penalty_multiplier = long_answer_penalty_multiplier
        self.p_crossover = p_crossover
        self.p_mutation1 = p_mutation1
        self.p_mutation2 = p_mutation2
        self.p_mutation3 = p_mutation3
        self.p_point_replace = p_point_replace

    def detail_maker(self, run_details=None):

        if run_details is None:
            line_format = '{:>4} {:>8} {:>16} '
            print(line_format.format('Gen', 'Length',
                                     'Fitness', ))

        else:

            line_format = '{:4d}  {:8d} {:16g}'

            print(line_format.format(run_details['generation'][-1],

                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     ))

    def fit(self, X, y):

        X, y = check_X_y(X, y, y_numeric=True)

        not_used, self.variable_count = X.shape

        self.function_set_util = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    pass
                self.function_set_util.append(_function_map[function])
            elif isinstance(function, Function):
                self.function_set_util.append(function)
            else:
                pass
        if not self.function_set_util:
            raise ValueError('No VALID FUNCTION EXISTS')

        self.arg_counts = {}
        for function in self.function_set_util:
            arg_count = function.arg_count
            self.arg_counts[arg_count] = self.arg_counts.get(arg_count, [])
            self.arg_counts[arg_count].append(function)

        self.metric = fitness_map[self.metric]

        self.method_probs = np.array([self.p_crossover,
                                      self.p_mutation1,
                                      self.p_mutation2,
                                      self.p_mutation3])
        self.method_probs = np.cumsum(self.method_probs)

        params = self.get_params()
        params['_metric'] = self.metric

        params['function_set'] = self.function_set_util
        params['arg_counts'] = self.arg_counts
        params['method_probs'] = self.method_probs

        if not hasattr(self, 'chromosomes'):
            self.chromosomes = []
            self.run_details = {'generation': [],
                                'average_length': [],
                                'average_fitness': [],
                                'best_length': [],
                                'best_fitness': [],
                                }

        prior_generations = len(self.chromosomes)
        n_more_generations = self.generations - prior_generations

        if n_more_generations == 0:
            fitness = [chromosome.normal_fitness for chromosome in self.chromosomes[-1]]

        self.detail_maker()

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self.chromosomes[gen - 1]

            population = population_evolver(self.population_size,
                                            parents,
                                            X,
                                            y,
                                            params)

            fitness = [chromosome.normal_fitness for chromosome in population]
            length = [chromosome.length_ for chromosome in population]

            long_answer_penalty_multiplier = self.long_answer_penalty_multiplier

            for chromosome in population:
                chromosome.fitness_with_length_penalty = chromosome.fitness(long_answer_penalty_multiplier)

            self.chromosomes.append(population)

            if gen > 0:  # delete previous gen
                self.chromosomes[gen - 1] = None

            best_chromosome = population[np.argmin(fitness)]

            self.run_details['generation'].append(gen)
            self.run_details['average_length'].append(np.mean(length))
            self.run_details['average_fitness'].append(np.mean(fitness))
            self.run_details['best_length'].append(best_chromosome.length_)
            self.run_details['best_fitness'].append(best_chromosome.normal_fitness)

            self.detail_maker(self.run_details)

            best_fitness = fitness[np.argmin(fitness)]
            if best_fitness <= self.stop_limit:
                break

        self.chromosome = self.chromosomes[-1][np.argmin(fitness)]

        return self

    def __str__(self):
        if not hasattr(self, 'chromosome'):
            return self.__repr__()
        return self.chromosome.__str__()

    def predict(self, X):

        X = check_array(X)
        not_used, variable_count = X.shape;
        if self.variable_count != variable_count:
            raise ValueError('Shape of Input Is Not fine!')

        y = self.chromosome.execute(X);
        return y
