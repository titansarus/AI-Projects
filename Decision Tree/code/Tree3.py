import pandas as pd
import math
from sklearn.model_selection import train_test_split
import operator
import gvgen
from graphviz import Source


def discrete_find_bins(df: pd.DataFrame, column_name, number_of_bins):
    maxi = df[column_name].max()
    mini = df[column_name].min();
    diff = (maxi - mini) / number_of_bins
    bins = []
    bins.append(round(mini - diff, 2))
    for i in range(number_of_bins):
        bins.append(round(mini + diff * i, 2))
    bins.append(round(maxi, 2))
    bins.append(round(maxi + diff, 2))
    return bins


def discrete_column(df: pd.DataFrame, column_name, bins_):
    label = []
    for i in range(len(bins_) - 1):
        label.append(i)
    dfnew = df.copy()
    dfnew[column_name + '-binned'] = pd.cut(x=df[column_name], bins=bins_, labels=label)
    return dfnew


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    col_name = list(train.columns)
    Pregnancies_bins = discrete_find_bins(train, 'Pregnancies', 5)
    Glucose_bins = discrete_find_bins(train, 'Glucose', 5)
    BloodPressure_bins = discrete_find_bins(train, 'BloodPressure', 5)
    SkinThickness_bins = discrete_find_bins(train, 'SkinThickness', 5)
    Insulin_bins = discrete_find_bins(train, 'Insulin', 5)
    BMI_bins = discrete_find_bins(train, 'BMI', 5)
    DiabetesPedigreeFunction_bins = discrete_find_bins(df, 'DiabetesPedigreeFunction', 5)
    Age_bins = discrete_find_bins(df, 'Age', 5)
    #################

    train = discrete_column(train, 'Pregnancies', Pregnancies_bins)
    train = discrete_column(train, 'Glucose', Glucose_bins)
    train = discrete_column(train, 'BloodPressure', BloodPressure_bins)
    train = discrete_column(train, 'SkinThickness', SkinThickness_bins)
    train = discrete_column(train, 'Insulin', Insulin_bins)
    train = discrete_column(train, 'BMI', BMI_bins)
    train = discrete_column(train, 'DiabetesPedigreeFunction', DiabetesPedigreeFunction_bins)
    train = discrete_column(train, 'Age', Age_bins)

    test = discrete_column(test, 'Pregnancies', Pregnancies_bins)
    test = discrete_column(test, 'Glucose', Glucose_bins)
    test = discrete_column(test, 'BloodPressure', BloodPressure_bins)
    test = discrete_column(test, 'SkinThickness', SkinThickness_bins)
    test = discrete_column(test, 'Insulin', Insulin_bins)
    test = discrete_column(test, 'BMI', BMI_bins)
    test = discrete_column(test, 'DiabetesPedigreeFunction', DiabetesPedigreeFunction_bins)
    test = discrete_column(test, 'Age', Age_bins)

    col_name.remove('Outcome')

    return train.drop(col_name, axis=1), test.drop(col_name, axis=1)


def entropy_func(q):
    return -(q * math.log2(q))


def B_func(q):
    if q == 0 or q == 1:
        return 0
    return entropy_func(q) + entropy_func(1 - q)


def entropy_of_div(divDf, output_name):
    pk = len(divDf[divDf[output_name] == 1])
    nk = len(divDf[divDf[output_name] == 0])
    if (pk + nk == 0):
        return 0
    return B_func((pk / (pk + nk)))


def entropy_of_all_div(df, col_name, res_name, cutoff):
    leftDf = df[df[col_name] < cutoff]
    rightDf = df[df[col_name] >= cutoff]
    left_entropy = entropy_of_div(leftDf, res_name)
    right_entropy = entropy_of_div(rightDf, res_name)
    entropy_total = (len(leftDf) / len(df) * left_entropy) + (len(rightDf) / len(df) * right_entropy)
    return entropy_total


def gain(df, col_name, res_name, cutoff):
    all_p_count = len(df[df[res_name] == 1])
    all_n_count = len(df[df[res_name] == 0])
    current_entropy = B_func(all_p_count / (all_p_count + all_n_count))
    remainder = entropy_of_all_div(df, col_name, res_name, cutoff)
    gain_ = current_entropy - remainder
    return gain_, current_entropy, remainder


def pluralValue(df: pd.DataFrame, outcome_name):
    return df.mode()[outcome_name][0]


def pluarlity_value_node(df: pd.DataFrame, outcome_name):
    all_p_count = len(df[df[outcome_name] == 1])
    all_n_count = len(df) - all_p_count;
    entropy = B_func((all_p_count / (all_n_count + all_p_count)))

    return DecisionLeaf(df.mode()[outcome_name][0], entropy=entropy)


class DecisionFork:
    def __init__(self, attr, cutoff, default_child=None, entropy=None, gain_=None, remainder_=None):
        self.attr = attr
        self.cutoff = cutoff
        self.default_child = default_child
        self.left_node = None
        self.right_node = None
        self.entropy = entropy
        self.gain_ = gain_
        self.remainder_ = remainder_

    def __call__(self, example):

        attr_val = example[self.attr]
        if attr_val < self.cutoff:
            return self.left_node(example)
        else:
            return self.right_node(example)

    def add(self, subtree, isLeft):

        if (isLeft):
            self.left_node = subtree
        else:
            self.right_node = subtree

    def display(self, indent=0):
        name = self.attr
        print('Test', name)
        print(' ' * 4 * indent, name, '<', self.cutoff, '==>', end=' ')
        self.left_node.display(indent + 1)
        print(' ' * 4 * indent, name, '>=', self.cutoff, '==>', end=' ')
        self.right_node.display(indent + 1)

    def __str__(self):
        return str(self.attr + '\n' + str(self.cutoff) + '\n' + 'Entropy =' + str(self.entropy) + '\n' + 'Gain =' +
                   str(self.gain_) + '\n' + 'Remainder =' + str(self.remainder_))


class DecisionLeaf:

    def __init__(self, result, entropy=None):
        self.result = result
        self.entropy = entropy

    def __call__(self, example):
        return self.result

    def display(self, indent=0):
        print('RESULT =', self.result)

    def __str__(self):
        return ' ' + str(self.result) + '\n' + 'Entropy =' + str(self.entropy)


def chooseAttribute(df: pd.DataFrame, attributes, res_name):
    attributes_importance = {}
    entropy = {}
    remains = {}
    for att in attributes:
        possible_values = df[att].unique()
        for val in possible_values:
            index = (att, val)
            attributes_importance[index], entropy[index], remains[index] = gain(df, att, res_name, val)
    answer = max(attributes_importance.items(), key=operator.itemgetter(1))[0]
    return answer, attributes_importance[answer], entropy[answer], remains[answer]


def isAllSame(df: pd.DataFrame):
    a = df.to_numpy()
    return (a[0] == a[1:]).all()


def DecisionTreeLearning(examples: pd.DataFrame, attributes: list, parent_examples, outcome_name, curr_depth,
                         max_depth=6):
    if examples.empty:
        return pluarlity_value_node(parent_examples, outcome_name)
    if isAllSame(examples[outcome_name]):
        return DecisionLeaf(examples[outcome_name].iloc[0], entropy=0)
    if not attributes:
        return pluarlity_value_node(examples, outcome_name)
    if curr_depth == max_depth:
        return pluarlity_value_node(examples, outcome_name)
    cols = list(examples.columns)
    cols.remove(outcome_name)
    attr, gain_, entropy_, remains_ = chooseAttribute(examples, cols, outcome_name)
    if (entropy_ < 1e-15):
        return DecisionLeaf(examples[outcome_name].iloc[0], entropy=0)
    tree = DecisionFork(attr[0], attr[1], pluarlity_value_node(examples, outcome_name), gain_=gain_, entropy=entropy_,
                        remainder_=remains_)
    new_cols = attributes

    subtree_left = DecisionTreeLearning(examples[examples[attr[0]] < attr[1]], new_cols, examples, outcome_name,
                                        curr_depth + 1,
                                        max_depth)
    tree.add(subtree_left, True)
    subtree_right = DecisionTreeLearning(examples[examples[attr[0]] >= attr[1]], new_cols, examples, outcome_name,
                                         curr_depth + 1,
                                         max_depth)
    tree.add(subtree_right, False)
    return tree


def graphMaker(g, mytree):
    if isinstance(mytree, DecisionLeaf):
        myItem = g.newItem(mytree.__str__())
        return myItem
    elif isinstance(mytree, DecisionFork):
        myItem = g.newItem(mytree.__str__())

        left_tree = graphMaker(g, mytree.left_node)
        l = g.newLink(myItem, left_tree)
        g.propertyAppend(l, "color", "blue")
        g.propertyAppend(l, "label", '<')
        right_tree = graphMaker(g, mytree.right_node)
        l2 = g.newLink(myItem, right_tree)
        g.propertyAppend(l2, "color", "blue")
        g.propertyAppend(l2, "label", '>=')
    return myItem


def makeVisualGraph(mytree):
    g = gvgen.GvGen()
    graphMaker(g, mytree)
    string = ""
    myfile = open("output_graphviz.txt", 'w')

    g.dot(myfile)
    myfile.close()
    myfile = open("output_graphviz.txt", 'r')
    lines = myfile.readlines()[1:]
    for line in lines:
        string = string + line

    srcc = Source(string)
    srcc.render(view=True)


def correction_test(test, mytree):
    right_guess = 0

    for i in range(len(test)):
        test_res = mytree(test.iloc[i])
        if test_res == test.iloc[i].Outcome:
            right_guess += 1

    return (right_guess / len(test) * 100)


df = pd.read_csv('diabetes.csv')
train, test = train_test_split(df, test_size=0.5)
train, test = preprocess(train, test)

cols = list(df.columns)
cols.remove('Outcome')

mytree = DecisionTreeLearning(train, cols, None, 'Outcome', 0, 6)
#
#
makeVisualGraph(mytree)

print(correction_test(test,mytree))

print(correction_test(train,mytree))
