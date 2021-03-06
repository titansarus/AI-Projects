import pandas as pd
import math
import operator
import gvgen
from graphviz import Source


def entropy_func(q):
    return -(q * math.log2(q))


def B_func(q):
    if q == 0 or q == 1:
        return 0
    return entropy_func(q) + entropy_func(1 - q)


def remainder(df: pd.DataFrame, col_name, res_name):
    all_p_count = len(df[df[res_name] == 1])
    all_n_count = len(df) - all_p_count;
    all_values = df[col_name].unique()
    s = 0.0
    for k in all_values:
        dcol = df[df[col_name] == k]
        pk = len(dcol[dcol[res_name] == 1])
        nk = len(dcol[dcol[res_name] == 0])
        s = s + ((pk + nk) / (all_p_count + all_n_count) * B_func((pk / (pk + nk))))
    return s

def gain(df: pd.DataFrame, col_name, res_name):
    all_p_count = len(df[df[res_name] == 1])
    all_n_count = len(df) - all_p_count;
    entrop = B_func((all_p_count / (all_n_count + all_p_count)))
    rem = remainder(df, col_name, res_name)
    return B_func((all_p_count / (all_n_count + all_p_count))) - remainder(df, col_name, res_name), entrop, rem


def chooseAttribute(df, attributes, res_name):
    attributes_importance = {}
    entropy = {}
    remains = {}
    for attr in attributes:
        attributes_importance[attr], entropy[attr], remains[attr] = gain(df, attr, res_name)
    answer = max(attributes_importance.items(), key=operator.itemgetter(1))[0]
    return answer, attributes_importance[answer], entropy[answer], remains[answer]


def pluralValue(df: pd.DataFrame, outcome_name):
    return df.mode()[outcome_name][0]


def isAllSame(df: pd.DataFrame):
    a = df.to_numpy()
    return (a[0] == a[1:]).all()



class DecisionFork:
    def __init__(self, attr, default_child=None, branches=None, entropy=None, gain_=None, remainder_=None):
        self.attr = attr
        self.default_child = default_child
        self.branches = branches or {}
        self.entropy = entropy
        self.gain_ = gain_
        self.remainder_ = remainder_

    def __call__(self, example):

        attr_val = example[self.attr]
        if attr_val in self.branches:
            return self.branches[attr_val](example)
        else:

            return self.default_child(example)

    def add(self, val, subtree):

        self.branches[val] = subtree

    def display(self, indent=0):
        name = self.attr
        print('Test', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)

    def __str__(self):
        return str(self.attr + '\n' + 'Entropy =' + str(self.entropy) + '\n' + 'Gain =' +
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




def pluarlity_value_node(df: pd.DataFrame, outcome_name):
    all_p_count = len(df[df[outcome_name] == 1])
    all_n_count = len(df) - all_p_count;
    entropy = B_func((all_p_count / (all_n_count + all_p_count)))

    return DecisionLeaf(df.mode()[outcome_name][0], entropy=entropy)


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
    all_values = examples[attr].unique()
    tree = DecisionFork(attr, pluarlity_value_node(examples, outcome_name), gain_=gain_, entropy=entropy_,
                        remainder_=remains_)
    for vk in all_values:
        new_cols = attributes
        if (attr in new_cols):
            new_cols.remove(attr)
        subtree = DecisionTreeLearning(examples[examples[attr] == vk], new_cols, examples, outcome_name, curr_depth + 1,
                                       max_depth)
        tree.add(vk, subtree)
    return tree


def graphMaker(g, mytree):
    if isinstance(mytree, DecisionLeaf):
        myItem = g.newItem(mytree.__str__())
        return myItem
    elif isinstance(mytree, DecisionFork):
        myItem = g.newItem(mytree.__str__())
        for key, val in mytree.branches.items():
            newTree = graphMaker(g, val)
            l = g.newLink(myItem, newTree)
            g.propertyAppend(l, "color", "blue")
            g.propertyAppend(l, "label", key)
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


df = pd.read_csv('restaurant.csv')

cols = list(df.columns)
cols.remove('Outcome')

mytree = DecisionTreeLearning(df, cols, None, 'Outcome', 0, 6)
makeVisualGraph(mytree)
