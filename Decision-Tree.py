import pandas as pd
import math
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import numpy as np

# Read in data
data = pd.read_csv('house_votes_84.csv')

# Shuffles the rows of the normalized data and returns a tuple with a training set and a test set in that order
def train_test_split():
    data_sample = data.sample(frac=1)
    return (data_sample.iloc[:348], data_sample.iloc[348:])

# Given a subset of the data, calculate the entropy
def calculate_entropy(subset):

    num_Democrats = 0
    num_Republicans = 0

    # Case where a size-1 dataframe is automatically converted to a list
    if type(subset) == list:
        for instance in subset:
            if instance == 0:
                num_Democrats += 1
            else:
                num_Republicans += 1
    else:
        for instance in subset['target']:
            if instance == 0:
                num_Democrats += 1
            else:
                num_Republicans += 1

    # Cases for if either class has 0 entries in the subset to avoid log domain error
    if num_Democrats == 0 and num_Republicans == 0:
        return 0
    elif num_Democrats == 0:
        return - (num_Republicans/len(subset) * math.log2(num_Republicans/len(subset)))
    elif num_Republicans == 0:
        return - (num_Democrats/len(subset) * math.log2(num_Democrats/len(subset)))

    return -(num_Democrats/len(subset) * math.log2(num_Democrats/len(subset))) - (num_Republicans/len(subset) * math.log2(num_Republicans/len(subset)))

# Given a subset of data and the partitions made by splitting on an attribute, return the information gain of that split
# This is used to find the best attribute to split the decision tree on
def calculate_information_gain(subset, partitions):
    
    partitions_size = sum(len(p) for p in partitions)
    entropies = []
    information = 0

    for partition in partitions:
        entropies.append(calculate_entropy(partition))

    for i in range(len(entropies)):
        information += entropies[i] * (len(partitions[i])/partitions_size)

    return calculate_entropy(subset) - information

# This node class is used during training to split on attributes and create the tree
class tree_node():

    def __init__(self):
        self.label = ''
        self.split_attribute = ''
        self.partitions = []
        self.children = [] # Array of tuples, the first value being the child node itself, and the second being the attribute value that leads to the node
        self.is_leaf = False
        self.attr_branch = 9
        self.parent = ''

    # Finds the best attribute to split on during a step in training
    def find_best_split(self, subset, attribute_list):

        criterion_vals = []

        for attr in attribute_list:
            yea_list = []
            nay_list = []
            no_vote_list = []
            for instance in subset.iterrows():
                if instance[1][attr] == 0:
                    no_vote_list.append(instance[1]['target'])
                if instance[1][attr] == 1:
                    nay_list.append(instance[1]['target'])
                if instance[1][attr] == 2:
                    yea_list.append(instance[1]['target'])
            
            criterion_vals.append( ( attr, calculate_information_gain(subset, [no_vote_list, nay_list, yea_list]) ) )
            self.split_attribute = max(criterion_vals, key=itemgetter(1))[0] 
        
        return self.split_attribute
        
# Given a new tree node, train the tree to be used for predictions
def train_tree(dataset, root):

    attribute_list = data.columns.to_list()[:16]

    # The recursive call of the function must be separated to reset the attribute list every time train_tree is used
    def train_tree_recur(dataset, root, attr_list):
        
        # Find the majority class in dataset for use in creating leaf nodes
        majority_class = ''
        num_Democrats = 0
        num_Republicans = 0
        for instance in dataset['target']:
            if instance == 0:
                num_Democrats += 1
            else:
                num_Republicans += 1
            if num_Democrats > num_Republicans:
                majority_class = 0
            elif num_Republicans > num_Democrats:
                majority_class = 1
            else:
                tie_breaker = random.randint(0, 1)
                if tie_breaker == 0:
                    majority_class = 0
                else:
                    majority_class = 1

        # Stopping Criteria

        # Stop if 75% or more of the current partition belongs to one class, and set the label as that class
        if num_Democrats > .75 * len(dataset.index):
            root.label = 0
            root.is_leaf = True
            return root
        elif num_Republicans > .75 * len(dataset.index):
            root.label = 1
            root.is_leaf = True
            return root
        
        # Stop if there are no more attributes left to test, and set the label as the current majority class
        if len(attr_list) == 0:
            root.label = majority_class
            root.is_leaf = True
            return root
        
        # Select best attribute to split and remove it from the list
        best_attr = root.find_best_split(dataset, attr_list)
        attr_list.remove(best_attr)
        
        # Create the partitions based on the best split attribute
        yea_partition = []
        nay_partition = []
        no_vote_partition = []

        for i in range(dataset[best_attr].size):
            if dataset[best_attr].values[i] == 0:
                no_vote_partition.append(dataset.iloc[i])
            elif dataset[best_attr].values[i] == 1:
                nay_partition.append(dataset.iloc[i])
            else:
                yea_partition.append(dataset.iloc[i])
        
        # Convert partition lists to dataframes for further operations
        yea_partition = pd.DataFrame(yea_partition)
        nay_partition = pd.DataFrame(nay_partition)
        no_vote_partition = pd.DataFrame(no_vote_partition)

        # Create leaf nodes and subtrees based on above partitions
        if len(yea_partition) == 0:
            yea_leaf = tree_node()
            yea_leaf.label = majority_class
            yea_leaf.is_leaf = True
            yea_leaf.attr_branch = 2
            root.children.append(yea_leaf)
        else:
            yea_child = train_tree_recur(yea_partition, tree_node(), attr_list)
            yea_child.attr_branch = 2
            root.children.append(yea_child)

        if len(nay_partition) == 0:
            nay_leaf = tree_node()
            nay_leaf.label = majority_class
            nay_leaf.is_leaf = True
            nay_leaf.attr_branch = 1
            root.children.append(nay_leaf)
        else:
            nay_child = train_tree_recur(nay_partition, tree_node(), attr_list)
            nay_child.attr_branch = 1
            root.children.append(nay_child)

        if len(no_vote_partition) == 0:
            no_vote_leaf = tree_node()
            no_vote_leaf.label = majority_class
            no_vote_leaf.is_leaf = True
            no_vote_leaf.attr_branch = 0
            root.children.append(no_vote_leaf)
        else:
            no_vote_child = train_tree_recur(no_vote_partition, tree_node(), attr_list)
            no_vote_child.attr_branch = 0
            root.children.append(no_vote_child)

        return root
    
    return train_tree_recur(dataset, root, attribute_list)

# Trains a tree with training data and uses it to classify test data
def make_predictions(training_set, test_set):

    predicted_classes = [] # Stores the final predictions of each instance as a tuple containing the indices and the predicted classes
    cur_node = train_tree(training_set, tree_node())

    for instance in test_set.iterrows():
        while True:
            if cur_node.is_leaf:
                predicted_classes.append((instance[0], cur_node.label))
                while cur_node.parent != '':
                    cur_node = cur_node.parent
                break
            else:
                # Go down the tree based on the best attribute
                attr_val = instance[1][cur_node.split_attribute] # The value of the attribut to split on
                for child in cur_node.children:
                    if child.attr_branch == attr_val:
                        child.parent = cur_node
                        cur_node = child

    assert(len(predicted_classes) == len(test_set.index)) # Check that the number of predictions made is equal to the number of rows in the test set

    return predicted_classes

# Computes how accurate the prediction algorithm is by training it on the training_set and testing it on the test_set
def compute_accuracy(training_set, test_set):
    number_correct = 0
    predictions = make_predictions(training_set, test_set)
    
    for i in range(len(predictions)):
        if test_set.iloc[i][16] == predictions[i][1]:
            number_correct += 1
    
    return number_correct / len(predictions)

# Creates 50 train-test splits, trains the data using the training data, and tests it on the test data
# The accuracies of each of these 50 trials are graphed and stored
def get_results():
    accuracies = []
    
    for i in range(50):
        train, test = train_test_split()
        accuracies.append(compute_accuracy(train, test))

    plt.plot(range(1, 51), accuracies)
    plt.xlabel("Trial Run Number")
    plt.ylabel("Accuracy")
    plt.title("Accuracies for 50 Trial Runs")
    plt.savefig("Plots/test_run_accuracies")
    plt.clf()

get_results()
