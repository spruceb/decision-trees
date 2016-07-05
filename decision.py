import itertools as it
import csv
import math
from collections import Counter
from collections import defaultdict as dfdict


class ClassificationTree:
    '''Class for generating classification decision trees and querying them

    Decision trees are a limited but simple machine learning method. They
    work on data sets where each data point has the same set of features
    and a single (finite domain) classification value. The idea is that
    given a set of training data, it is possible to produce a tree-structure
    of feature queries, the answers to which will give the correct 
    classification of any data in the training set. 

    Trivially the tree could simply have a distinct path for each data point
    in training, however this would not at all generalize to other data.
    Instead, information theory is used to pick the queries that give the most
    information about the classifications. The entropy of the original
    classification data measures, in some sense, how much information is
    necessary to find the classification. Then the information gain of each
    feature is measured (this is the entropy of the classification given
    the feature), and the one with the most "entropy gain", or difference
    from the total classification entropy, is the feature who's value tells
    us the most about the classification.

    Once this feature is found the process is repeated until there is a
    possible value for a feature such that all data points "down that path"
    have the same classification. This results in an answer node, of which
    there will likely be many. Additionally, if a path results in a single
    feature where no value has all the same classification, the classification 
    with the most occurrences for that "value path" is selected as the answer.

    This process results in a tree with two desirable features. First it
    is simple, in accordance with Occam's razor. Since it is selecting the
    option with most information gain at each point, it will model one of
    the simpler possible hypotheses (although not always the simplest, as
    finding that requires exponential computation time). Second, it shouldn't
    be too over-fitted, that is specific to the test data set. 

    However this approach is still limited by over-fitting, which will still
    occur to some degree. Also if some feature has a large number of different
    values, or even mostly unique values, this will result in a very large and
    probably inaccurate tree. It works best when each feature has a small,
    finite domain. This limitation also manifests in the fact that these trees
    cannot give a result if a query has a feature value that wasn't in the
    training set (without modification to the algorithm)
    '''

    def __init__(self, data):
        '''Sets up the required data and builds the tree

        `data` should a list of tuples of the form:
            `(feature_dict, classification)`
        where each tuple is a data point. `feature_dict` is a dictionary
        mapping names of features to values for the data point, and
        classification is the classification of that data point.

        In several places in this constructer, `dfdict`, or defaultdicts,
        are used. They work like normal dicts, except when a key without a
        value is accessed, they set that value to the result of calling
        the function they're passed instead of throwing a key error. This
        is useful for the various counting tasks being performed.

        Additionally, Counters dicts that behave in a similar way, except
        that it simply maps from values to their integer counts.
        '''

        self.data = data
        # using zip in this manner will separate the list of 2-tuples into
        # two lists of the first and second values of those tuples
        self.features, self.classifications = zip(*data)

        self.classification_domain = set(self.classifications)
        self.feature_indexes = list(self.features[0].keys())

        # all of these default dicts are set up to be filled in the for below
        
        # maps from features to their possible values
        self.feature_domains = dfdict(set)
        # maps from features to a Counter (which maps from the feature values
        # to the number of their occurrences)
        self.feature_value_counts = dfdict(Counter)
        # maps from features to their possible values to a Counter of the
        # classifications for data points with that value for that feature
        self.feature_classification_counts = dfdict(
            lambda: dfdict(Counter))
        for row, class_ in self.data:
            for key, value in row.items():
                self.feature_domains[key].add(value)
                self.feature_value_counts[key][value] += 1
                self.feature_classification_counts[key][value][class_] += 1

        self.classification_counts = dict(Counter(self.classifications))
        self.build_tree()

    def build_tree(self):
        '''Recursively builds question and answer nodes from the data

        This performs the algorithm described above. It finds the features
        with the most entropy gain, then creates new trees for each possible
        value that feature might have
        '''

        initial_entropy = self.entropy(self.classification_counts.values())
        feature_entropies = {feature: initial_entropy - self.feature_entropy(feature)
                             for feature in self.feature_indexes}
        # this gets the key with the maximum mapped value in features_entropies
        top_feature = max(feature_entropies, key=feature_entropies.get)

        children = {}        
        if len(feature_entropies) <= 1:
            # gets final answer nodes when there's only one feature left
            children = {value: AnswerNode(value, answer) for value, answer in
                        self.single_feature_answers(top_feature).items()}
        else:
            for possible_value in self.feature_domains[top_feature]:
                new_data = self.data_for_feature_value(
                    top_feature, possible_value)
                if all(row[1] == new_data[0][1] for row in new_data):
                    # all classifications are the same for this value
                    children[possible_value] = AnswerNode(possible_value,
                                                          new_data[0][1])
                else:
                    # recurses into a new tree with the filtered data
                    # uses it to find further nodes for this branch
                    new_tree = ClassificationTree(new_data)
                    children[possible_value] = new_tree.root
                    
        self.root = QuestionNode(top_feature, children)

    def single_feature_answers(self, feature_index):
        '''Returns the best classifications for the final feature values

        When the algorithm reaches a point where there is only one feature
        left, calculating entropy is pointless. Rather answers must be chosen
        for each possible value of the feature. Since some of them might not be
        perfect at this stage, each value is given the majority classification
        for data points with that feature value
        '''
        return {
            value: max(count.keys(), key=count.get)
            for value, count in
            self.feature_classification_counts[feature_index].items()
        }

    def data_for_feature_value(self, feature_index, feature_value):
        '''Returns data that has feature_value at feature_index

        When a feature is chosen, afterwards we need to create new trees
        for each value of that feature. Thus we choose the data points
        that have that value, and additionally strip that feature from all
        the feature dicts
        '''
        return tuple(
            ({k: v for k, v in features.items() if k != feature_index},
             class_)
            for features, class_ in self.data
            if features[feature_index] == feature_value)

    def feature_entropy(self, feature_index):
        '''Returns the entropy of the classification given a feature

        This is a conditional entropy value. Since entropy is closely tied to
        probability, this is analog us to the conditional probability of
        X given Y. In this case it's just the sum of the entropies of the
        data points where the feature has a certain value, weighted by
        the probability of that value.
        '''

        counts = self.feature_value_counts[feature_index]
        count_total = sum(counts.values())
        entropy = 0
        for value in self.feature_domains[feature_index]:
            value_p = counts[value] / count_total
            value_entropy = self.entropy(
                self.feature_classification_counts[
                    feature_index][value].values()
            )
            entropy += value_p * value_entropy
        return entropy

    def entropy(self, occurrence_counts):
        '''Calculates the entropy of a random variable with a finite domain

        Uses the definition of entropy, as found on Wikipedia or in various
        textbooks.
        `occurrence_counts` is a list of the number of times each distinct
        value occurs
        '''
        total = sum(occurence_counts)
        probabilities = (count / total for count in occurrrence_counts)
        return -sum(p * math.log2(p) for p in probabilities)

    def classify(self, element):
        '''Returns the result of querying the tree for the given data point

        `element` must be a dict with the same features self was trained on'''
        return self.root.answer(element)

    def __repr__(self):
        return str(self.root)


class QuestionNode:
    '''A tree node for a query of a feature'''

    def __init__(self, question_index, child_nodes):
        '''`child_nodes` is a dict from answer values to child nodes

        child nodes can be either answers or more questions
        '''
        self.children = child_nodes
        self.question_index = question_index

    def answer(self, element):
        '''Recursively returns the answer to this node based on the feature dict passed'''
        next_node = self.children[element[self.question_index]]
        return next_node.answer(element)

    def __repr__(self):
        return "Question(query: {},\n \tanswers: {})\n".format(
            self.question_index,
            self.children)


class AnswerNode:
    '''A tree node for a classification for a given feature value'''

    def __init__(self, feature_value, answer):
        self.feature_value = feature_value
        self.answer_value = answer

    def answer(self, *args):
        '''Returns the answer

        Called in `QuestionNode.answer`, which is why there's unused arguments
        '''
        return self.answer_value

    def __repr__(self):
        return "Answer({}: {})".format(self.feature_value, self.answer_value)

    
def dict_reader_to_data(dict_reader):
    '''Converts a `csv.DictReader` to classification tree data'''
    class_key = dict_reader.fieldnames[-1]
    return [({k: v for k, v in row.items() if k != class_key}, row[class_key])
            for row in dict_reader]


def test_files(training_file, testing_file):
    '''Generates a classifier from `training` and tests it on `testing_file`'''
    with open(training_file, 'r') as f:
        csv_dicts = csv.DictReader(f)
        data = dict_reader_to_data(csv_dicts)
        tree = ClassificationTree(data)
    with open(testing_file, 'r') as f:
        csv_dicts = csv.DictReader(f)
        data = dict_reader_to_data(csv_dicts)
        features, _ = zip(*data)
        for feature_dict, classification in data:
            result = tree.classify(feature_dict)
            print(result == classification,
                  "| result {} : answer {}".format(result, classification))

if __name__ == '__main__':
    test_files('mushroom_train.csv', 'mushroom_test.csv')
