import re
import sys

import nltk
import numpy
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    result = []
    
    with open(corpus_path, "r") as corpus:
        for line in file:
            line = line.strip()
            words = line.split()
            snippet = words[:-1]
            lable = int(words[-1])
            result.append((snippet, label))
    return result


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word.endswith("-n't") or word in negation_words:
        return True
    return False


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    temp_string = " ".join(snippet) #convert the snippet to a single string
    tagged_string = nltk.pos_tag(nltk.tokenize(temp_string))
    length = len(snippet)
    negation_index = 0
    # find the negation word
    while not is_negation(snippet[negation_index]):
        negation_index += 1
    # no tagging if the case is "not only"
    if snippet[negation_index] == "not" and snippet[negation_index+1] == "only":
        return snippet
    tagging_pos = negation_index + 1
    while tagging_pos < length:
        # break if find either sentence-ending punctuation, a negation-ending word, or a comparative
        if snippet[tagging_pos] in negation_enders or snippet[tagging_pos] in sentence_enders or (tagged_string[tagging_pos][-1] in set(["JJR", "RBR"])):
            break
        snippet[tagging_pos] = "NOT_" + snippet[tagging_pos]
        tagging_pos += 1
    return snippet
        
        
        


# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    feature_dict = {}
    counter = 0
    for sentence in corpus:
        for word in sentence[0]:
            if word not in feature_dict:
                feature_dict[word] = counter
                counter += 1
    return feature_dict
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    feature_vector = numpy.zeros(len(feature_dict))
    for word in snippet:
        feature_vector[feature_dict[word]] += 1
    return feature_vector


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    X = numpy.empty([len(corpus), len(feature_dict)])
    Y = numpy.empty(len(corpus))
    for i in range(len(corpus)):
        X[i,], Y[i]= vectorize_snippet(i[0]), i[1]
    return (X, Y)


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    for col in range(X.shape[1]):
        max_val, min= max(X[,col]), min(X[, col])



# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    pass


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    pass


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    pass


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    pass


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
