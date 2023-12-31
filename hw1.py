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
    corpus_file = open(corpus_path, 'r')
    
    # with open(corpus_path, "r") as corpus_file:
    for line in corpus_file:
        line = line.strip()
        words = line.split('\t')
        snippet = words[0].split()
        label = int(words[1])
        result.append((snippet, label))
    corpus_file.close()
    return result


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word.endswith("n't") or word.lower() in negation_words:
        return True
    return False


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    temp_string = " ".join(snippet) #convert the snippet to a single string
    tagged_string = nltk.pos_tag(nltk.word_tokenize(temp_string))
    negation_flag = False
    for i, word in enumerate(snippet):
        if negation_flag:
            # find the ext negation word if the condition is true
            if word in sentence_enders or word in negation_enders or tagged_string[i][-1] in set(["JJR", "RBR"]):
                negation_flag = False
                continue
            snippet[i] = "NOT_" + snippet[i]
        if is_negation(word):
            if i < len(snippet) - 1 and word == "not" and snippet[i+1] == "only":
                continue
            # flip the flag
            negation_flag =  not negation_flag
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
        if word not in feature_dict: continue
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
        X[i,:], Y[i]= vectorize_snippet(corpus[i][0], feature_dict), corpus[i][1]
    return (X, Y)


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    for col in range(X.shape[1]):
        max_val, min_val = max(X[:,col]), min(X[:, col])
        if max_val == min_val:
            X[:,col] = 0
        else:
            X[:,col] = (X[:,col] - min_val) / (max_val - min_val)


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    corpus = load_corpus(corpus_path)
    for i in range(len(corpus)):
        corpus[i]= (tag_negation(corpus[i][0]), corpus[i][1])
    feature_dict = get_feature_dictionary(corpus)
    vector = vectorize_corpus(corpus, feature_dict)
    normalize(vector[0])
    model = LogisticRegression()
    model.fit(vector[0], vector[1])
    return (model, feature_dict)
    


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    length = len(Y_test)
    TP = FP = FN = 0
    for i in range(length):
        if Y_test[i] == 1:
            if Y_pred[i] == 1:
                TP += 1
            else:
                FN += 1
        elif Y_test[i] == 0 and Y_pred[i] == 1:
            FP += 1
    p = TP/(TP+FP) if TP+FP else 0
    r = TP/(TP+FN) if TP+FN else 0
    f = 2*p*r/(p+r) if p+r else 0
    return (p, r, f)
                


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    corpus = load_corpus(corpus_path)
    for i in range(len(corpus)):
        corpus[i]= (tag_negation(corpus[i][0]), corpus[i][1])
    (X, Y_test) = vectorize_corpus(corpus, feature_dict)
    normalize(X)
    Y_pred = model.predict(X)
    return evaluate_predictions(Y_pred, Y_test)


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    feature_list = []
    for i in range(len(logreg_model.coef_[0])):
        feature_list.append((i, logreg_model.coef_[0][i]))
    top_k_features = sorted(feature_list, key=lambda x: abs(x[1]), reverse=True)[:k]
    feature_dict_val_to_key = {val:key for key,val in feature_dict.items()}
    top_k_features_with_word = [(feature_dict_val_to_key[i], weight) for i, weight in top_k_features]
    return top_k_features_with_word
        


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
