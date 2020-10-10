#from __future__ import division
from math import log, exp
#from operator import mul
from collections import Counter
import os
import pylab
import _pickle as cPickle


CDATA_FILE = "countdata.pickle"
FDATA_FILE = "reduceddata.pickle"

class MyDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key) 
        return 0

bestKfeatures = set()   
pos_dict = MyDict()
neg_dict = MyDict()
totals = [0, 0]


def negation_handling(text):
    words = text.split()
    
    delims = "?.,!:;"
    final_list = []
    negation_counter = False 
    
    for w in words:
        st = w.strip(delims).lower()
        neg_word = "not_" + st if negation_counter else st
        final_list.append(neg_word)
        
        if any(neg in w for neg in ["not", "n't", "no"]):
            negation_counter = not negation_counter

        if any(c in w for c in delims):
            negation_counter = False

    return final_list

# Remove features that appear only once.
def refine_features():
    
    global pos_dict, neg_dict

    for key in list(pos_dict): 
        if pos_dict[key] <= 1 and neg_dict[key] <= 1: 
            del pos_dict[key]

    for key in list(neg_dict): 
        if neg_dict[key] <= 1 and pos_dict[key] <= 1:
            del neg_dict[key]

def train_model():
    global pos_dict, neg_dict, totals
    #max_limit = 12500

    limit = 5000
    
    for file in os.listdir("./aclImdb/train/pos/")[:limit]:
        for key in set(negation_handling(open("./aclImdb/train/pos/" + file, encoding="utf-8").read())):
            pos_dict[key] += 1
            neg_dict['not_' + key] += 1
    for file in os.listdir("./aclImdb/train/neg")[:limit]:
        for key in set(negation_handling(open("./aclImdb/train/neg/" + file, encoding="utf-8").read())):
            neg_dict[key] += 1
            pos_dict['not_' + key] += 1
    
    refine_features()

    totals[0] = sum(pos_dict.values())
    totals[1] = sum(neg_dict.values())
    
    countdata = (pos_dict, neg_dict, totals)
    cPickle.dump(countdata, open(CDATA_FILE, 'wb'))

# Compute the weighted mutual information of a word.
def mutual_information(word):
    T = totals[0] + totals[1]
    W = pos_dict[word] + neg_dict[word]

    I = 0
    if W==0:
        return 0

    if neg_dict[word] > 0:
        I += (totals[1] - neg_dict[word]) / T * log ((totals[1] - neg_dict[word]) * T / (T - W) / totals[1])
        I += neg_dict[word] / T * log (neg_dict[word] * T / W / totals[1])

    if pos_dict[word] > 0:
        I += (totals[0] - pos_dict[word]) / T * log ((totals[0] - pos_dict[word]) * T / (T - W) / totals[0])
        I += pos_dict[word] / T * log (pos_dict[word] * T / W / totals[0])
    
    return I

# Probability that word occurs in pos_dict documents using list of bestKfeatures
def isDocumentPositive(text):
    words = set(word for word in negation_handling(text) if word in bestKfeatures)
    
    if (len(words) == 0): 
        return True
    
    pos_prob = sum(log((pos_dict[word] + 1) / (2 * totals[0])) for word in words)
    neg_prob = sum(log((neg_dict[word] + 1) / (2 * totals[1])) for word in words)
    return pos_prob > neg_prob

# Returns best k features from positive and negative dictionaries.
def get_relevant_features():
    pos_dump = MyDict({k: pos_dict[k] for k in pos_dict if k in bestKfeatures})
    neg_dump = MyDict({k: neg_dict[k] for k in neg_dict if k in bestKfeatures})
    totals_dump = [sum(pos_dump.values()), sum(neg_dump.values())]
    return (pos_dump, neg_dump, totals_dump)

# Select top k features
def feature_selection_trials():
    
    global pos_dict, neg_dict, totals, bestKfeatures
    
    words = list(set(list(pos_dict)+list(neg_dict)))
    print ("Total no of features:", len(words))
    print("Accuracy of each 500 set of features in a range of 8000 to 40,000")
    words.sort(key=lambda x: -mutual_information(x))
    num_features, accuracy = [], []
    bestk = 0
    limit = 500
    path = "./aclImdb/test/"
    step = 500
    start = 8000
    best_accuracy = 0.0

    for w in words[:start]:
        bestKfeatures.add(w)

    for k in range(start, 40000, step):
        for w in words[k:k+step]:
            bestKfeatures.add(w)

        correct_pred = 0
        total_attempt = 0

        for file in os.listdir(path + "pos")[:limit]:
            correct_pred += isDocumentPositive(open(path + "pos/" + file, encoding="utf-8").read()) == True
            total_attempt += 1

        for file in os.listdir(path + "neg")[:limit]:
            correct_pred += isDocumentPositive(open(path + "neg/" + file, encoding="utf-8").read()) == False
            total_attempt += 1

        num_features.append(k+step)
        accuracy.append(correct_pred / total_attempt)

        if (correct_pred / total_attempt) > best_accuracy:
            bestk = k
        print (k+step, correct_pred / total_attempt)

    bestKfeatures = set(words[:bestk])

    cPickle.dump(get_relevant_features(), open(FDATA_FILE, 'wb'))


    pylab.plot(num_features, accuracy)
    pylab.show()

# Prints accuracy of model.
def calculate_accuracy():

    path = "./aclImdb/test/"
    limit = 2000

    correct_pred = 0
    total_attempt = 0
    

    for file in os.listdir(path + "pos")[:limit]:
        correct_pred += isDocumentPositive(open(path + "pos/" + file, encoding="utf-8").read()) == True
        total_attempt += 1
    for file in os.listdir(path + "neg")[:limit]:
        correct_pred += isDocumentPositive(open(path + "neg/" + file, encoding="utf-8").read()) == False
        total_attempt += 1

    print("Total number of correct prediction = %d and Total number of trials = %d " %(correct_pred, total_attempt))
    
    accuracy = correct_pred/total_attempt
    print("Accuracy = %0.4f " % accuracy)


# Prints if the sent document is Postive or Negative using created dictionaries
def model_demo(text):
    pprob, nprob = 0, 0

    words = set(word for word in negation_handling(text) if word in pos_dict or word in neg_dict)

    if (len(words) == 0): 
        print ("No words to compare with")
        return True

    
    for word in words:
        pp = log((pos_dict[word] + 1) / (2 * totals[0]))
        np = log((neg_dict[word] + 1) / (2 * totals[1]))

        print ("%15s %.9f %.9f" % (word, exp(pp), exp(np)))

        pprob += pp
        nprob += np

    print ("Positive" if pprob > nprob else "Negative", "log-diff = %.9f" % abs(pprob - nprob))


if __name__ == '__main__':
    train_model()
    feature_selection_trials()
    calculate_accuracy()
    print("Positive case example")
    model_demo(open("pos_example").read())
    print("Negative case example")
    model_demo(open("neg_example").read())
