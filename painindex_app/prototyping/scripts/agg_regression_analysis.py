"""This is a third pass at gauging pain intensity from google search results.

We use a regression now on aggregated search results for each insect.
Each search result is expressed as a feature vector of word frequencies.

y = pain intensity of the corresponding bug
X = vector of word frequencies in the result, for each of the 
    most common words across dataset, normalized by instances of "the"


TODO: use a NLP library to parse results, identify n-grams. Porter stemmer.
We are currently only using individual words. 
"""


import json
import re
from sklearn import linear_model
import numpy as np
from collections import defaultdict
import random
from pprint import pprint
from nltk import stem


def main():
    # with open('../data/outputs/pains_20140910.txt') as json_pains:
    with open('../data/outputs/pains_20140928.txt') as json_pains:
        pains = json.load(json_pains)

    # with open('../data/outputs/search_results_20140910.txt') as json_search_results:
    with open('../data/outputs/search_results_20140928_painful.txt') as json_search_results:
        search_results = json.load(json_search_results)

    # Sensitivity analysis: see what happens if we change the number of examples.
    # m = 100
    # search_results = {pain: search_results[pain][:m] for pain in search_results}

    # Get rid of pain sources with few search results.
    rmin = 20
    search_results = {k:v  for k, v in search_results.items() if len(v) > rmin}

    results_train, results_test = split_data(search_results, 
        split_frac=0.6, seed=461234)

    results_train_wds = wordified(results_train)
    results_test_wds = wordified(results_test)

    common_wds = get_common_words(results_train_wds, 10000)
    print "FEATURE VECTOR LENGTH:", len(common_wds)

    # Each search result is represented by a vector x with 
    #   x[i] = num times the ith word of common_wds appears in the result.
    results_train_features = make_features(results_train_wds, common_wds)
    results_test_features = make_features(results_test_wds, common_wds)

    # Create the dataset: We treat each result as a standalone feature.
    X_train, y_train, pains_train = make_data(results_train_features, pains)
    X_test, y_test, pains_test = make_data(results_test_features, pains)

    # Ad hoc evaluation of model performance: avg squared difference between
    # predicted and true pain intensities.
    def eval_performance(X, y, painsources, trainedmodel): 
        # print "EVALUATING PERFORMANCE"
        diffs = []
        for i, pain in enumerate(painsources):
            this_x = X[i]
            this_y = y[i]

            pain_predicted = trainedmodel.predict([this_x])
            pain_true = this_y
            diff = pain_predicted - pain_true

            print "Pain =", pain
            print "True, predicted, diff =", [pain_true, pain_predicted[0], diff[0]]

            diffs.append(diff)

        avg_sq_diff = sum(x**2 for x in diffs)/len(diffs)
        print "Avg squared diff:", avg_sq_diff


    # Run LASSO or RIDGE for a variety of alphas
    # alphas = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    alphas = [0.3]

    for alpha in alphas:
        print "\nalpha = %s:" % alpha

        # SELECT LASSO OR RIDGE
        reg = linear_model.Ridge(alpha=alpha)
        # reg = linear_model.Lasso(alpha=alpha)

        model_train = reg.fit(X_train, y_train)
        y_pred_train = model_train.predict(X_train) 
        y_pred_test = model_train.predict(X_test)

        # See the weights on the feature vector words:
        weights = zip(common_wds, model_train.coef_)
        weights = sorted(weights, key=lambda x: x[1], reverse=True)
        print "WEIGHTS:"
        pprint(weights[:30])
        pprint(weights[-10:])

        eval_performance(X_train, y_train, pains_train, model_train)
        eval_performance(X_test, y_test, pains_test, model_train)



##########################################################################


    
def split_data(results, split_frac, seed):
    """Return results split into two pieces according to split_frac.
        Pain names are shuffled according to seed.
    """
    pain_names = results.keys()
    random.seed(seed)
    random.shuffle(pain_names)

    num_pains_train = int( split_frac*len(pain_names) )
    pains_train = pain_names[:num_pains_train]
    pains_test = pain_names[num_pains_train:]

    results_train = {pain: results[pain] for pain in pains_train}
    results_test = {pain: results[pain] for pain in pains_test}

    return results_train, results_test


def wordified(search_results):
    """Return a dict with key=pain, val=list of all words in all results for that pain.
    """
    # We strip out each word of each pain name, so the training process
    # cannot rely on essentially knowing the name of the bug.
    pain_wds = set([wd.lower() for pain in search_results 
        for wd in re.findall(r'\b\w+\b', pain)])

    # Split each search result into words.
    wordified1 =  {pain: 
        [ get_words(result['text'], excluded=pain_wds) for result in results ]
        for pain, results in search_results.items()}

    # Combine all search results for a given pain.
    wordified2 = {pain: [wd for result in results for wd in result] 
        for pain, results in wordified1.items()}

    return wordified2



def get_words(text, excluded):
    """Return a processed list of words in text, minus excluded words.
    """
    porter = stem.porter.PorterStemmer()

    wds = [wd.lower() for wd in re.findall(r'\b\w+\b', text)]
    stems = [porter.stem(wd) for wd in wds if wd not in excluded]

    # Let's see what happens if we include bigrams:
    # bigrams = [(stems[i] + ' ' + stems[i+1]) for i in range(len(stems)-1)]
    # stems.extend(bigrams)

    return stems

def get_common_words(results_wordified, num_wds):
    "Return a list of the num_wds most common words across wordified results."
    common_wds = {}
    for wordlist in results_wordified.values():
        for wd in wordlist:
            common_wds[wd] = common_wds.get(wd, 0) + 1
    sorted_wds = sorted(common_wds.items(), key=lambda x: x[1], reverse=True)

    return [wd[0] for wd in sorted_wds[:num_wds]]

def make_features(results_train_wds, common_wds):
    """Convert each result into a feature vector x.
       x[i] = num times the ith word of common_wds appears in the result.
    """
    return {pain: make_feature(results, common_wds)
        for pain, results in results_train_wds.items()}

def make_feature(wordlist, common_wds):
    """Turn a list of words into a feature vector.
    We normalize by instances of "the".
    """
    the_count = wordlist.count("the")
    if the_count == 0:
        # This could happen if a pain has very few (or even zero) search results.
        # We should probably return None and exclude the pain from the model.
        the_count = 1
    return [wordlist.count(wd) / float(the_count) for wd in common_wds]

def make_data(results_train_features, pains):
    """Create a dataset of X, y.
    Each row of X is the feature for a single pain with corresponding intensity y.
    In addition to X and y, we return a vector painsources which
    records the pain name for each example.
    """
    X, y, painsources = [], [], []
    for pain, feature in results_train_features.items():
        y.append(pains[pain])
        X.append(feature)
        painsources.append(pain)

    return X, y, painsources







# # This is left over from sentiment_analysis. We will want a corresponding
# # function though:
# def make_pain_predictor(pains, word_sentiments, pain_sentiments, degree):
#     """Return a function that makes pain predictions for any set of results.
#     Results are in their original dictionary form (with 'text' field).
#     The function returns a dict of pain name: predicted pain level.

#     This function is trained using word_sentiments and pain_sentiments.
#     degree is the degree of the polynomial to fit with sentiment_converter.
#     """
#     # f will convert pain_sentiment into predicted pain level.
#     # This is necessary because sentiment itself does not equal predicted pain.
#     # sentiment_converter runs a regression to determine coefficients
#     # for the transformation, so it's important that the conversion is only
#     # run once and then used in every call to pain_predictor below.
#     f = sentiment_converter(pains, pain_sentiments, degree=degree)

#     def pain_predictor(results):
#         sentiments = find_pain_sentiments(wordified(results), word_sentiments)

#         # If want to include keys with None predictions:
#         # predicted_pain = {}
#         # for pain in sentiments:
#         #     if sentiments[pain] is None:
#         #         predicted_pain[pain] = None
#         #     else:
#         #         predicted_pain[pain] = f(sentiments[pain])
        
#         # If want to omit keys with None predictions:
#         predicted_pain = {pain: f(sentiments[pain]) for pain in sentiments
#             if sentiments[pain] is not None}
#         return predicted_pain

#     return pain_predictor


# def sentiment_converter(pains, pain_sentiments, degree):
#     """Return a function which converts pain sentiment into predicted pain.
#     The sentiment scores do not directly correspond to the original
#     pain scale. We regress true pain on sentiments, with higher order terms
#     up to the degree'th degree of sentiment,
#     to produce a function that converts sentiment to predicted true pain. 
#     """

#     def vectorized(sent):
#         "Turn a sentiment into a feature vector of higher order terms."
#         if sent is None: 
#             return None
#         return np.array([sent**p for p in range(0, degree+1)])

#     X, y = [], []

#     for pain, sent in pain_sentiments.items():
#         # Each example has terms 1, x, x**2, ..., x**degree
#         x = vectorized(sent)
#         if x is None: 
#             continue
#         X.append(x)
#         y.append(pains[pain])

#     # I have included an intercept manually, no need to add one.
#     LM = linear_model.LinearRegression(fit_intercept=False)
#     LMfit = LM.fit(X, y)

#     # Return a function that gives the fitted value for a sentiment s.
#     return lambda s: vectorized(s).dot(LMfit.coef_)


if __name__ == '__main__':
    main()
