"""This is a second pass at gauging pain intensity from google search results.

We try ridge and lasso regressions.
Each search result is expressed as a feature vector of the
words it does and does not contain.

First pass: consider each search result as a separate data point.
y = pain intensity of the corresponding bug
X = vector of frequencies in the result, for each of the 
    most common words across dataset.

Alternatives: aggregate the words about each bug.
And/or use some complex aggregation scheme. For instance, it may be that half
the results about a given bug are bland scientific prose, while the other half
are actually about the pain. If only 10 percent of results discuss screams
of agony, that may be sufficient to predict high pain, rather than getting it
diluted.

TODO: use a NLP library to parse results, identify n-grams. Porter stemmer.
We are currently only using individual words. 
"""


import json
import re
from sklearn import linear_model
import numpy as np
from collections import defaultdict
import random


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
    rmin = 5
    search_results = {k:v  for k, v in search_results.items() if len(v) > rmin}

    results_train, results_test = split_data(search_results, 
        split_frac=0.6, seed=46)

    results_train_wds = wordified(results_train)
    results_test_wds = wordified(results_test)

    common_wds = get_common_words(results_train_wds, 100)

    # Each search result is represented by a vector x with 
    #   x[i] = num times the ith word of common_wds appears in the result.
    results_train_features = make_features(results_train_wds, common_wds)
    results_test_features = make_features(results_test_wds, common_wds)

    # Create the dataset: We treat each result as a standalone feature.
    X_train, y_train, pains_train = make_data(results_train_features, pains)
    X_test, y_test, pains_test = make_data(results_test_features, pains)

    # Fit the Ridge Regression on the training data.
    ridge_reg = linear_model.Ridge(alpha=1)
    model_train_ridge = ridge_reg.fit(X_train, y_train)
    # print model_train_ridge.coef_    
    # fitted_train = model_train_ridge.predict(X_train)
    # print fitted_train[:40]
    # Really not sure what score does...doesn't seem to work as I thought.
    # print model_train_ridge.score(X_train, y_train)
    # print model_train_ridge.score(X_test, y_test)
    y_pred_train_ridge = model_train_ridge.predict(X_train) 
    y_pred_test_ridge = model_train_ridge.predict(X_test)

    print "RIDGE:"
    print ("r^2 on train data : %f" % 
        (1 - np.linalg.norm(y_train - y_pred_train_ridge) ** 2
            / np.linalg.norm(y_train) ** 2)
    )
    print ("r^2 on test data : %f" % 
        (1 - np.linalg.norm(y_test - y_pred_test_ridge) ** 2
            / np.linalg.norm(y_test) ** 2)
    )


    # Try Lasso Regression:
    lasso_reg = linear_model.Lasso(alpha=0)
    model_train_lasso = lasso_reg.fit(X_train, y_train)
    y_pred_train_lasso = model_train_lasso.predict(X_train) 
    y_pred_test_lasso = model_train_lasso.predict(X_test)

    # r**2 isn't really what we want, at the level of the individual result.
    # But this is a quick metric; see below for the aggregated results.
    print "LASSO:"
    print ("r^2 on train data : %f" % 
        (1 - np.linalg.norm(y_train - y_pred_train_lasso) ** 2
            / np.linalg.norm(y_train) ** 2)
    )
    print ("r^2 on test data : %f" % 
        (1 - np.linalg.norm(y_test - y_pred_test_lasso) ** 2
            / np.linalg.norm(y_test) ** 2)
    )


    # CODE BORROWED FROM linreg_textdata.py:
    def eval_performance(X, y, painsources, trainedmodel): 
        print "\nEVALUATING PERFORMANCE"       
        diff = []
        for pain in set(painsources):

            # Extract the indices corresponding to this pain
            indices = [i for (i, x) in zip(range(len(painsources)), painsources)
                if x == pain]
            # Examples of the same pain are contiguous.
            a, b = indices[0], indices[-1]+1
            pain_predicted = trainedmodel.predict(X[a:b])
            pain_true = y[a:b]

            # We want the avg prediction for each pain source
            pain_predicted_avg = pain_predicted.mean()
            pain_true = pain_true[0]
            avg_diff = (pain_predicted_avg - pain_true)

            print "Pain source:", pain
            print "True, predicted, avg diff:", (pain_true, pain_predicted_avg, avg_diff)
            diff.append(avg_diff)

        # print diff
        avg_sq_diff = sum(x**2 for x in diff)/len(diff)
        print "Avg squared diff:", avg_sq_diff

    print "\nAggregated performance for RIDGE REG:"
    eval_performance(X_train, y_train, pains_train, model_train_ridge)
    eval_performance(X_test, y_test, pains_test, model_train_ridge)
    
    print "\nAggregated performance for LASSO REG:"
    eval_performance(X_train, y_train, pains_train, model_train_lasso)
    eval_performance(X_test, y_test, pains_test, model_train_lasso)

    # Diagnosis: It looks like the ridge and lasso are BOTH flatlining on
    # the test set. 
    # When I set alpha=0 for lasso, i.e. regular OLS, lasso shows clear signs
    # of fitting the training data but it's random noise on the test set.
    # Even on the training data, it does appear muted toward the average.
    # Think about this...how does this compare to the mutedness of the
    # sentiment analysis exercise?

    # I think performance will be much better with a less noisy feature:
    # aggregated text for each insect.
    # Right now, individual features are so noisy that
    # the solver will be highly penalized every time an individual
    # result is way too high or way too low, so the safe thing is to 
    # give roughly the same weight to every word.





    
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
    """Return a copy of search_results with each result turned into a wordlist.

        In other words, return a dict with key=pain name, val=list of results 
        where each result is now a list of the words the result contains.
    """
    # We strip out each word of each pain name, so the training process
    # cannot rely on essentially knowing the name of the bug.
    pain_wds = set([wd.lower() for pain in search_results 
        for wd in pain.split(' ')])
    return {pain: [ get_words(result['text'], excluded=pain_wds) for result in results ]
        for pain, results in search_results.items()}


def get_words(text, excluded):
    """Return a processed list of words in text, minus excluded words.
    """
    # TODO: Use nltk to do this properly.
    # Or at least regex
    # e.g. re.findall(r'\b\w+\b', text)

    return [wd for wd in text.lower().split() if wd not in excluded]

def get_common_words(results_wordified, num_wds):
    "Return a list of the num_wds most common words across wordified results."
    common_wds = {}
    for results in results_wordified.values():
        for result in results:
            for wd in result:
                common_wds[wd] = common_wds.get(wd, 0) + 1
    sorted_wds = sorted(common_wds.items(), key=lambda x: x[1], reverse=True)

    return [wd[0] for wd in sorted_wds[:num_wds]]

def make_features(results_train_wds, common_wds):
    """Convert each result into a feature vector x.
       x[i] = num times the ith word of common_wds appears in the result.
    """
    return {pain: [make_feature(result, common_wds) for result in results]
        for pain, results in results_train_wds.items()}

def make_feature(wordlist, common_wds):
    "Turn a list of words into a feature vector."
    return [wordlist.count(wd) for wd in common_wds]

def make_data(results_train_features, pains):
    """Create a flattened dataset of X, y.
    Each row of X is a single feature with corresponding y = intensity.


    In addition to X and y, we return a vector painsources which
    records the pain name for each example.
    """
    X, y, painsources = [], [], []
    for pain, results in results_train_features.items():
        for result in results:
            y.append(pains[pain])
            X.append(result)
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
