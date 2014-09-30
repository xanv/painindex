"""This is a fourth pass at gauging pain intensity from google search results.

We explore a more careful choice of relevant features.
As before, we use a regression on search results for each insect.
But now we focus only on the words near "painful" in search results.
The nearby words are likely to modify "painful" and thus give a sense
of *how* painful.

In the first pass,
we essentially reduce each search result to the nearby text and then
perform the same exercise as in agg_regression_analysis.py.

y = pain intensity of the corresponding bug
X = vector of word frequencies in the result, for each of the 
    most common words across dataset, normalized by number of results.


As a motivating example, here are some actual search results for
 "lionfish" "painful":

"How to treat a lionfish sting and administer first aid treatment in the event 
of ... For most people the throbbing, intense pain is going to last for a few 
hours and will ..."

"Although not fatal, the sting of a lionfish is extremely painful. 
Because these fish are not aggressive toward people, contact and poisoning is 
usually accidental."

^We see words like "intense" and "throbbing." This is much more focused than
the surrounding text.


I have also incorporated the ability to make predictions for arbitrary inputs.
This lets me evaluate performance loosely, outside the training set.
This is important because the training set consists of pains within a narrow
domain, namely sting pain from hymenopterans.

The very first pass does NOT do well on external sources of pain.
One reason is that the sting pain results use words like 'sting',
which itself gets a high coeff of 0.89. That's bad, because in the 
training context it is largely a constant term, but outside this context 
it is not usually present. So this alone represents a handicap of -0.89
applied to non-sting pain. 

For instance, we might expect the black widow spider to do relatively well.
Is it not also a bug? Yet it does not sting -- it bites.
The alg is giving it an estimated intensity of 0.88, systematically lower
than ALL predictions in training/test data, presumably because it lacks
many of the common words that incidentally occur around stinging insects.

When I read about the black widow bite, I can certainly tell that it's 
very painful, so in principle this is a fixable problem.
Upon closer inspection, we may also be having issues due to the fact that
the qualitative *type* of pain is different. Latrodectism is characterized
by muscle cramps etc, which don't show up much in the hymenopteran data.


One solution is to simply broaden the scope of the training set, which
is sort of the point of the website. Another solution is to carefully 
remove these types of words from consideration.

A third option is to re-weight the words, so that words common to many
bugs across the scale are given little weight. Unimportant words should not
be allowed to leach magnitude from the constant term.
We are using a bag-of-words approach here, and should
look into tf-idf:
http://en.wikipedia.org/wiki/Tf%E2%80%93idf

Other comments:
* I think it's pretty clear that Ridge is the right choice here. There are 
quite a few words with predictive power, not just a few crucial ones.
I tried lasso just to see, and it really didn't do well.

Things to try:
* Use tf-idf
* See what's going on with the intercept
* Try using higher order features and/or bigrams.

"""


import json
import re
from sklearn import linear_model
import numpy as np
from collections import defaultdict
import random
from pprint import pprint
import nltk


def main():
    with open('../data/outputs/pains_20140928.txt') as json_pains:
        pains = json.load(json_pains)
    with open('../data/outputs/search_results_20140928_painful.txt') as json_search_results:
        search_results = json.load(json_search_results)

    # We also do some plausibility checks on completely unrated pains.
    with open('../data/outputs/search_results_unrated_20140928.txt') as json_search_results_unrated:
        search_results_unrated = json.load(json_search_results_unrated)


    # Sensitivity analysis: see what happens if we change the number of examples.
    # m =100
    # search_results = {pain: search_results[pain][:m] for pain in search_results}

    # Get rid of pain sources with few search results.
    rmin = 20
    search_results = {k:v  for k, v in search_results.items() if len(v) > rmin}

    results_train, results_test = split_data(search_results, 
        split_frac=0.6, seed=46)

    PAIN_RADIUS = 3
    results_train_wds = wordified(results_train, pain_radius=PAIN_RADIUS)
    results_test_wds = wordified(results_test, pain_radius=PAIN_RADIUS)
    results_unrated_wds = wordified(search_results_unrated, pain_radius=PAIN_RADIUS)

    common_wds = get_common_words(results_train_wds, 1000)
    print "FEATURE VECTOR LENGTH:", len(common_wds)

    # Each search result is represented by a vector x with 
    #   x[i] = num times the ith word of common_wds appears in the result.
    results_train_features = make_features(results_train_wds, common_wds)
    results_test_features = make_features(results_test_wds, common_wds)
    results_unrated_features = make_features(results_unrated_wds, common_wds)

    # Create the dataset: We treat each result as a standalone feature.
    X_train, y_train, pains_train = make_data(results_train_features, pains)
    X_test, y_test, pains_test = make_data(results_test_features, pains)
    X_unrated, pains_unrated = make_data(results_unrated_features)

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

        avg_abs_diff = sum(abs(x) for x in diffs)/len(diffs)
        print "Avg abs diff:", avg_abs_diff


    # Run LASSO or RIDGE for a variety of alphas
    # alphas = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    alphas = [0.1]
    # alphas = [0.0001]

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
        pprint(weights[:100])
        pprint(weights[-10:])

        print "Training performance:"
        eval_performance(X_train, y_train, pains_train, model_train)
        print "\nTest performance:"
        eval_performance(X_test, y_test, pains_test, model_train)

        # Check out predictions for the unrated data too:
        y_pred_unrated = model_train.predict(X_unrated)
        predictions = zip(pains_unrated, y_pred_unrated)
        print "\nUnrated predictions:"
        pprint(predictions)



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


def wordified(search_results, pain_radius=None):
    """Return a dict with key=pain, val=list of wordified results for that pain.
        pain_radius restricts us to only words within distance pain_radius 
        of 'pain', with no restriction if pain_radius=None.
    """
    # We strip out each word of each pain name, so the training process
    # cannot rely on essentially knowing the name of the bug.
    pain_wds = set([wd.lower() for pain in search_results 
        for wd in re.findall(r'\b\w+\b', pain)])

    # Split each search result into words.
    # wordified1 =  {pain: 
    #     [ get_words(result['text'], excluded=pain_wds) for result in results ]
    #     for pain, results in search_results.items()}
    wordified = {}
    print "Wordifying pains:"
    for pain, results in search_results.items():
        # print pain
        wordified[pain] = [ 
            get_words(result['text'], excluded=pain_wds, pain_radius=pain_radius) 
            for result in results 
        ]
    print 'Done wordifying.\n'


    # # Combine all search results for a given pain.
    # wordified2 = {pain: [wd for result in results for wd in result] 
    #     for pain, results in wordified1.items()}

    return wordified



def get_words(text, excluded, pain_radius=None):
    """Return a processed list of words in text, minus excluded words.
    We now use only the words within pain_radius of 'pain' words.
    """
    porter = nltk.stem.porter.PorterStemmer()
    
    wds = [wd.lower() for wd in nltk.word_tokenize(text)]
    # This would take only alphanumeric words:
    # wds = [wd.lower() for wd in re.findall(r'\b\w+\b', text)]
    stems = [porter.stem(wd) for wd in wds if wd not in excluded]
    # Let's see what happens if we include bigrams:
    # bigrams = [(stems[i] + ' ' + stems[i+1]) for i in range(len(stems)-1)]
    # stems.extend(bigrams)

    if pain_radius is None:
        return stems

    # All words with 'pain' as a root will be stemmed to 'pain', so we catch
    # any variation here.
    pain_indices = [i for i, wd in enumerate(stems) if wd == 'pain']

    pain_neighbors = [wd for i in pain_indices 
        for wd in stems[i - pain_radius: i + pain_radius + 1]]
    # Keep only one of each word. This deals with overlaps while possibly giving
    # up a smidgeon of power.
    return list(set(pain_neighbors))


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
    """Convert each set of results into a feature vector x.
       x[i] = avg num times the ith word of common_wds appears in the results
       for the painsource corresponding to x.
    """
    return {pain: make_feature(results, common_wds)
        for pain, results in results_train_wds.items()}

def make_feature(results, common_wds):
    """Turn a list of wordified results into a feature vector.
    """
    num_results = len(results)
    counts = defaultdict(int)
    for result in results:
        for wd in result:
            counts[wd] += 1

    return [counts[wd] / float(num_results) for wd in common_wds]

def make_data(results_train_features, pains=None):
    """Create a dataset of X, y.
    Each row of X is the feature for a single pain with corresponding intensity y.
    In addition to X and y, we return a vector painsources which
    records the pain name for each example.
    """

    # This happens if we just have X data, as in the case with unrated
    # data. This is just a quick way of making fresh predictions.
    if pains is None:
        X = results_train_features.values()
        painsources = results_train_features.keys()
        return X, painsources

    X, y, painsources = [], [], []
    for pain, feature in results_train_features.items():
        y.append(pains[pain])
        X.append(feature)
        painsources.append(pain)

    return X, y, painsources



if __name__ == '__main__':
    main()
