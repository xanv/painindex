"""This is a fourth pass at gauging pain intensity from google search results,
and the first one I would say is truly successful.

Here we explore a more careful choice of relevant features.
As before, we use a regression on search results for each insect.
But now we focus only on the words near "painful" in search results.
The nearby words are likely to modify "painful" and thus give a sharper sense
of *how* painful.

Previous approaches have suffered from "kitchen sink" feature sets,
and I could see that the algorithms were putting high weight on garbage
features. My intuition was being driven by spam classification algs,
where there may be 10,000+ features. But here, this is not a good approach.

The primary issue, I think, is that we don't have enough data for that approach.
I used to think that regularization was the panacea to overfitting,
but think about what's going on here.
Adding tons of features makes it easy to overfit. We can rein this
in by regularizing with ridge or lasso. But what does ridge do?
It squashes everything down towards zero, promoting a spread
of weight across all the words. This prevents us from cherry-picking
the particular features that best fit the training data and giving them
super-high weight, but it does NOT magically help us to find the features
that best generalize outside the training data. With so many features and so few
data points, the algorithm is still bound to identify ungeneralizable words
as most important because they happen to fit the training data best.
The fact that their coeffs get squashed down is at best a dilution, not a
solution, to the problem.
Lasso regression has the same problem. It will pick a few words and call them
most important, but they are unlikely to be the right words.

The solution is to help our algorithm out by reducing noise in the set of words
before we even get started.
This is accomplished by focusing only on the words near "painful".

In addition, we can restrict our feature set to use only the k most common
words across the data. We can experiment with choice of k, and perhaps
do something like ruling out short words. I have tried restricting only to stems
of at least 4 letters, which works fine but more experimentation is needed.

*

PROCEDURE:
We essentially reduce each search result to the nearby text and then
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

*

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

Actually, I don't see what tf-idf would accomplish given that we're running
a regression. Well, it's a way of normalizing features, which wouldn't matter
for straight OLS, but given that we're regularizing with ridge regression,
it would matter. But why not just set normalize=True for Ridge?

I tried that and...well, it completely changed everything.
The highest weighted words are no longer "sting" etc, but...they don't exactly
make sense either. passag, part, stock, account, steve, gun. Right...
Think about if there is any advantage to reweighting words. There really
ought to be. Each word should be on equal footing for regularization purposes.

Next, I tried simply removing highly rated words like 'sting' and 'insect'
which are obviously specific to insect sting pain. This helped a bit,
but it's more of a demo than an actual solution.

In straight OLS, the coeff on sting should reflect its residual predictive power
over and above the other terms, including the constant. So if this got a positive
beta in OLS, it would indicate that more mention of "sting" actually
coincides with higher pain insects. But if possible, we should deliberately
deprive ourselves of this language for times when we are trying to apply
our model to non-stinging pains. It is still useful within this domain, but
we should project it out, more generally. So...think about the right projection
that would do that.

[Does sklearn's ridge regularization include the constant term as another feature?
If so, then sting will pick up some of the constant term,
which is an objectively terrible idea. The last thing we want is for a bunch
of words which are common across the domain of sting pain to displace the
intercept coeff! Then when we move to a different problem domain, we effectively
lose the intercept, leveling everything down.
If ridge does it wrong, that's bad. The solution is NOT to add my own vector
of 1's, because that will DEFINITELY be regularized.
Ugh. The internet is being unhelpful on this front.
It looks like the intercept IS included in regularization.
http://scikit-learn.org/stable/modules/linear_model.html

Options:
1. Tweak the code (no).
2. Write my own objective fn and use a scipy.optimize solver. (attractive)
3. Temporary kludge: Manually include a constant column but make it very large,
   e.g. 1e9. That way, the optimal coef will be very close to zero and can be chosen
   essentially without regularization penalty.
   In the limit, the coef is essentially unregularized.
   Actually, this is a good solution for the moment. In fact, I could even
   build a wrapper in which you specify the columns to not regularize, and
   it inflates them by a large factor.

UPDATE: I tried 3 with mixed results. The intercept WAS getting squashed down,
and even putting in a constant column of 10's instead of 1's raised the
size of beta0 * intercept from 1.04 to 2.02. That's what it should be.
Raising it to 10**10 had little additional effect (2.04).

However...it did not seem to increase my out of sample ratings systematically.
It's entirely possible that the increased weight of the constant has been offset
by increased negative offsets among other common words, and when I go to the
new domain, I'm losing sting-related common words but retaining things like
"the" which would have offset them before, and which now simply drive results
downward.

Next, try removing stopwords.

Update: Yeah. I removed stop words and "schmidt" "index". I lost accuracy
on my test data, but I gained accuracy on my true out-of-sample, non-sting data.

Next steps:
Examining the words with large pos/neg weight is highly instructive.
For example, I notice that "bite" carries a negative coeff of -.8. That's a
big deal, and may be coming from the fact that among stinging insects,
less-painful stings may be readily confused with "bites". But out of this group,
a spider bite may be very painful.
This kind of mistake, aside from being corrected manually, must be dealt with
by expanding the training set.

It's an interesting thing to remark on though.

Other things to try:
I really should try 2-grams, 3-grams.
And keep inspecting the output and ruling out words that don't belong.
]


Anyway, I need a more general way to rule out words like sting and insect.
The obvious approach is to actually expand
the training set. In the absence of this, though, I could try an exercise
where I restrict myself to only words used across both problem domains,
and then use the rated data to train.
This is using lionfish as training data in a first pass, but only schmidt
data to actually train.

I expect some version of this will lead to an improvement; however, there
is clearly no substitute for actual training. Different pain domains can
use the same set of words in different ways to indicate pain.

Actually, ultimately I may want to make good use of domain-specific language
inside any domain, so long as I can identify the domain.
I'm starting to see where this leads. There is no single optimal translation
from written language into pain, but written language also contains other
clues about which translation we should use. I can identify differences
between writing about insects and fish, which means so can a learning algorithm.
We will ultimately want to perform
a clustering exercise to sort pain sources into different groups based
on the language used to describe them. We will then use the data within
the group to estimate pain.

Oh, now this is interesting. Not to get ahead of myself, but this latter problem
is starting to take on aspects of recommender systems. The clustering approach
is simplistic and absolute: a pain gets sorted into a particular bucket, and is handled
like all the other pains in that bucket. But we could take a more nuanced
view of the world. Each pain is similar to others to various degrees. There is
an underlying vector of characteristics for each pain. One (set of)
characteristics
is the intensity etc, another set of characteristics is the words used to describe
this class of pain sources, and a third set of characteristics is the words
used in this class of pain sources to actually describe the pain intensity.
We can use the second set to identify a class of pain, and the third set
to make the translation. But more generally, the second set may tell us not
just one class but rather how similar to each class. We can build up a
"personalized" translation table by taking a weighted combination of these
classes.

That's very vague and possibly rubbish, but something to think about down the
line.

"""


import json
import re
from sklearn import linear_model
import numpy as np
from collections import defaultdict
import random
from pprint import pprint
import nltk
from nltk.corpus import stopwords


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
        split_frac=0.6, seed=48)

    MAX_FEATURES = 1000
    MAX_PHRASE_LEN = 3

    PAIN_RADIUS = 3
    results_train_wds = wordified(results_train, PAIN_RADIUS, MAX_PHRASE_LEN)
    results_test_wds = wordified(results_test, PAIN_RADIUS, MAX_PHRASE_LEN)
    results_unrated_wds = wordified(search_results_unrated, PAIN_RADIUS, MAX_PHRASE_LEN)

    common_wds = get_common_words(results_train_wds, MAX_FEATURES)
    NUM_FEATURES = len(common_wds) # min of len(results_train_wds) and MAX_FEATURES
    print "FEATURE VECTOR LENGTH:", NUM_FEATURES

    # Each search term corresponds to a single aggregated training
    # example, consisting of a vector of the
    # avg counts of each word in common_wds appearing in the search results
    # for that term.
    results_train_features = make_features(results_train_wds, common_wds)
    results_test_features = make_features(results_test_wds, common_wds)
    results_unrated_features = make_features(results_unrated_wds, common_wds)

    # Setting INTERCEPT_SIZE to a large value will effectively reduce the
    # extent to which the constant is regularized. In the limit as
    # INTERCEPT_SIZE approaches infinity, we effectively remove the
    # intercept from the regularization.
    # For interpretation, the intercept coeff should then be multiplied by
    # INTERCEPT_SIZE.
    # (This applies specifically for Ridge regression, with its squared penalty)
    INTERCEPT_SIZE = 100 # 10**10
    X_train, y_train, pains_train = make_data(results_train_features, pains,
        intercept_size=INTERCEPT_SIZE)
    X_test, y_test, pains_test = make_data(results_test_features, pains,
        intercept_size=INTERCEPT_SIZE)
    X_unrated, pains_unrated = make_data(results_unrated_features,
        intercept_size=INTERCEPT_SIZE)

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

    for alpha in alphas:
        print "\nalpha = %s:" % alpha

        # SELECT LASSO OR RIDGE
        # reg = linear_model.Ridge(alpha=alpha, normalize=True)
        reg = linear_model.Ridge(alpha=alpha, fit_intercept=False)
        # reg = linear_model.Lasso(alpha=alpha)

        model_train = reg.fit(X_train, y_train)
        y_pred_train = model_train.predict(X_train)
        y_pred_test = model_train.predict(X_test)

        # See the weights on the feature vector words:
        assert len(model_train.coef_) == len(common_wds) + 1
        weights = zip(["<INTERCEPT>"] + common_wds, model_train.coef_)
        weights = sorted(weights, key=lambda x: x[1], reverse=True)

        # The constant has now been added manually so there is no .intercept
        # weights = [('<CONSTANT_TERM>', model_train.intercept_)] + weights
        print "WEIGHTS:"
        intercept_val = dict(weights)['<INTERCEPT>']
        print "Intercept:", intercept_val
        # See the intercept "as if" it were a col of 1's.
        print "Scaled intercept:", intercept_val * INTERCEPT_SIZE
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


def wordified(search_results, pain_radius=None, max_phrase_len=1):
    """Return a dict with key=pain, val=list of wordified results for that pain.
        pain_radius restricts us to only words within distance pain_radius
        of 'pain', with no restriction if pain_radius=None.
        Each "wordified" result actually contains n-grams of length
        n = 1 to max_phrase_len.
    """
    # We strip out each word of each pain name, so the training process
    # cannot rely on essentially knowing the name of the bug.
    pain_wds = set([wd.lower() for pain in search_results
        for wd in re.findall(r'\b\w+\b', pain)])

    # In addition, we have manually inspected the outputted word coeffs,
    # and are removing highly ranked words that are evidently specific
    # to insect sting pain:
    pain_wds = pain_wds.union(set(['sting', 'insect', 'schmidt', 'index']))
    stopwds = set(stopwords.words('english'))

    excluded = pain_wds | stopwds


    # Split each search result into words.
    # wordified1 =  {pain:
    #     [ get_words(result['text'], excluded=pain_wds) for result in results ]
    #     for pain, results in search_results.items()}
    wordified = {}
    print "Wordifying pains:"
    for pain, results in search_results.items():
        # print pain

        # lists of 1-grams for each result:
        words =  [
            get_words(result['text'], excluded=excluded, pain_radius=pain_radius)
            for result in results
        ]

        # 2-grams and so forth are formed by concatenating adjacent 1-grams.
        all_phrases = []
        for result in words:
            phrases = list(result)
            for n in range(2, max_phrase_len + 1):
                for i in range(0, len(result) - n):
                    phrases.append(' '.join(result[i:i+n]))
            all_phrases.append(phrases)

        wordified[pain] = all_phrases

    print 'Done wordifying.\n'
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
    # excluded is also allowed to contain stems; get rid of any such stems
    stems = [s for s in stems if s not in excluded]
    # Experiment: get rid of short stems
    # stems = [s for s in stems if len(s) > 3]

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

def make_data(results_train_features, pains=None, intercept_size=1):
    """Create a dataset of X, y.
    Each row of X is the feature for a single pain with corresponding intensity y.
    In addition to X and y, we return a vector painsources which
    records the pain name for each example.
    """

    # This happens if we just have X data, as in the case with unrated
    # data. This is just a quick way of making fresh predictions.
    if pains is None:
        X = results_train_features.values()
        X = add_intercept(X, intercept_size)
        painsources = results_train_features.keys()
        return X, painsources

    X, y, painsources = [], [], []
    for pain, feature in results_train_features.items():
        y.append(pains[pain])
        X.append(feature)
        painsources.append(pain)

    X = add_intercept(X, intercept_size)

    return X, y, painsources

def add_intercept(X, intercept_size=1):
    """Return a copy of X with intercept column added, scaled by intercept_size.
    """
    X2 = np.array(X)
    intercept = np.ones((X2.shape[0], 1)) * intercept_size
    return np.concatenate((intercept, X2), 1)





if __name__ == '__main__':
    main()
