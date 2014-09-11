"""This is a first pass at gauging pain intensity from google search results.
We use a simplistic approach that analyzes the text blurb associated with each
search result.

1) Each result is associated with a pain source and thus a pain intensity.
2) For each word, average the intensities of every result the word appears in;
    call this the word_sentiment.
3) For each result, average together the sentiments of each word in the result, 
    to get the result_sentiment.
4) We now have a metric to compare results. results with higher intensities should
    have higher sentiments.
5) For each pain source, get a predicted sentiment by averaging together all the
    result_sentiments of results about that pain source. This gives us 
    pain_sentiment for each pain source.
6) Pain sources with higher intensity should have higher sentiments.
    However, these will not correspond directly with
    the units of the original pain index scale, because we have diluted
    intensities by averaging strongly predictive words with common words.
7) To correct for this, we now fit a polynomial regression of y = pain intensity
    and x = predicted pains:
    y = b0 + b1 * x + b2 * x^2 + ... + bn * x^n
    for some choice of degree n.
    This gives us a conversion from pain_sentiment to predicted intensity.

Note that this approach is similar to the tweet sentiment approach where
we start with sentiment ratings for a small list of individual words, and use
them to derive sentiment ratings for other words and tweets as a whole.
In our case, we start with sentiment ratings for whole results and use them
to derive sentiment ratings for individual words and thus results not in our
training data.

Advantages and disadvantages:

1) This is a good first approach, quick and easy. It may perform well.
And we have the potential to collect a huge amount of data, which may
make it work very well.

2) The approach is sensitive to what sort of text the search query picks out. To
extend this to predict pain for new subjects, we could easily do google searches
and get the analogous result texts. But in the current form, requiring the
"stung" keyword makes this problematic. The method is effective for stings
because the result text is highly relevant and somewhat uniform across subjects.
For another subject, we could manually try to come up with a different word that
had the same effect, but this is clearly a limitation and there's no guarantee
it would give us comparable data. In other words, we want blurbs that capture
people complaining about their pain.

Perhaps "painful" would be a better keyword. We should try that next.

3) We are throwing away potentially a lot of information that the greater body
of the search result contains. At present, this seems a worthwhile price to pay
for the high relevance of the result text.

4) A deeper problem is that our simple approach doesn't allow us to express more
complex relationships between the appearance of words and predicted pain levels.
If the word "ouch" only appears in tweets with intensity 4, then its mere
presence should predict a 4 regardless of the other words. A regression  could
take this into account, whereas here we just weight everything equally.

Looking at the results, it appears we are seriously diluting the impact
of important words by giving them equal weight despite their predictive power.
However, it may still perform well when we throw lots of data at it.

Refinements:  Use a regression. (Think of each result as a feature vector of the
words it does and does not contain) Use a natural language processing library to
parse results and identify n-grams; we are currently only using individual
words, which is a limitation. And much much more... 
"""


import json
import re
from sklearn import linear_model
import numpy as np
from collections import defaultdict
import random


def main():
    with open('../data/outputs/pains_20140910.txt') as json_pains:
        pains = json.load(json_pains)

    with open('../data/outputs/search_results_20140910.txt') as json_search_results:
        search_results = json.load(json_search_results)

    results_train, results_test = split_data(search_results, 
        split_frac=0.6, seed=4747)

    results_train_wds = wordified(results_train)

    # Get sentiments for each word and pain in results_train
    word_sentiments = find_word_sentiments(results_train_wds, pains)
    pain_sentiments = find_pain_sentiments(results_train_wds, word_sentiments)

    # Using word_sentiments and pain_sentiments computed over the training set,
    # we create a function that predicts pain for any input.
    predict_pain = make_pain_predictor(pains, word_sentiments, pain_sentiments,
        degree=2)

    predicted_pain_train = predict_pain(results_train)
    predicted_pain_test = predict_pain(results_test)



    # DIAGNOSTICS:
    # Evaluate performance on training vs test data here.

    # Currently the results are okay but there are problems.
    # Depending on the seed, we get 0.1-0.2 avg abs diff for the training set
    # and 0.5-0.7 abs diff for the test set.
    # I could live with being off by half a unit on a 4-point scale, but
    # consider that most bugs are between 2 and 4, so if we just guessed 3 
    # (or 2.6) every time, we would do pretty well.
    # 
    # seed = 47 gets us 0.59 but does poorly on inspection: Warrior Wasp and 
    # Bullet Ant get 2.5's! (And yes, Tarantula Hawk is in the training set).
    # All of the predicted ratings are between 2 and 3.


    print predicted_pain_train
    print predicted_pain_test

    diff_train = {pain: predicted_pain_train[pain] - pains[pain] 
        for pain in predicted_pain_train}
    diff_test = {pain: predicted_pain_test[pain] - pains[pain] 
        for pain in predicted_pain_test}

    avg_diff_train = sum(abs(t) for t in diff_train.values()) / len(diff_train)
    avg_diff_test = sum(abs(t) for t in diff_test.values()) / len(diff_test)

    print "\n", diff_train
    print "\n", diff_test
    print "\n", avg_diff_train, avg_diff_test
    


def wordified(search_results):
    """Return a copy of search_results with each result turned into a wordlist.

        In other words, return a dict with key=pain name, val=list of results 
        where each result is now a list of the words the result contains.
    """
    # We strip out the name of the pain from each text, so the training process
    # cannot rely on essentially knowing the name of the bug.
    # (This is not a huge deal for the current approach that takes the equal-weighted
    # average over each word)
    return {pain: [get_words(result['text'].replace(pain, '')) for result in search_results[pain]]
        for pain in search_results}

def get_words(text):
    "Return a processed list of words in text."
    # TODO: Use nltk to do this properly.
    # Or at least regex
    # e.g. re.findall(r'\b\w+\b', text)
    return text.lower().split()

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

def find_word_sentiments(results_wordified, pains):
    "Return dict of sentiment scores for all words in all results."

    # Run through all results, keeping track of total sentiment 
    # and count for each word encountered, so we can compute average.
    word_sentiments = defaultdict(lambda: [0, 0])
    for pain in results_wordified:
        intensity = pains[pain]
        for words in results_wordified[pain]:
            for wd in words:
                word_sentiments[wd][0] += intensity
                word_sentiments[wd][1] += 1
    # Take average:
    word_sentiments = {wd: word_sentiments[wd][0]/word_sentiments[wd][1]
        for wd in word_sentiments}
    return word_sentiments

def find_pain_sentiments(results_wordified, word_sentiments):
    "Return dict with key=pain name, val = average sentiment."
    pain_sentiments = {}

    for pain in results_wordified:
        results = results_wordified[pain]

        # if len(results) == 0:
        #     print "No results for %s..." % pain
        #     continue
        # print "Found %d results for %s." % (len(results), pain)
    
        result_sentiments = [find_result_sentiment(result, word_sentiments) 
            for result in results]
        result_sentiments = [s for s in result_sentiments if s is not None]

        try:
            pain_sentiments[pain] = sum(result_sentiments) / len(result_sentiments)
        except ZeroDivisionError:
            pain_sentiments[pain] = None
    
    return pain_sentiments

def find_result_sentiment(words, word_sentiments):
    """Compute average sentiment for a result represented as a list of words.
        Words not in word_sentiments are ignored.
        If we don't have sentiment for any wd in words, return None.
    """
    word_ratings = [word_sentiments[wd] for wd in words 
        if wd in word_sentiments]
    
    if len(word_ratings) == 0:
        return None

    return sum(word_ratings) / len(word_ratings)

def make_pain_predictor(pains, word_sentiments, pain_sentiments, degree):
    """Return a function that makes pain predictions for any set of results.
    Results are in their original dictionary form (with 'text' field).
    The function returns a dict of pain name: predicted pain level.

    This function is trained using word_sentiments and pain_sentiments.
    degree is the degree of the polynomial to fit with sentiment_converter.
    """
    # f will convert pain_sentiment into predicted pain level.
    # This is necessary because sentiment itself does not equal predicted pain.
    # sentiment_converter runs a regression to determine coefficients
    # for the transformation, so it's important that the conversion is only
    # run once and then used in every call to pain_predictor below.
    f = sentiment_converter(pains, pain_sentiments, degree=degree)

    def pain_predictor(results):
        sentiments = find_pain_sentiments(wordified(results), word_sentiments)

        # If want to include keys with None predictions:
        # predicted_pain = {}
        # for pain in sentiments:
        #     if sentiments[pain] is None:
        #         predicted_pain[pain] = None
        #     else:
        #         predicted_pain[pain] = f(sentiments[pain])
        
        # If want to omit keys with None predictions:
        predicted_pain = {pain: f(sentiments[pain]) for pain in sentiments
            if sentiments[pain] is not None}
        return predicted_pain

    return pain_predictor


def sentiment_converter(pains, pain_sentiments, degree):
    """Return a function which converts pain sentiment into predicted pain.
    The sentiment scores do not directly correspond to the original
    pain scale. We regress true pain on sentiments, with higher order terms
    up to the degree'th degree of sentiment,
    to produce a function that converts sentiment to predicted true pain. 
    """

    def vectorized(sent):
        "Turn a sentiment into a feature vector of higher order terms."
        return np.array([sent**p for p in range(0, degree+1)])

    X, y = [], []

    for pain, sent in pain_sentiments.items():
        # Each example has terms 1, x, x**2, ..., x**degree
        x = vectorized(sent)
        X.append(x)
        y.append(pains[pain])

    # I have included an intercept manually, no need to add one.
    LM = linear_model.LinearRegression(fit_intercept=False)
    LMfit = LM.fit(X, y)

    # Return a function that gives the fitted value for a sentiment s.
    return lambda s: vectorized(s).dot(LMfit.coef_)


if __name__ == '__main__':
    main()
