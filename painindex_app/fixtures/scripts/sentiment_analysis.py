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

Yes, looking at the results, it appears we are seriously diluting the impact
of important words by giving them equal weight despite their predictive power.

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


def main():
    with open('../data/outputs/pains_20140910.txt') as json_pains:
        pains = json.load(json_pains)

    with open('../data/outputs/search_results_20140910.txt') as json_search_results:
        search_results = json.load(json_search_results)

    results_wordified = wordified(search_results)
    word_sentiments = find_word_sentiments(results_wordified, pains)
    pain_sentiments = find_pain_sentiments(results_wordified, word_sentiments)

    # print word_sentiments
    print pain_sentiments

    f = get_sentiment_converter(pains, pain_sentiments, degree=1)



def wordified(search_results):
    """Return a copy of search_results with each result turned into a wordlist.

        In other words, return a dict with key=pain name, val=list of results 
        where each result is now a list of the words the result contains.
    """
    return {pain: [get_words(result['text']) for result in search_results[pain]]
        for pain in search_results}

def get_words(text):
    "Return a list of words in text."
    # TODO: Use nltk to do this properly.
    # Or at least regex
    # e.g. re.findall(r'\b\w+\b', text)
    return text.lower().split()

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

        if len(results) == 0:
            print "No results for %s..." % pain
            continue
        print "Found %d results for %s." % (len(results), pain)
    
        result_sentiments = [find_result_sentiment(result, word_sentiments) 
            for result in results]
        pain_sentiments[pain] = sum(result_sentiments) / len(result_sentiments)
    
    return pain_sentiments

def find_result_sentiment(words, word_sentiments):
    "Compute average sentiment for a result represented as a list of words."
    return sum(word_sentiments[wd] for wd in words) / len(words)


def get_sentiment_converter(pains, pain_sentiments, degree):
    """Return a function which converts pain sentiment into predicted pain.
    The sentiment scores are not properly scaled to correspond to the original
    pain scale. We regress true pain on sentiments (with higher order terms)
    to produce a function that converts sentiment to predicted true pain. 
    """

    X, y = [], []

    for pain, sent in pain_sentiments.items():
        # Each example has terms 1, x, x**2, ..., x**degree
        x = [sent**i for i in range(0, degree+1)]
        X.append(x)
        y.append(pains[pain])

    # I have included an intercept manually, so I don't need
    # it to implicitly add a vector of ones.
    LM = linear_model.LinearRegression(fit_intercept=False)

    LMfit = LM.fit(X, y)

    # fittedvals = LMfit.predict(X)
    # for i, yhat in enumerate(fittedvals):
    #     print yhat, y[i]

    # print LMfit.coef_

    # Return a function that gives the fitted value for a sentiment x.
    return lambda x: LMfit.predict([x])[0]

if __name__ == '__main__':
    main()
