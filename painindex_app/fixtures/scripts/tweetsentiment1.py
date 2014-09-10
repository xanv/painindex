"""This is a first pass at gauging pain intensity from tweets.
We use a simplistic approach, motivated in part by tweet sentiment analysis: 
1) Each tweet is associated with a pain source and thus a pain intensity.
2) For each word, average the intensities of every tweet the word appears in;
    call this the word_sentiment.
3) For each tweet, average together the sentiments of each word in the tweet, 
    to get the tweet_sentiment.
4) We now have a metric to compare tweets. Tweets with higher intensities should
    have higher sentiments.
5) For each pain source, get a predicted sentiment by averaging together all the
    tweet_sentiments of tweets about that pain source. This gives us 
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
we start with sentiment ratings for a smal list of individual words, and use
them to derive sentiment ratings for other words and tweets as a whole.
In our case, we start with sentiment ratings for whole tweets and use them
to derive sentiment ratings for individual words and thus tweets not in our
training data.

Advantages and disadvantages:

1) This is a good first approach, quick and easy. It may perform well.

2) A disadvantage is that it won't necessarily generalize to non-tweet writing.
For example, consider an article that discusses the painfulness of a yellowjacket
sting but then goes on to talk about lots of other unrelated stuff.
The predicted intensity will be diluted. We can correct for this by trying
to extract just the discussion around the yellowjacket's sting. In fact,
we can incorporate such samples into the training data.

3) A deeper problem is that it doesn't allow us to express more complex relationships
between the appearance of words and predicted pain levels. If the word "ouch" only
appears in tweets with intensity 4, then its mere presence should predict a 4
regardless of the other words. A properly-designed regression would take this
into account, whereas here we just weight everything equally.

Refinements: 
Use a regression.
Use a natural language processing library to parse tweets and ID n-grams.
And more...
"""


import json
import re
from collections import defaultdict


def main():
    with open('../data/outputs/twitter_results.txt') as json_tweets:
        tweets = json.load(json_tweets)
    with open('../data/outputs/pains.txt') as json_pains:
        pains = json.load(json_pains)

    tweets_words = get_tweets_words(tweets)
    word_sentiments = find_word_sentiments(tweets_words, pains)
    pain_sentiments = find_pain_sentiments(tweets_words, word_sentiments)

    # print word_sentiments
    print pain_sentiments


def get_tweets_words(tweets):
    "Return dict with key=pain name, val=list of list of words in each tweet."
    tweets_words = {pain: [get_words(tweet['text']) for tweet in tweets[pain]]
        for pain in tweets}
    return tweets_words

def get_words(text):
    "Return a list of words in text."
    # TODO: Use nltk to do this properly.
    # Or at least use regex
    # return re.findall(r'\b\w+\b', text)
    return text.lower().split()

def find_word_sentiments(tweets_words, pains):
    "Return dict of sentiment scores for all words in all tweets."

    # Run through all tweets, keeping track of total sentiment 
    # and count for each word eoncountered, so we can compute average.
    word_sentiments = defaultdict(lambda: [0, 0])
    for pain in tweets_words:
        intensity = pains[pain]
        for words in tweets_words[pain]:
            for wd in words:
                word_sentiments[wd][0] += intensity
                word_sentiments[wd][1] += 1
    # Take average:
    word_sentiments = {wd: word_sentiments[wd][0]/word_sentiments[wd][1]
        for wd in word_sentiments}
    return word_sentiments

def find_pain_sentiments(tweets_words, word_sentiments):
    "Return dict with key=pain name, val = average sentiment."
    pain_sentiments = {}

    for pain in tweets_words:
        tweets = tweets_words[pain]

        if len(tweets) == 0:
            print "No tweets for %s..." % pain
            continue
        print "Found %d tweets for %s." % (len(tweets), pain)
    
        tweet_sentiments = [find_tweet_sentiment(tweet, word_sentiments) 
            for tweet in tweets]
        pain_sentiments[pain] = sum(tweet_sentiments) / len(tweet_sentiments)
    
    return pain_sentiments

def find_tweet_sentiment(words, word_sentiments):
    "Compute average sentiment for a tweet represented as a list of words."
    return sum(word_sentiments[wd] for wd in words) / len(words)


# TODO:
# 7)  fit a polynomial regression of y = pain intensity
#     and x = predicted pains:
#     y = b0 + b1 * x + b2 * x^2 + ... + bn * x^n
#     for some choice of degree n.
#     This gives us a conversion from pain_sentiment to predicted intensity.




if __name__ == '__main__':
    main()
