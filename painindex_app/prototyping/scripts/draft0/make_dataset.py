"""Convert article texts into feature vectors.
    Do this for each article produced by get_articles.py.
"""

import json
import re
import operator


def make_dataset(features, pain_index):
    """Construct the X and y data matrices.
        
        features is a dict with key = pain name,
        value = all feature vectors for that pain source.
        
        pain_index is a dict with key = pain name, value = intensity

        In addition to X and y, we return a vector painsources which
        records the pain name for each example.
    """
    X, y, painsources = [], [], []

    for pain in features:
        for f in features[pain]:
            X.append(f)
            y.append(pain_index[pain])
            painsources.append(pain)
    return X, y, painsources


def make_features(search_results, num_features):
    """Return a dict with (k,v) = (pain, list of article feature vectors).
        search_results is the output from get_articles.py.
        num_features is the length of a feature vector.

        A feature vector is built from a list of word frequencies for 
        the corresponding article.
    """
    # Each article is first broken down into a wordlist of the words
    # that appear in it.
    # article_wordlists[pain] = list of wordlists, one for each article
    # about that source of pain.
    article_wordlists = make_wordlists(search_results)
    
    # Next we count word frequencies over all the articles.
    wordcounts = find_wordcounts(article_wordlists)
    toplist = most_frequent_words(wordcounts, num_features)
    # print toplist

    features = {}
    for pain in article_wordlists:
        wordlists = article_wordlists[pain]
        features[pain] = [make_feature(w, toplist) for w in wordlists]
    return features, toplist




def make_wordlists(search_results):
    "Return a dict with key=pain, val = list of wordlists for each text."
    article_wordlists = {}

    for pain in search_results:
        texts = get_texts(pain, search_results)
        wordlists = []

        for text in texts:
            # Replace this text with a list of its words
            wordlists.append(get_words(text))

        article_wordlists[pain] = wordlists

    return article_wordlists

# Maybe we should refactor search_results to store as value 
# [google search url, [results_dict list]] where the keys
# of each results_dict are url, urltext, pagetext.
# Then I could look up a text with search_results[pain][1][resultnum][pagetext] 
# Still convoluted...alternatively we can build this stuff into the db,
# at some point.
def get_texts(pain, search_results):
    """Return list of all texts associated with pain in search_results dict.
        search_results is of the form:
        {'yellowjacket': [google search url, [[link text, url, page text] for each search result], ...}
    """
    texts = [ result[2] for result in search_results[pain][1] ]
    return texts

def get_words(text):
    "Return a list of words in an article text."
    return re.findall(r"\b[a-z]+\b", text.lower())

def find_wordcounts(article_wordlists):
    "Get word counts over all the data."
    wordcounts = {}

    for pain in article_wordlists:
        for wordlist in article_wordlists[pain]:
            for word in wordlist:
                wordcounts[word] = wordcounts.get(word, 0) + 1

    return wordcounts

def most_frequent_words(wordcounts, num):
    """Return a list of the top num most frequent words.
        The list is ordered by decreasing frequency.
    """
    items = wordcounts.items()
    items.sort(key=operator.itemgetter(1), reverse=True)
    return [items[i][0] for i in range(num)]

# We might want to seriously consider normalizing at this step.
# We could simply do binary feature vectors, or we could divide
# by number of words in the article.
def make_feature(wordlist, toplist):
    """Return a feature vector corresponding to a wordlist.
        The wordlist represents an article.
        We count the incidences in wordlist of each word in toplist.
    """
    # This could be optimized if necessary.
    feature = [wordlist.count(word) for word in toplist]
    binary_feature = [int(f>0) for f in feature] 
    # I return the binary version for now, rather than worry about
    # the proper normalization.
    return binary_feature


# The functions below could be part of their own file.
# Or I should rename this file to make_dataset.py

# This is partly redundant with get_search_results.py's get_pains.
def get_pain_index(filename):
    """Return a list of pain names and intensities from a txt file.
        The txt file must be comma-delimited with the pain name as 
        the first element of each line, followed by intensity
    """
    pains = open(filename, 'r').read().strip().split('\n')
    pains = [line.split(',') for line in pains]
    return {k:v for [k,v] in pains} 




if __name__ == "__main__":
    with open('../data/outputs/search_results_and_texts_2014_09_05_1527.txt', 'r') as json_data:
        search_results = json.load(json_data)

    NUM_FEATURES = 200
    features, toplist = make_features(search_results, NUM_FEATURES)
    # print features['fire ant'][:2]

    pain_index = get_pain_index('../data/inputs/schmidt_ratings1.txt')

    dataset = make_dataset(features, pain_index)

    with open('../data/outputs/paindata.txt', 'w') as outfile:
        json.dump(dataset, outfile)

    # Save the toplist, so we can see what words correspond to
    # each element of a feature vector.
    with open('../data/outputs/toplist%d.txt' % NUM_FEATURES, 'w') as outfile2:
        json.dump(toplist, outfile2)


