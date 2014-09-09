import json
from sklearn import linear_model
import numpy as np
import operator as op
from make_dataset import get_texts

with open('../data/outputs/paindata_2014_09_05_1527_morefeatures.txt', 'r') as json_data:
    X, y, pains = json.load(json_data)

X = [[int(i) for i in row] for row in X]
y = [float(i) for i in y]

# We split the data into training, cross-validation, and test.
m = len(y)
# print m
# print pains[:60]
m_train = int(0.6 * m)
m_cv = int(0.2 * m)
m_test = m - m_train - m_cv

X_train, y_train, pains_train = X[:m_train], y[:m_train], pains[:m_train]
X_cv, y_cv, pains_cv = X[m_train:m_train+m_cv], y[m_train:m_train+m_cv], pains[m_train:m_train+m_cv]
X_test, y_test, pains_test = X[m_train+m_cv:], y[m_train+m_cv:], pains[m_train+m_cv:]


# Ridge regression (regularized OLS)
# See: http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
clf = linear_model.Ridge(alpha=1)
model_train = clf.fit(X_train, y_train)
# print model_train.coef_

# print len(model_train.coef_)
# print len(X_train[0])

fitted_train = model_train.predict(X_train)
# print fitted_train[:40]

print model_train.score(X_train, y_train)
print model_train.score(X_cv, y_cv)
print model_train.score(X_test, y_test)


## TRAINING DATA ##

diff = []
for pain in set(pains_train):
    print "Pain source:", pain

    # Extract the indices corresponding to this pain
    indices = [i for (i, x) in zip(range(m_train), pains_train)
        if x == pain]
    # Examples of the same pain are contiguous.
    a, b = indices[0], indices[-1]+1
    pain_predicted = model_train.predict(X_train[a:b])
    pain_true = y_train[a:b]

    # We want the avg prediction for each pain source
    pain_predicted_avg = pain_predicted.mean()
    pain_true = pain_true[0]
    avg_diff = (pain_predicted_avg - pain_true)

    print "True, predicted, avg diff:", (pain_true, pain_predicted_avg, avg_diff)
    diff.append(avg_diff)

print diff
avg_sq_diff = sum(x**2 for x in diff)/len(diff)
print "Avg squared diff (train data):", avg_sq_diff

print("")

## Cross-validate ##

diff = []
for pain in set(pains_cv):
    print "Pain source:", pain

    # Extract the indices corresponding to this pain
    indices = [i for (i, x) in zip(range(m_cv), pains_cv)
        if x == pain]
    # Examples of the same pain are contiguous.
    a, b = indices[0], indices[-1]+1
    pain_predicted = model_train.predict(X_cv[a:b])
    pain_true = y_cv[a:b]

    # We want the avg prediction for each pain source
    pain_predicted_avg = pain_predicted.mean()
    pain_true = pain_true[0]
    avg_diff = (pain_predicted_avg - pain_true)

    print "True, predicted, avg diff:", (pain_true, pain_predicted_avg, avg_diff)
    diff.append(avg_diff)

print diff
avg_sq_diff = sum(x**2 for x in diff)/len(diff)
print "Avg squared diff (CV data):", avg_sq_diff  

## Test ##
print ""

diff = []
for pain in set(pains_test):
    print "Pain source:", pain

    # Extract the indices corresponding to this pain
    indices = [i for (i, x) in zip(range(m_test), pains_test)
        if x == pain]
    # Examples of the same pain are contiguous.
    a, b = indices[0], indices[-1]+1
    pain_predicted = model_train.predict(X_test[a:b])
    pain_true = y_test[a:b]

    # We want the avg prediction for each pain source
    pain_predicted_avg = pain_predicted.mean()
    pain_true = pain_true[0]
    avg_diff = (pain_predicted_avg - pain_true)

    print "True, predicted, avg diff:", (pain_true, pain_predicted_avg, avg_diff)
    diff.append(avg_diff)

print diff
avg_sq_diff = sum(x**2 for x in diff)/len(diff)
print "Avg squared diff (test data):", avg_sq_diff


print '\n'

# Diagnostics: 
with open('../data/outputs/search_results_and_texts_2014_09_05_1527.txt', 'r') as json_data:
    search_results = json.load(json_data)

# what are words worth?
with open('../data/outputs/toplist200.txt', 'r') as json_toplist:
    toplist = json.load(json_toplist)

assert len(toplist) == len(model_train.coef_)
wordvals = zip(toplist, model_train.coef_)
# print wordvals
sorted_wordvals = sorted(wordvals, key=op.itemgetter(1), reverse=True)
print sorted_wordvals[:10]


# Error analysis: look at articles that are getting really misjudged.
# See what is causing the problems.
for pain in list(set(pains_train))[3:]:

    # Extract the indices corresponding to this pain
    indices = [i for (i, x) in zip(range(m_train), pains_train)
        if x == pain]
    # Examples of the same pain are contiguous.
    a, b = indices[0], indices[-1]+1
    pain_predicted = model_train.predict(X_train[a:b])
    pain_true = y_train[a:b]

    # Get the indices of the articles with worst over- and underestimates.
    overestimates = pain_predicted - pain_true
    worst_over_idx = overestimates.argmax()
    worst_under_idx = overestimates.argmin()

    texts = get_texts(pain, search_results)

    # Text of article with worst overestimate:
    worst_over_article = texts[worst_over_idx]
    worst_under_article = texts[worst_under_idx]

    print "\nPain:", pain
    print "True intensity:", pain_true[0]
    print "\nArticle with worst overestimate of pain intensity:"
    print "(Diff: %f)" % overestimates[worst_over_idx]
    print worst_over_article 
    # print "\nArticle with worst underestimate of pain intensity:"
    # print "(Diff: %f)" % overestimates[worst_under_idx]
    # print worst_under_article

    break