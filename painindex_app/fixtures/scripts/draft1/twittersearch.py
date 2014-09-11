"""THIS FILE IS DEPRECATED.
We moved on to scraping google results.
The volume just wasn't good for relevant tweets, even
when scraped.

Collect data from Twitter in two ways:
1) Use the Twitter search API to query for sting related tweets.
2) Scrape Twitter to manually grab the results of such queries.

The API is convenient but only returns a small subset of search results.
It is intended for ongoing realtime searching.
Our purposes are different: We want a one-time snapshot of as many results per
pain source as possible. Scraping is the solution.
"""

import json
import socket
import oauth2 as oauth
import urllib, urllib2, cookielib
from bs4 import BeautifulSoup
from time import sleep
from random import uniform
import os


# Twitter API setup
# To use this, you will have to set up an app on Twitter, generate credentials,
# and store them as environment variables on your local machine.
api_key = os.environ['TWITTER_PAININDEX_API_KEY']
api_secret = os.environ['TWITTER_PAININDEX_API_SECRET']
access_token_key = os.environ['TWITTER_PAININDEX_ACCESS_TOKEN_KEY']
access_token_secret = os.environ['TWITTER_PAININDEX_ACCESS_TOKEN_SECRET']
_debug = 0
oauth_token    = oauth.Token(key=access_token_key, secret=access_token_secret)
oauth_consumer = oauth.Consumer(key=api_key, secret=api_secret)
signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()
http_method = "GET"
http_handler  = urllib2.HTTPHandler(debuglevel=_debug)
https_handler = urllib2.HTTPSHandler(debuglevel=_debug)



def main():
    pains = import_pains('../data/inputs/schmidt_ratings_complete.csv')
    names = pains.keys()

    # I have given up on the Twitter API for this exercise, but the code is
    # fully functional.

    # Use Twitter API
    # tweets = get_tweets_api(names)

    # Scrape Twitter
    tweets_scraped = get_tweets_scraped(names)

    # Save the tweets and pains files for later use.
    with open('../data/outputs/pains.txt', 'w') as outfile1:
        json.dump(pains, outfile1)
    # with open('../data/outputs/twitter_results.txt', 'w') as outfile2:
    #     json.dump(tweets, outfile2)
    with open('../data/outputs/twitter_results_scraped.txt', 'w') as outfile3:
        json.dump(tweets_scraped, outfile3)


def import_pains(filename):
    """Return a dict of name:intensity pairs from a csv file.
        The file must be in the form of schmidt_ratings_complete.csv.
    """
    data_str = open(filename, 'r').read().strip()
    data = [line.split(',') for line in data_str.split('\n')]

    colnames = data[0]
    name_idx, intensity_idx = colnames.index('name'), colnames.index('pain')

    # Grab only common names and pain intensities, for only bugs with names.
    # There are also some redundant names, like several species of wasp all
    # colloquially known as Paper Wasps. Get rid of redundant names
    # by simply keeping the first instance of each.
    pains = {}
    names = set()
    for row in data[1:]:
        name, intensity = row[name_idx], row[intensity_idx]
        if name != '' and name not in names:
            pains[name] = float(intensity)
            names.add(name)

    print ("Pain index has been imported; %d of %d pains remain after processing."
        % (len(pains), len(data)-1))
    return pains



def get_tweets_api(pain_names):
    """Get Twitter Search API results corresponding to the pain names.
        Return a dict with key=pain_name, val=list of tweet objects.
    """
    tweets = {}
    print "Grabbing tweets..."

    # When debugging, don't run this on all pain_names.
    # You can easily hit Twitter's rate limit within a 15-minute window.
    # for i, name in enumerate(pain_names[:2]):
    for i, name in enumerate(pain_names):
        query = urllib.urlencode({'q': name + ' stung', 'count': '100'})
        url = "https://api.twitter.com/1.1/search/tweets.json?" + query
        print url

        parameters = []
        response = twitterreq(url, "GET", parameters)
        # If the rate limit has been exceeded, we will get garbage data.
        # Make sure to raise an error in this case!
        calls_left = int(response.info().getheader('x-rate-limit-remaining'))
        if calls_left == 0:
            raise Exception("Rate Limit Exceeded! (Wait 15 minutes)")

        # The response for Twitter search is a single-line response,
        # namely a json string with a search_metadata field and statuses field.
        # The latter contains a list of tweet objects, which we want.
        lines = response.readlines()
        assert len(lines) == 1 # (unlike the streaming API)
        tweets[name] = json.loads(lines[0])['statuses']

    print "\nAll tweets have been grabbed!"
    return tweets

def twitterreq(url, method, parameters):
    "Create, sign, and open a twitter api request."
    req = oauth.Request.from_consumer_and_token(oauth_consumer,
        token=oauth_token, http_method=http_method,
        http_url=url, parameters=parameters)

    req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)
    headers = req.to_header()

    if http_method == "POST":
        encoded_post_data = req.to_postdata()
    else:
        encoded_post_data = None
        url = req.to_url()

    opener = urllib2.OpenerDirector()
    opener.add_handler(http_handler)
    opener.add_handler(https_handler)

    response = opener.open(url, encoded_post_data)

    return response



def get_tweets_scraped(pain_names):
    """Scrape tweets from twitter search results for terms in names.
        Return a dict with key=pain_name, val=list of tweets.

        This function is meant to produce output that 
        superficially mirrors the output of get_tweets_api.
        Although we don't get full Twitter API tweet objects
        as in get_tweets_api, each tweet in the list here is a dictionary
        with a 'text' field.
        If in the future we need to exploit additional features of
        tweets, further mirroring must be done.
    """

    # Twitter search results load dynamically on scroll-down.
    # If we want more search results, we will have to use Selenium
    # or some other method. For now, we just grab the first results.

    tweets = {}
    print "Scraping tweets..."

    # When debugging, don't run this on all pain_names.
    # for i, pain in enumerate(pain_names[:2]):
    for i, pain in enumerate(pain_names):

        # Simulate human user
        sleep(1 + uniform(-0.3, 0.3))

        query = urllib.urlencode({'q': pain + ' stung'})
        url = 'https://twitter.com/search?' + query
        print url

        soup = get_soup(url)
        if soup is None: 
            continue

        tweet_items = soup.select('p.js-tweet-text.tweet-text')
        tweet_texts = [tweet.get_text() for tweet in tweet_items]

        print "Num tweets: %d" % len(tweet_texts)

        tweets[pain] = [ {'text': text} for text in tweet_texts]

    print "\nAll tweets have been scraped!"
    return tweets

def get_soup(url):
    "Return a BeautifulSoup object for the given url."

    # Some pages will block requests from the python default User-Agent.  
    # A solution is to set the headers that go along with the request.
    header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

    req = urllib2.Request(url, headers=header)

    try:
        page = urllib2.urlopen(req, timeout=10)
        # print page.read()
        return BeautifulSoup(page)
    except: # If we have ANY error retrieving the page, move on.
        print "No soup for you!"
        return None    



if __name__ == '__main__':
    main()