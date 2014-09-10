import json
import socket
import oauth2 as oauth
import urllib, urllib2
import os


# Twitter API setup
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
    tweets = get_tweets(names)

    # Save the tweets and pains files for easy reloading later.
    with open('../data/outputs/twitter_results.txt', 'w') as outfile1:
        json.dump(tweets, outfile1)
    with open('../data/outputs/pains.txt', 'w') as outfile2:
        json.dump(pains, outfile2)



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
        % (len(pains), len(data)-1)
    )
    return pains


def get_tweets(pain_names):
    """Get twitter search results corresponding to the pain names.
        Return a dict with key=pain_name, val=list of tweet objects.
    """
    tweets = {}
    print "Grabbing tweets..."

    # When debugging, don't run this on all pain_names.
    # You can easily hit Twitter's rate limit within a 15-minute window.
    for i, name in enumerate(pain_names[:10]):
    # for i, name in enumerate(pain_names):
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
    "Create, sign, and open a twitter request."
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


if __name__ == '__main__':
    main()