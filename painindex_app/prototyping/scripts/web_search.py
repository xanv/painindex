"""Scrape google for relevant search results.
In previous versions, we had grabbed urls of search results
and scraped each such page. Here we simply take the snippet
of text that google presents along with each search result.
This is about a tweet-sized amount of text and guaranteed
to be the relevant part of the search result.
"""

import json
import socket
import oauth2 as oauth
import urllib, urllib2, cookielib
from bs4 import BeautifulSoup
from time import sleep
from random import uniform
import os



def main():
    "All configuration (filepaths, debug=True/False) should be done here."

    pains = import_pains('../data/inputs/schmidt_ratings_simple.csv')
    names = pains.keys()

    # Dict with key = pain name, val = list of search result texts
    search_results = get_search_results(names, debug=False)


    with open('../data/outputs/pains.txt', 'w') as outfile1:
        json.dump(pains, outfile1)
    with open('../data/outputs/search_results.txt', 'w') as outfile2:
        json.dump(search_results, outfile2)


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


def get_search_results(pain_names, debug=False):
    """Scrape pain name search results from google.
        Return a dict with key=pain_name, val=list of results.
        Each result is a simple text string. It is the text underneath
        the link of a google search result.

        If debug=True, we only scrape one result.
    """
    search_results = {}

    print "Scraping search results..."
    print "\nDEBUG STATUS: debug = %s" % debug
    num_pains_to_scrape = 1 if debug else len(pain_names)

    for i, pain in enumerate(pain_names[:num_pains_to_scrape]):
        # Simulate human user
        sleep(1 + uniform(-0.3, 0.3))

        # We enclose the name of the pain in quotes, and also "stung".
        # num = number of search results per page. We don't really
        #   have full control over this; google allows num=100 though.
        query = urllib.urlencode({
            'q': '"%s" "%s"' % (pain, 'painful'),
            'num': 100
        })
        url = 'https://www.google.com/search?' + query
        print url

        soup = get_soup(url)
        if soup is None: 
            continue

        search_items = soup.select('span.st')
        search_texts = [item.get_text() for item in search_items]

        print "Num results: %d" % len(search_texts)

        # We store each search result as a dictionary with a text field
        # containing its text.  We will stick with this general format so we
        # can save other details of the search results if desired, without
        # changing the downstream code.
        search_results[pain] = [ {'text': text} for text in search_texts]

    # print search_results

    print "\nAll search results have been scraped!"
    return search_results

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