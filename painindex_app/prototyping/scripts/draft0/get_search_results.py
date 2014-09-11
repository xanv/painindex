"Scrape google search pages for urls to articles about each painsource."

import urllib2, cookielib
from bs4 import BeautifulSoup
import json
import socket
from time import sleep
from random import uniform

 

def get_search_results(num_results=100, sleep_time=1, testing=False):
    """Return a dictionary with key: pain name (e.g. 'yellow jacket'),
        val[0]: google search url
        val[1]: list of [link text, url] for each search result on page.

        num_results is the number of search results to get.
        sleep_time is the average sleep time in seconds between requests.
        If testing = True, we only search for 1 pain.
    """

    search_results = {}
    pains = get_pains('../data/inputs/schmidt_ratings1.txt')
    url_head = ('https://www.google.com/search?num=' 
        + str(num_results) + '&q=bug+sting+'
    )

    if testing == True: 
        num_pains = 1
    else:
        num_pains = len(pains)

    for pain in pains[:num_pains]:
        # Randomize sleep time to simulate human
        sleep(sleep_time + uniform(-0.3, 0.3))

        query_url = url_head + pain.replace(' ', '+')
        print "On page:", query_url
        soup = get_soup(query_url)

        if soup is None: continue

        items = soup.select('h3.r > a')
        # Grab the link text and url of each result
        items = [[item.text, item['href']] for item in items]
        
        search_results[pain] = [query_url, items]
    
    return search_results


def get_pains(filename):
    """Return a list of pain names from a txt file.
        The txt file must be comma-delimited with the pain name as 
        the first element of each line.
    """
    pains = open(filename, 'r').read().strip().split('\n')
    return [line.split(',')[0] for line in pains]


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

    # urllib2.urlopen(url) would work if we didn't need these headers.
    # Instead, we generate a Request object to open.
    req = urllib2.Request(url, headers=header)

    try:
        page = urllib2.urlopen(req, timeout=10)
        # print page.read()
        return BeautifulSoup(page)
    except:
        # If we have ANY error retrieving the page, move on.
        print "No soup for you!"
        return None
    # except urllib2.HTTPError, e:
    #     print 'HTTPError, no soup for you!'
    #     return None
    # except urllib2.URLError, e:
    #     print 'Bad url or timeout, no soup for you!'
    #     return None
    # except socket.timeout, e:
    #     print "Timeout error, no soup for you!"
    #     return None





if __name__ == '__main__':

    search_results = get_search_results(100) #get_search_results(testing=True)

    # print search_results

    with open('../data/outputs/search_results.txt', 'w') as outfile:
        json.dump(search_results, outfile)

