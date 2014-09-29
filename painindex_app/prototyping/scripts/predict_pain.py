"""Scrape google for relevant search results.
This is like web_search but specifically for an input list of unrated pains.
This is just for a quick test of how well the trained model does on qualitatively
different types of data.
"""

from web_search import *


def main():
    "All configuration (filepaths, debug=True/False) should be done here."

    with open('../data/inputs/unrated_pains.txt', 'r') as f:
        pains = f.read().strip().split('\n')

    search_results = get_search_results(pains, debug=False)

    with open('../data/outputs/search_results_unrated.txt', 'w') as outfile:
        json.dump(search_results, outfile)


if __name__ == '__main__':
    main()