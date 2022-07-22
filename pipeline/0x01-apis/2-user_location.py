#!/usr/bin/env python3
""" Prints the location of a specific GitHub user. """


if __name__ == '__main__':
    import requests
    import sys

    URL = sys.argv[1]

    response = requests.get(URL)

    if response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        print('Reset in {} min'.format(response.headers['X-RateLimit-Reset']))
    elif response.ok:
        print(response.json()['location'])
