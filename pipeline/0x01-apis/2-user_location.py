#!/usr/bin/env python3
""" Prints the location of a specific GitHub user. """


if __name__ == '__main__':
    from datetime import datetime
    import requests
    import sys

    URL = sys.argv[1]

    response = requests.get(URL)

    if response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        minutes_until_reset = int(
            (
                datetime.fromtimestamp(
                    int(response.headers['X-RateLimit-Reset']))
                - datetime.now()
            ).total_seconds() / 60
        )
        print('Reset in {} min'.format(minutes_until_reset))
    elif response.ok:
        print(response.json()['location'])
