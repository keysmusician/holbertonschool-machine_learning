#!/usr/bin/env python3
""" Displays the number of launches per SpaceX rocket. """


if __name__ == '__main__':
    import requests

    base_URL = 'https://api.spacexdata.com/'

    launch_counts = dict()

    launches = requests.get(base_URL + 'v4/launches/').json()
    for launch in launches:
        rocket = requests.get(
            base_URL + 'v4/rockets/' + launch['rocket']).json()

        try:
            launch_counts[rocket['name']] += 1
        except KeyError:
            launch_counts[rocket['name']] = 1

    sorted_launch_counts = sorted(
        launch_counts.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True
    )

    for rocket, launches in sorted_launch_counts:
        print("{}: {}".format(rocket, launches))
