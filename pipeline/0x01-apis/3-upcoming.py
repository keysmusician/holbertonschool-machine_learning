#!/usr/bin/env python3
""" Displays information about the upcoming SpaceX launch. """


if __name__ == '__main__':
    import requests

    base_URL = 'https://api.spacexdata.com/'

    response = requests.get(base_URL + 'v5/launches/latest')

    if response.ok:
        latest_launch = response.json()

        launch_name, date, rocket_ID, launchpad_ID = (
            latest_launch['name'],
            latest_launch['date_local'],
            latest_launch['rocket'],
            latest_launch['launchpad'],
        )

        rocket_name = requests.get(
            base_URL + 'v4/rockets/' + rocket_ID).json()['name']

        launchpad = requests.get(
            base_URL + 'v4/launchpads/' + launchpad_ID).json()

        launchpad_name, launchpad_locality = (
            launchpad['name'],
            launchpad['locality']
        )

        print('{} ({}) {} - {} ({})'.format(
            launch_name,
            date,
            rocket_name,
            launchpad_name,
            launchpad_locality
        ))
