#!/usr/bin/env python3
""" Defines `availableShips` """
import requests
import string


def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given number of passengers.

    passengerCount: The number of passengers a ship can hold.
    """
    SWAPI_URL = 'https://swapi-api.hbtn.io/api/starships/'

    capacious_starships = []

    response = requests.get(SWAPI_URL)

    while (response.ok):
        response_JSON = response.json()

        for starship in response_JSON.get('results', []):
            try:  # Attempt to coerce the 'passengers' string into an integer:
                passenger_capacity = int(
                    ''.join(
                        [
                            character for character in
                            starship.get('passengers') if character in
                            string.digits
                        ]
                    )
                )
            except ValueError:
                continue
            if passenger_capacity >= passengerCount:
                capacious_starships.append(
                    starship.get('name', 'MISSING NAME'))

        next_page = response_JSON.get('next')

        if next_page is None:
            break

        response = requests.get(next_page)

    return capacious_starships
