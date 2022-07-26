#!/usr/bin/env python3
""" Defines `sentientPlanets` """
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species.
    """
    SWAPI_URL = 'https://swapi-api.hbtn.io/api/species/'

    sentient_host_planets = set()

    response = requests.get(SWAPI_URL)
    while (response.ok):
        response_JSON = response.json()
        for species in response_JSON.get('results', []):
            if species.get('designation') == 'sentient':
                try:
                    if requests.get(species.get('homeworld')).json()['name'] == 'Rodia':
                        print('Rodia')
                    sentient_host_planets.add(
                        requests.get(species.get('homeworld')).json()['name'])
                except (
                    requests.exceptions.MissingSchema,
                    requests.JSONDecodeError
                ) as error:
                    print(species)
                    print(error)
                    continue

        next_page = response_JSON.get('next')

        if next_page is None:
            break

        response = requests.get(next_page)

    return sentient_host_planets
