#!/usr/bin/env python3
""" Defines `sentientPlanets` """
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species.
    """
    SWAPI_URL = 'https://swapi-api.hbtn.io/api/species/'

    sentient_host_planets = []

    response = requests.get(SWAPI_URL)
    while (response.ok):
        response_JSON = response.json()
        for species in response_JSON.get('results', []):
            if 'sentient' in (
                species.get('designation'),
                species.get('classification')
            ):
                # Get and append homeworld name
                try:
                    sentient_host_planets.append(
                        requests.get(species.get('homeworld')).json()['name'])
                except requests.exceptions.MissingSchema:
                    continue

        next_page = response_JSON.get('next')

        if next_page is None or not next_page:
            break

        response = requests.get(next_page)

    return sentient_host_planets
