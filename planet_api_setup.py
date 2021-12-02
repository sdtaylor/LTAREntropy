import os
from requests.auth import HTTPBasicAuth

from planet import api

""" 
A single spot to config  the api key
"""

#------------------------
# Planet API
api_key = ''
PLANET_API_KEY = os.environ.get('PL_API_KEY', api_key)

planet_client = api.ClientV1(api_key=PLANET_API_KEY)
# set up requests to work with api
planet_auth = HTTPBasicAuth(PLANET_API_KEY, '')