"""
This is the main file of 
the predictor.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from utils import *

# -- globals
seasons = [
    '1314', '1415', '1516', '1617',
    '1718', '1819', '1920', '2021'
]

stats = [
    'drives', 'rebounds', 'speed&distance', 'fga'
]

# ----------

# generate required datasets
outputFullStats(stats, seasons)


print('Done')
