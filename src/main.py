"""
This is the main file of 
the predictor.

@ Manuel Fideles (2018282990)
@ Alexandre Cortez Santos (???)
"""

from dependencies import *
from utils import *
from tests import *


# -- globals
seasons = [
    '1314', '1415', '1516', '1617',
    '1718', '1819', '1920', '2021'
]

stats = [
    'drives', 'rebounds', 'speed&distance', 'fga'
]

# ----------

# generate and save required datasets:
# tracking ...
if not len(listdir(processedDataDir)):
    outputFullStats(stats, seasons)


# ... and injuries
exportData(
    trimInjuries(
        importData(path.join(rawDataDir, 'og_injuries.csv'))
    ),
    path.join(processedDataDir, 'injuries.csv')
)

# import pre-pre-processed player tracking data
drives = importData(path.join(processedDataDir, 'drives.csv'))
fga = importData(path.join(processedDataDir, 'fga.csv'))
rebounds = importData(path.join(processedDataDir, 'rebounds.csv'))
speed_distance = importData(path.join(processedDataDir, 'speed&distance.csv'))
injuries = importData(path.join(processedDataDir, 'injuries.csv'))

# Teams vs # of Injuries
plotHistogram(
    seriesToFrame(
        injuries['Team'].value_counts(),
        ['Team', '# of Injuries']
    )
)

print('Done')
