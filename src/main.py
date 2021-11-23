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
    'drives', 'rebounds', 'speed&distance', 'fga',
    'og_injuries'
]

debug = False
# ----------

# generate and save required datasets:
# tracking and injuries
if len(listdir(processedDataDir)) != len(stats):
    if outputFullStats(stats, seasons):
        print('-- Generated datasets! --')

# import pre-pre-processed player tracking data
drives = importData(path.join(processedDataDir, 'drives.csv'))
fga = importData(path.join(processedDataDir, 'fga.csv'))
rebounds = importData(path.join(processedDataDir, 'rebounds.csv'))
speed_distance = importData(path.join(processedDataDir, 'speed&distance.csv'))
injuries = importData(path.join(processedDataDir, 'injuries.csv'))
print('-- Imported datasets! --')

# Exploratory Data Analysis
if debug:
    # 1 -- Teams with the most injuries
    # !! - Limited to the top 'limit' most injured teams
    topInjuriesTeams = seriesToFrame(
        injuries['Team'].value_counts(),
        ['Team', '# of Injuries']
    )
    plotHistogram(
        topInjuriesTeams,
        ['Teams', '# of events'],
        limit=10
    )

    # 2 -- Players with the most injuries
    # !! - Limited to the top 'limit' most injured players
    topInjuriesPlayers = seriesToFrame(
        injuries['Player'].value_counts(),
        ['Player', '# of Injuries']
    )
    plotHistogram(
        topInjuriesPlayers,
        topInjuriesPlayers.columns,
        limit=10
    )

    # 3 -- Injuries per player

# 4 -- Injuries per year
topInjuriesYear = seriesToFrame(
    injuries['Year'].value_counts(),
    ['Year', '# of Injuries']
).sort_values('Year').reset_index(drop=True)
plotLineGraph(
    topInjuriesYear,
    topInjuriesYear.columns,
)


print('Done')
